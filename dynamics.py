import numpy as np
import tensorflow as tf

from auxiliary_tasks import JustPixels
from utils import small_convnet, flatten_two_dims, unflatten_first_dim, getsess, unet


class Dynamics(object):
    def __init__(self, auxiliary_task, predict_from_pixels, experiment_config, feat_dim=None, scope='dynamics'):
        self.experiment_config = experiment_config
        self.scope = scope
        self.auxiliary_task = auxiliary_task
        self.hidsize = self.auxiliary_task.hidsize
        self.feat_dim = feat_dim
        self.obs = self.auxiliary_task.obs
        self.last_ob = self.auxiliary_task.last_ob
        self.ac = self.auxiliary_task.ac
        self.ac_space = self.auxiliary_task.ac_space
        self.ob_mean = self.auxiliary_task.ob_mean
        self.ob_std = self.auxiliary_task.ob_std
        if predict_from_pixels:
            self.features = self.get_features(self.obs, reuse=False)
        else:
            self.features = tf.stop_gradient(self.auxiliary_task.features)

        self.out_features = self.auxiliary_task.next_features

        with tf.variable_scope(self.scope + "_loss"):
            self.loss, self.generator_loss, self.discriminator_loss, self.discriminator_reward = self.get_loss()

    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=self.feat_dim, last_nl=nl, layernormalize=False)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def train_discriminator(self, prev_state, action, true_state, pred_state, true_frac=0.5):
        """
        Takes in a set of previous states, actions, and both the true result and the predicted result.
        Creates a set of targets from these (basically randomly picking from both), then creating the discriminator from that.
        True_frac is the fraction of the combined list that is composed of true states.
        """
        batch_size = tf.shape(prev_state[:, 0])

        # Create a list of flags where 1 means use the true state.
        true_state_flags = tf.random_uniform(batch_size, minval=0, maxval=1) < true_frac
        combined_states = tf.where(true_state_flags, x=true_state, y=pred_state)
        targets = tf.where(true_state_flags, x=tf.ones(batch_size), y=tf.zeros(batch_size))  # Prefer to explicitly define rather than relying on bool conversion
        targets = tf.expand_dims(targets, axis=-1)

        predictions = self.create_discriminator(prev_state, action, combined_states)
        discrim_loss = tf.losses.sigmoid_cross_entropy(targets, predictions, reduction=tf.losses.Reduction.NONE)

        return predictions, discrim_loss

    def create_discriminator(self, prev_state, action, state):
        """
        Create the discriminator that takes in state features, and predicts whether they were generated or real.
        The targets are expected to be 1 if the state is a true state, and 0 if it is a generated state.
        """
        # Combine our features.
        # We will have a separate optimizer for the discriminator and the generator, so keeping the gradient here should be fine
        # TODO: make the gradient stopping more consistent. Right now basically prev_state and action should always be stopped, but current state is conditional.
        cat_features = tf.concat([tf.stop_gradient(prev_state),
                                  tf.stop_gradient(action),
                                  state], axis=-1)

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(cat_features, 512, activation=tf.nn.leaky_relu)  # TODO spowers: not just hardcoded. Also what's up with this leaky_relu? Is it PReLU? Where is the alpha...?
            x = tf.layers.dense(x, 512, activation=tf.nn.leaky_relu)  # TODO: architecture arbitrary
            x = tf.layers.dense(x, 1)

        return x

    def get_loss(self):
        ac = tf.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)

        def add_ac(x):
            return tf.concat([x, ac], axis=-1)

        with tf.variable_scope(self.scope):
            x = flatten_two_dims(self.features)
            x = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)

            def residual(x):
                res = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)
                res = tf.layers.dense(add_ac(res), self.hidsize, activation=None)
                return x + res

            for _ in range(4):
                x = residual(x)
            n_out_features = self.out_features.get_shape()[-1].value
            x = tf.layers.dense(add_ac(x), n_out_features, activation=None)
            x = unflatten_first_dim(x, sh)

        # Compute the loss that allows us to update the discriminator
        # Eventually this will be replaced with a loss for the dynamics model that attempts to exclude entropy
        discrim_predictions, discrim_loss = self.train_discriminator(prev_state=flatten_two_dims(self.features),
                                                                     action=ac,
                                                                     true_state=flatten_two_dims(tf.stop_gradient(self.out_features)),
                                                                     pred_state=flatten_two_dims(x))
        discrim_train_loss = tf.reduce_mean(unflatten_first_dim(discrim_loss, sh), -1)  # Really just removing the last 1. At the moment this just reflects symmetry with below.

        # There are two methods for getting the generator loss: one is to negate the discriminator (drive x features away from 0), and the other is to drive the x features to 1.
        if self.experiment_config.generator_loss_from_discrim_loss:
            generator_train_loss = -discrim_train_loss
        else:
            # Should just effectively ignore the ones to the "pred", because there are no gradients to update
            _, generator_loss = self.train_discriminator(prev_state=flatten_two_dims(self.features),
                                                         action=ac,
                                                         true_state=flatten_two_dims(x),
                                                         pred_state=flatten_two_dims(tf.stop_gradient(self.out_features)),  # Since true_frac is 1, none of these get used. The out_features are a fancy placeholder
                                                         true_frac=1.0)
            generator_train_loss = tf.reduce_mean(unflatten_first_dim(generator_loss, sh), -1)  # Really just removing the last 1. At the moment this just reflects symmetry with below.

        # Update the dynamics both to make the features match the input ones, and to satisfy the generator.
        dynamics_loss = tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, -1)

        if self.experiment_config.discrim_based_on_pred:
            discrim_input = tf.stop_gradient(x)
        else:
            discrim_input = tf.stop_gradient(self.out_features)

        # Now use the discriminator to evaluate the prediction that was made, and use that as the reward.
        discrim_pred_for_x = self.create_discriminator(prev_state=flatten_two_dims(self.features),
                                                       action=ac,
                                                       state=flatten_two_dims(discrim_input))
        discrim_pred_for_x = tf.sigmoid(discrim_pred_for_x)  # Since the discriminator output is logits for the loss function.
        discrim_pred_for_x = tf.reduce_mean(unflatten_first_dim(discrim_pred_for_x, sh), -1)  # Really just removing the last 1. At the moment this just reflects symmetry with above. TODO less obtuse

        eps = 1e-8
        # If discrim_pred is close to 1, the term in the log will be close to 0, reward will be inf, so add an eps.
        # If discrim pred is close to 0, the term in the log will be close to inf, so add an eps to the denom
        discrim_reward = -tf.log((1-discrim_pred_for_x + eps)/(discrim_pred_for_x + eps))

        # Invert the reward if we're using the predicted version
        if self.experiment_config.discrim_based_on_pred:
            discrim_reward = -discrim_reward

        if self.experiment_config.use_dynamics_in_discrim_reward:
            discrim_reward = dynamics_loss * discrim_reward

        return dynamics_loss, generator_train_loss, discrim_train_loss, discrim_reward

    def calculate_loss(self, ob, last_ob, acs):
        n_chunks = 8
        n = ob.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0, "There must be at least {} envs per process.".format(n_chunks)
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)

        if self.experiment_config.use_discrim_loss_as_curiosity:
            curiosity_loss = self.discriminator_reward
        else:
            curiosity_loss = self.loss

        return np.concatenate([getsess().run(curiosity_loss,
                                             {self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
                                              self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)


class UNet(Dynamics):
    def __init__(self, auxiliary_task, predict_from_pixels, feat_dim=None, scope='pixel_dynamics'):
        assert isinstance(auxiliary_task, JustPixels)
        assert not predict_from_pixels, "predict from pixels must be False, it's set up to predict from features that are normalized pixels."
        super(UNet, self).__init__(auxiliary_task=auxiliary_task,
                                   predict_from_pixels=predict_from_pixels,
                                   feat_dim=feat_dim,
                                   scope=scope)

    def get_features(self, x, reuse):
        raise NotImplementedError

    def get_loss(self):
        nl = tf.nn.leaky_relu
        ac = tf.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)
        ac_four_dim = tf.expand_dims(tf.expand_dims(ac, 1), 1)

        def add_ac(x):
            if x.get_shape().ndims == 2:
                return tf.concat([x, ac], axis=-1)
            elif x.get_shape().ndims == 4:
                sh = tf.shape(x)
                return tf.concat(
                    [x, ac_four_dim + tf.zeros([sh[0], sh[1], sh[2], ac_four_dim.get_shape()[3].value], tf.float32)],
                    axis=-1)

        with tf.variable_scope(self.scope):
            x = flatten_two_dims(self.features)
            x = unet(x, nl=nl, feat_dim=self.feat_dim, cond=add_ac)
            x = unflatten_first_dim(x, sh)
        self.prediction_pixels = x * self.ob_std + self.ob_mean

        assert False, "spowers has not yet piped the discriminator through here"

        return tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, [2, 3, 4]), x
