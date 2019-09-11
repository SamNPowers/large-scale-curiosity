from experiment_config_base import ExperimentConfigBase, UnknownExperimentConfigEntry


class CuriosityExperimentConfig(ExperimentConfigBase):
    def __init__(self, config_path, output_dir):
        super().__init__(config_path, output_dir)

        # Defaults here. These get overwritten by the config json

        # Large scale boolean and configs. Should align exactly with the command line parameters in large_scale_curiosity_pytorch's run.py
        self.envs_per_process = 128
        self.nsteps_per_seg = 128
        self.feat_learning = "none"
        self.env = "BreakoutNoFrameskip-v4"

        # Params used within the large_scale_curiosity codebase
        self.use_discrim_loss_as_curiosity = False
        self.dynamics_loss_off = False
        self.discrim_learning_rate = 1e-5
        self.generator_learning_rate = 1e-5
        self.discrim_based_on_pred = False

    def _load_single_experiment(self, config_json):
        self.envs_per_process = config_json.pop('envs_per_process', self.envs_per_process)
        self.nsteps_per_seg = config_json.pop('nsteps_per_seg', self.nsteps_per_seg)
        self.feat_learning = config_json.pop('feat_learning', self.feat_learning)
        self.env = config_json.pop('env', self.env)

        self.use_discrim_loss_as_curiosity = config_json.pop('use_discrim_loss_as_curiosity', self.use_discrim_loss_as_curiosity)
        self.dynamics_loss_off = config_json.pop('dynamics_loss_off', self.dynamics_loss_off)
        self.discrim_learning_rate = config_json.pop('discrim_learning_rate', self.discrim_learning_rate)
        self.generator_learning_rate = config_json.pop('generator_learning_rate', self.generator_learning_rate)
        self.discrim_based_on_pred = config_json.pop('discrim_based_on_pred', self.discrim_based_on_pred)

        if len(config_json) > 0:
            raise UnknownExperimentConfigEntry("JSON still had elements after parsing: {}".format(config_json.keys()))
