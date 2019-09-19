# Hacky file just for graph purposes.

import pickle
import matplotlib.pyplot as plt

#with open( "/Users/powerss/Git/large-scale-curiosity-github/experiment_configs/tmp/-1_debug/4/env0_0.pk", "rb" ) as pickled_file:
with open( "/Users/powerss/Git/large-scale-curiosity-github/experiment_configs/tmp/-1_debug/5/env0_0.pk", "rb" ) as pickled_file:
    env_data = pickle.load(pickled_file)

acs = env_data['acs']
int_rew = env_data['int_rew']

action_rewards = [[] for _ in range(10)]  # 10 actions

for id, ac in enumerate(acs):
    action_rewards[ac].append(int_rew[id])

for i in range(len(action_rewards)):
    plt.plot(action_rewards[i])
pass