# Hacky file just for graph purposes.

import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

env_data = defaultdict(list)

with open( "/Users/powerss/Git/large-scale-curiosity-github/experiment_configs/tmp/-1_debug/11/env0_0.pk", "rb" ) as pickled_file:
    # We may have appended multiple objects to the same file (one per episode); load them all.
    while True:
        try:
            loaded_data = pickle.load(pickled_file)
        except EOFError:
            print("File completely loaded")
            break
        else:
            for key, val in loaded_data.items():
                env_data[key].extend(val)

acs = env_data['acs']
int_rew = env_data['int_rew']

action_rewards = [[] for _ in range(10)]  # 10 actions

for id, ac in enumerate(acs):
    action_rewards[ac].append(int_rew[id])

for i in range(len(action_rewards)):
    plt.plot(action_rewards[i])
pass