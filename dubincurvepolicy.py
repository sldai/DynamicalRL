import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../DynamicalEnvs/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../CurvesGenerator/")
from dubins_path import calc_dubins_path
import matplotlib.pyplot as plt
from dubin import DubinEnv, d
import numpy as np
def dubin_curve():
    env = DubinEnv()
    env.reset()
    path = calc_dubins_path(*env.state_[:3], *env.goal_[:3], curv=np.tanh(env.max_phi)/d)
    env.render(t=None)
    plt.plot(path.x, path.y, linewidth=1)
    plt.show()

if __name__ == "__main__":
    dubin_curve()