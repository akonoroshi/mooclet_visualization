import pandas as pd
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

def plot_beta_dist(alphas: pd.Series, betas: pd.Series, stop: int, level: str, out_folder: str):
    for i in range(stop):
        alpha_curr = alphas[i]
        beta_curr = betas[i]
        x = np.arange (0.01, 1, 0.01)
        plt.plot(x, beta.pdf(x, alpha_curr, beta_curr))
    plt.title("Beta Distribution for First {} Students For {}".format(stop, level))
    plt.savefig("{}{}_{}.png".format(out_folder, level, stop))
    plt.close()

def plot_assignment_prob(ap_lists: list, levels: list, stop: int, factor: str, out_folder: str):
    for i in range(len(ap_lists)):
        plt.plot(ap_lists[i][:stop], ".-", label = levels[i])
    plt.legend()
    plt.xlabel("student index")
    plt.title("Assignment Probability for {} Factor For First {} Students".format(factor, stop))
    plt.ylabel("assignment probability")
    plt.savefig("{}assignment_probability_{}_{}.png".format(out_folder, factor, stop))
    plt.close()

def plot_success_failure(df: pd.DataFrame, versions: list, levels: list, stop: int, factor: str, out_folder: str):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] # if you have more levels, add on your own
    for i in range(len(versions)):
        plt.plot(df["failures_{}".format(versions[i])][0:stop], color=colors[i], marker="o", label="{} failures".format(levels[i]))
        plt.plot(df["successes_{}".format(versions[i])][0:stop], color=colors[i], marker= "*", label="{} successes".format(levels[i]))
    plt.ylabel("Count")
    plt.title("Number of Successes and Failures {} Factor For First {} Students".format(factor, stop))
    plt.xlabel("student index")
    plt.legend()
    plt.savefig("{}success_failure_{}_{}.png".format(out_folder, factor, stop))
    plt.close()