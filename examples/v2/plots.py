import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

emilia_results = pd.read_csv("results_emilia.csv")
librilatest_results = pd.read_csv("results_librilatest.csv")
libritts_results = pd.read_csv("results_libritts.csv")
myst_results = pd.read_csv("results_myst.csv")
torgo_results_dys = pd.read_csv("results_torgo_dys.csv")
torgo_results_ctr = pd.read_csv("results_torgo_ctr.csv")

# concatenate all results
all_results = pd.concat(
    [
        emilia_results,
        librilatest_results,
        libritts_results,
        myst_results,
        torgo_results_dys,
        torgo_results_ctr,
    ]
)


sns.barplot(x="benchmark_name", y="score", hue="dataset", data=all_results)
plt.show()
