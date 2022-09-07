"""Visualize cell counts only"""

import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
from config import basepath

# load dataset
df = pd.read_csv(osp.join(basepath, "counts.csv"))


# create plot
plt.plot(df["count green"], label="green cells", color="green")
plt.plot(df["count red"], label="red cells", color="red")
plt.title("Absolute cell counts")
plt.xlabel("Frame")
plt.ylabel("Cell count")
plt.legend()
plt.savefig(osp.join(basepath, "cell_count.png"))
