from model import EntropyModel
import pandas as pd
from plot import GenericPlot
import pickle
import time
import gc
import copy
import sys


def main():

    dfs_arr = []
    for ngram_size in [2, 3, 4, 5]:
        file = open(f"./results/exp1-{ngram_size}-gram-model-paragraphs.arr", "rb")
        model = pickle.load(file)
        file.close()
        dfs_arr.append(model.run_entropy_elitism_analysis())

    file = open(f"./results/exp1-model-paragraphs.dfs", "wb")
    pickle.dump(dfs_arr, file)
    file.close()


def view_tables():

    file = open(f"./results/exp1-model-paragraphs.dfs", "rb")
    dfs_arr = pickle.load(file)
    for idx, dfs in enumerate(dfs_arr):
        print(f"{idx+2}-gram\n")
        for df in dfs:
            print(df.to_markdown() + "\n")


if __name__ == "__main__":
    start = time.time()
    # main()
    view_tables()
    end = time.time()
    print(end - start)
