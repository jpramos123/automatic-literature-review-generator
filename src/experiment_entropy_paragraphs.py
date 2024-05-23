from model import EntropyModel
import pandas as pd
from plot import GenericPlot
import pickle
import time
import gc
import copy
import sys


def main():

    df = pd.read_pickle("../data/emerald/data.pkl")

    models_arr = []
    for ngram_size in [2, 3, 4, 5]:
        entropy_model = EntropyModel(num_iterations=10000, ngram_size=ngram_size)
        entropy_model.run_model(df, entropy_model.build_paragraphs_data)
        with open(
            f"./results/exp1-{ngram_size}-gram-model-paragraphs.arr", "wb"
        ) as file:
            pickle.dump(entropy_model, file)
        # models_arr.append(copy.deepcopy(entropy_model))
        del entropy_model
        gc.collect()
        # entropy_model.print_subplots_entropy_curve('elitism')

    plot = GenericPlot()
    plot.plot_entropy(models_arr, "exp1_paragraph.png")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(end - start)
