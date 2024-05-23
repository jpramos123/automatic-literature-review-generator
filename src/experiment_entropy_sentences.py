from model import EntropyModel
import pandas as pd
from plot import GenericPlot
import pickle


def main():
    df = pd.read_pickle("../data/emerald/data.pkl")
    models_arr = []
    for ngram_size in [2, 3, 4, 5]:
        entropy_model = EntropyModel(num_iterations=10000, ngram_size=ngram_size)
        entropy_model.run_model(df, entropy_model.build_sentences_data)
        models_arr.append(entropy_model)
        # entropy_model.print_subplots_entropy_curve('elitism')

    with open("./results/exp1-models.arr", "wb") as file:
        pickle.dump(models_arr, file)

    plot = GenericPlot()
    plot.plot_entropy(models_arr, "exp1_sentences.png")


if __name__ == "__main__":
    main()
