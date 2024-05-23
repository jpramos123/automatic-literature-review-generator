import matplotlib.pyplot as plt


class GenericPlot:

    def __init__(self):
        pass

    def plot_entropy(self, entropies_model, fig_name):

        fig, ax = plt.subplots()
        for model in entropies_model:
            x = [it[0] for it in model.entropies_elitism]
            y = [entropy[1] for entropy in model.entropies_elitism]
            ax.plot(x, y, label=f"{model.ngram_size}-gram")
        ax.legend()
        plt.title("Entropy Curves")
        plt.xlabel("Iterations")
        plt.ylabel("Entropy")
        plt.savefig(f"./figures/{fig_name}")
        plt.show()
