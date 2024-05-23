import copy
import random
import timeit
import matplotlib.pyplot as plt
from summa.preprocessing import textcleaner
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from textblob import Word
from ngram import Ngram
import numpy
import pandas as pd
import sys

# nltk.download("stopwords")
# nltk.download("wordnet")

_DATA_SIZE_ARR = [25, 100, 250]


class EntropyModel:

    def __init__(self, num_iterations, ngram_size):
        self.ngram_entropies = None
        self.sentences_dict = None
        self.sentences_fixed = None
        self.sentences_types = None
        self.num_iterations = num_iterations
        self.ngram_size = ngram_size
        self.entropies_elitism = None
        self.final_ngrams = None
        self.final_time = None
        self.elitism_final_time = None
        self.elitism_ngrams = None

    """
    This piece of code create the n-grams with its corresponding size
    and create a dictionary called ngram_entropies that collects a lot
    of data to be analized.
    """

    def run_model(self, df, data_type_func, data_size=None):
        if not data_size:
            data_size = df.shape[0]
        entropy_model = {}
        sentences_dict, sentences_fixed, sentences_types = data_type_func(df, data_size)
        ngram_entropies = {}

        print("# iterations: ", self.num_iterations)
        final_entropy = None
        final_ngrams = None
        entropies = None
        entropies_elitism = None
        print("Gram size: " + str(self.ngram_size))
        ngrams = [Ngram(self.ngram_size) for _ in range(4)]
        sentences = sentences_fixed[:]
        random.shuffle(sentences)
        for s_idx in range(len(sentences)):
            i = s_idx % 4
            ngrams[i].train(sentences[s_idx], s_idx)
        start = timeit.default_timer()
        (
            final_entropy,
            final_ngrams,
            entropies,
            entropies_elitism,
            elitism_final_time,
            elitism_ngrams,
        ) = self.compute_global_entropy(ngrams, self.num_iterations, sentences, start)
        stop = timeit.default_timer()
        final_time = stop - start
        # if num_iterations not in ngram_entropies.keys():
        #    ngram_entropies[num_iterations] = {
        #        n_size: {"entropies": "", "entropies_elitism": "", "ngrams": ""}
        #    }
        # if n_size not in ngram_entropies[num_iterations].keys():
        #    ngram_entropies[num_iterations][n_size] = {
        #        "entropies": "",
        #        "entropies_elitism": "",
        #        "ngrams": "",
        #    }
        self.ngram_entropies = entropies
        self.entropies_elitism = entropies_elitism
        self.final_ngrams = final_ngrams
        self.final_time = final_time
        self.elitism_final_time = elitism_final_time
        self.elitism_ngrams = elitism_ngrams
        self.sentences_dict = sentences_dict
        self.sentences_fixed = sentences_fixed
        self.sentences_types = sentences_types

    def compute_global_entropy(self, ngrams_arr, num_iterations, sentences, start_time):
        """
        This function calculates the global entropy of 4 ngrams together
        and finds the combination of ngram with best Entropy using elistism.

        The ngrams that are chose to make the sentence exchange are selected
        randomly. The same applies to the sentences that are selected to be
        exchanged.
        """
        global_entropy = sum([n.entropy() for n in ngrams_arr]) / 4
        final_ngrams = copy.deepcopy(ngrams_arr)
        entropies = []
        entropies_elitism = []
        elitism_it = 0
        global_it = 0
        elitism_final_time = 0
        elitism_ngrams = []
        while global_it < num_iterations:
            curr = random.choice(ngrams_arr)
            other = random.choice(ngrams_arr)
            if curr != other:
                if len(curr.sentences_idx) > 2:
                    sentence, sentence_idx_in_arr = curr.incremental_training(
                        other, sentences
                    )
                    current_global_entropy = sum([n.entropy() for n in ngrams_arr]) / 4
                    entropies.append((global_it, current_global_entropy))
                    global_it += 1
                    if current_global_entropy < global_entropy:  # Elitism
                        final_ngrams = copy.deepcopy(ngrams_arr)
                        global_entropy = current_global_entropy
                        entropies_elitism.append((elitism_it, global_entropy))
                        elitism_ngrams.append((elitism_it, final_ngrams))
                        elitism_it += 1
                        end_time = timeit.default_timer()
                        elitism_final_time = end_time - start_time
                    else:
                        curr.rollback_incremental_training(
                            other, sentence, sentence_idx_in_arr
                        )
                        elitism_it += 1
        return (
            global_entropy,
            final_ngrams,
            entropies,
            entropies_elitism,
            elitism_final_time,
            elitism_ngrams,
        )

    def print_subplots_entropy_curve(self, type, xlim=None):

        time_type = "time" if type == "entropies" else "elitism_time"
        entropies = (
            self.ngram_entropies if type == "entropies" else self.entropies_elitism
        )
        # fig, axs = plt.subplots(2, 2, layout="constrained")
        # fig.suptitle("Entropy Curve for {}-gram".format(self.ngram_size))
        # axes = axs.flat
        # for idx, gram_size in enumerate(entropies):
        #    axes[idx].plot(*zip(*entropies))
        #    axes[idx].set_xlabel("Iterations", fontsize=10)
        #    axes[idx].set_ylabel("Entropy", fontsize=10)
        #    axes[idx].set_title(
        #        "{}-gram / time: {} seconds".format(
        #            gram_size,
        #            round(self.final_time, 3),
        #        ),
        #        fontsize=10,
        #    )
        #    axes[idx].set_xlim(0, xlim)
        x = [it[0] for it in entropies]
        y = [entropy[1] for entropy in entropies]
        plt.title("Entropy Curve for {}-gram".format(self.ngram_size))
        plt.plot(x, y)
        plt.xlabel("Iterations")
        plt.ylabel("Entropy")
        plt.show()

    def clear_text(self, text):
        """
        text: string

        Return a SyntacticUnit object

        Useful attributes are: su.text and su.token

        """

        return textcleaner.clean_text_by_sentences(text)

    def build_analysis_df(ngram_entropies, sentences):
        """
        This piece of code creates a pandas Datafrase to looks at the data in tabular form
        """
        arr_sentence_bucket = []
        for it_size in ngram_entropies.keys():
            for gram_size in ngram_entropies[it_size].keys():
                for idx, ngram in enumerate(
                    ngram_entropies[it_size][gram_size]["ngrams"]
                ):
                    for sentence_idx in ngram.sentences_idx:
                        arr_sentence_bucket.append(
                            (
                                sentences[sentence_idx],
                                idx,
                                gram_size,
                                it_size,
                                ngram_entropies[it_size][gram_size]["time"],
                                ngram_entropies[it_size][gram_size]["elitism_time"],
                                ngram_entropies[it_size][gram_size][
                                    "entropies_elitism"
                                ][-1][1],
                                ngram_entropies[it_size][gram_size][
                                    "entropies_elitism"
                                ][-1][0],
                            )
                        )
        df_sentences_buckets = pd.DataFrame(
            arr_sentence_bucket,
            columns=[
                "sentence",
                "bucket",
                "n_gram_size",
                "iteration_size",
                "time",
                "elitism_time",
                "best_entropie",
                "best_entropie_iteration",
            ],
        )
        return df_sentences_buckets

    def build_sentences_data(self, df_data, size=25):
        sentences_dict = {
            "objective": [],
            "methodology": [],
            "results": [],
            "motivation": [],
        }
        sentences_fixed = []
        sentences_types = {}
        for sentence_type in sentences_dict:  # Loop over dictionary keys
            for txt in df_data[sentence_type]:  # Get sentece_type text from db
                clean_txt = self.clear_text(txt)
                for sentence in clean_txt:  # Get each clean sentences from the object
                    sentences_dict[sentence_type].append(sentence)

        for sentence_type in sentences_dict:  # Loop over dictionary keys
            for sentence in sentences_dict[sentence_type][:size]:
                sentences_fixed.append(sentence.token)
                sentences_types[sentence.token] = (sentence_type, sentence.text)
        return sentences_dict, sentences_fixed, sentences_types

    def save_to_file(path, data):
        import pickle

        file = open(path, "xb")
        pickle.dump(data, file)
        file.close()

    def build_paragraphs_data(self, df_data, size=25):
        paragraph_dict = {
            "objective": [],
            "methodology": [],
            "results": [],
            "motivation": [],
        }
        paragraph_fixed = []
        paragraph_types = {}

        for paragraph_type in paragraph_dict:  # Loop over dictionary keys
            for txt in df_data[paragraph_type]:  # Get sentece_type text from db
                txt = self.clear_text_paragraph(txt)
                paragraph_dict[paragraph_type].append(txt)

        for paragraph_type in paragraph_dict:  # Loop over dictionary keys
            for paragraph in paragraph_dict[paragraph_type][:size]:
                paragraph_fixed.append(paragraph)
                paragraph_types[paragraph] = (paragraph_type, paragraph)
        return paragraph_dict, paragraph_fixed, paragraph_types

    def clear_text_paragraph(self, txt):
        txt = txt.lower().strip().replace("\n", " ").replace("\r", " ")
        txt = re.sub(r"[^a-zA-Z\']", " ", txt)
        txt = re.sub(r"[^\x00-\x7F]+", "", txt)
        txt = re.sub(r"http\S+", "", txt)
        tokenizer = RegexpTokenizer(r"\w+")
        txt = tokenizer.tokenize(txt)
        txt = [word for word in txt if txt not in stopwords.words("english")]
        txt = " ".join(txt)
        txt = " ".join([Word(word).lemmatize() for word in txt.split()])
        return txt

    def analyze_ngram(self, ngram):
        types_arr = []
        for sentence_idx in ngram.sentences_idx:
            types_arr.append(self.get_sentence_type(sentence_idx))

        unique_types = numpy.unique(types_arr, return_counts=True)
        unique_types_dict = dict(zip(*unique_types))
        unique_types_dict["entropy"] = ngram.entropy()
        return unique_types_dict

    def run_same_elitism_bucket_analysis_by_idx(self, idx):
        ngrams = self.elitism_ngrams[idx]  # 4 ngrams por idx
        analyzed_ngrams = []
        for ngram in ngrams[1]:
            analyzed_ngrams.append(self.analyze_ngram(ngram))
        return analyzed_ngrams

    def run_entropy_elitism_analysis(self):
        first = 0
        mid = int(len(self.elitism_ngrams) / 2)
        last = len(self.elitism_ngrams) - 1
        analysis_dict = {}
        for idx in [first, mid, last]:
            types_entropies = self.run_same_elitism_bucket_analysis_by_idx(idx)
            ngrams_entropy_avg = (
                sum([entropy["entropy"] for entropy in types_entropies]) / 4
            )
            analysis_dict[idx] = types_entropies, ngrams_entropy_avg

        analysis_dfs_arr = []
        for idx in analysis_dict:
            analysis_dfs_arr.append(self.build_analysis_dfs(analysis_dict[idx]))
        return analysis_dfs_arr

    def build_analysis_dfs(self, analysis_data):
        cols = [
            "objective",
            "methodology",
            "results",
            "motivation",
            "entropy",
            "global_entropy",
        ]
        df = pd.DataFrame.from_dict(analysis_data[0])
        df["global_entropy"] = analysis_data[1]
        df = df.fillna(0)
        return df[cols]

    def get_sentence_type(self, sentence_idx):
        return self.sentences_types[self.sentences_fixed[sentence_idx]][0]
