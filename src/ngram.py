import random
import numpy as np

BEGGIN_TOKEN = "<s> "
END_TOKEN = " </s>"
MISSING_TOKEN = "<UNK>"


class Ngram:
    """
    ** Trying to apply some knowledge from Fluent Python book **
    Lazy way of programming (using generators)

    """

    def __init__(self, n_size):
        self.n_size = n_size  # size of the n-gram
        self.gram = {}  # array with dicts, each index is a dict which is a n-gram
        self.sentences_idx = []  # array with sentences indexes

    def _format_text(self, text, n_idx):
        text = (BEGGIN_TOKEN * n_idx) + text + (END_TOKEN * n_idx)
        return text.split()

    def train(self, text, text_idx):
        """
        Create an array of n-grams [{1-gram}, {2-gram}, ... , {n-gram}]
        """

        self.sentences_idx.append(text_idx)
        # if not self.grams_arr:
        #  self.grams_arr = [{} for _ in range(self.n_size)] # initializes the n-gram array

        final_text = self._format_text(text, self.n_size)
        for i in range(len(final_text)):
            sequence = tuple(final_text[i : (self.n_size + 1) + i])
            if len(sequence) == self.n_size + 1:
                if sequence in self.gram:
                    self.gram[sequence] += 1
                else:
                    self.gram[sequence] = 1
            else:
                break

    def incremental_training(self, other, sentences_arr):

        def _select_random_sentence():
            sentence_idx = random.choice(self.sentences_idx)
            return self.sentences_idx.index(sentence_idx), sentence_idx

        idx_in_arr, sentence_idx = _select_random_sentence()
        sentence = sentences_arr[sentence_idx]
        # print("REMOVING SENTENCE: {} IN INDEX {}".format(sentence, sentence_idx))
        self.remove_sentence(sentence, idx_in_arr)
        other.train(sentence, sentence_idx)

        return sentence, sentence_idx

    def remove_sentence(self, text, text_idx):

        self.sentences_idx.pop(text_idx)

        # if not self.grams_arr:
        #  self.grams_arr = [{} for _ in range(self.n_size)] # initializes the n-gram array

        final_text = self._format_text(text, self.n_size)
        for i in range(len(final_text)):
            sequence = tuple(final_text[i : (self.n_size + 1) + i])
            if len(sequence) == self.n_size + 1:
                self.gram[sequence] -= 1
                if self.gram[sequence] == 0:
                    del self.gram[sequence]
            else:
                break

    def entropy(self):
        normalizer = sum(self.gram.values())
        prob_dict = dict(
            (key, float(value) / normalizer) for key, value in self.gram.items()
        )

        return -sum(p * np.log2(p) for p in prob_dict.values())

    def rollback_incremental_training(self, other, sentence, sentence_idx):

        idx_in_arr = other.sentences_idx.index(sentence_idx)
        self.train(sentence, sentence_idx)
        other.remove_sentence(sentence, idx_in_arr)
