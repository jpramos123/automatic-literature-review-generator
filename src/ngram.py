import random
import numpy as np

BEGGIN_TOKEN = '<s> '
END_TOKEN = ' </s>'
MISSING_TOKEN = '<UNK>'

class Ngram:
  """
    ** Trying to apply some knowledge from Fluent Python book **
    Lazy way of programming (using generators)

  """

  def __init__(self, n_size):
    self.n_size = n_size # size of the n-gram
    self.grams_arr = [] # array with dicts, each index is a dict which is a n-gram
    self.sentences_idx = [] # array with sentences indexes

  def _format_text(self, text, n_idx):
    text = (BEGGIN_TOKEN*n_idx) + text + (END_TOKEN*n_idx)
    return text.split()

  def train(self, text, text_idx):
    
    """
      Create an array of n-grams [{1-gram}, {2-gram}, ... , {n-gram}]
    """

    self.sentences_idx.append(text_idx)
    if not self.grams_arr:
      self.grams_arr = [{} for _ in range(self.n_size)] # initializes the n-gram array
    
    for n_idx in range(self.n_size):      
      final_text = self._format_text(text, n_idx)
      for i in range(len(final_text)):
        sequence = tuple(final_text[i:(n_idx+1)+i])
        if len(sequence) == n_idx+1:
          if sequence in self.grams_arr[n_idx]:
            self.grams_arr[n_idx][sequence] += 1            
          else:
            self.grams_arr[n_idx][sequence] = 1
        else:
          break

  def incremental_training(self, other, sentences_arr):

    def _select_random_sentence():
      sentence_idx = random.choice(self.sentences_idx)
      return self.sentences_idx.index(sentence_idx), sentence_idx
    
    idx_in_arr, sentence_idx = _select_random_sentence()
    sentence = sentences_arr[sentence_idx]
    #print("REMOVING SENTENCE: {} IN INDEX {}".format(sentence, sentence_idx))
    self.remove_sentence(sentence, idx_in_arr)
    other.train(sentence, sentence_idx)

  def remove_sentence(self, text, text_idx):
    
    self.sentences_idx.pop(text_idx)

    if not self.grams_arr:
      self.grams_arr = [{} for _ in range(self.n_size)] # initializes the n-gram array
    
    for n_idx in range(self.n_size):      
      final_text = self._format_text(text, n_idx)
      for i in range(len(final_text)):
        sequence = tuple(final_text[i:(n_idx+1)+i])
        if len(sequence) == n_idx+1:
          self.grams_arr[n_idx][sequence] -= 1
          if self.grams_arr[n_idx][sequence] == 0:
            del self.grams_arr[n_idx][sequence]
        else:
          break

  def entropy(self):
    normalizers = [sum(v.values()) for v in self.grams_arr]
    probabilities = []
    entropies = []
    for idx, gram in enumerate(self.grams_arr):
      prob_dict = dict((key, float(value)/normalizers[idx]) for key,value in gram.items())
      probabilities.append(prob_dict)
    
    for prob in probabilities:
      entropy = -sum(p*np.log2(p) for p in prob.values())
      entropies.append(entropy)

    return entropies
  
  def rollback_incremental_training(self, other, sentence_arr):
    """
    TODO: In case the entropy of the current change does not get better
          rollback the incrementral training
    """
    pass
