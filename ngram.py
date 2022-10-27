from lib2to3.pgen2 import grammar
from this import d

from collections import defaultdict

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

  def _format_text(self, text, n_idx):
    text = (BEGGIN_TOKEN*n_idx) + text + (END_TOKEN*n_idx)
    return text.split()

  def train(self, text):
    if not self.grams_arr:
      self.grams_arr = [{} for _ in range(self.n_size)] # initializes the n-gram array
    
    for n_idx in range(self.n_size):      
      final_text = self._format_text(text, n_idx)
      for i in range(len(final_text)):
        sequence = tuple(final_text[i:(n_idx+1)+i])
        if len(sequence) == n_idx+1:
          #print(f.split()[i:n+i], i)
          if sequence in self.grams_arr[n_idx]:
            self.grams_arr[n_idx][sequence] += 1            
          else:
            self.grams_arr[n_idx][sequence] = 1
        else:
          break
