from src.ngram import Ngram


class TestNgram:

  sentences = ["hello wid how are you", "hello world"]
  grams_arr = [{("hello",): 2, 
                ("wid",): 1, 
                ("how",): 1,
                ("are",): 1,
                ("you",): 1,
                ("world",): 1
                }, 
                {("<s>","hello"): 2,
                  ("hello", "wid"): 1,
                  ("wid", "how"): 1,
                  ("how", "are"): 1,
                  ("are", "you"): 1,
                  ("you", "</s>"): 1,
                  ("hello", "world"): 1,
                  ("world", "</s>"): 1
                }, 
                {("<s>", "<s>", "hello"): 2,
                 ("<s>", "hello", "wid"): 1,
                 ("hello", "wid", "how"): 1,
                 ("wid", "how", "are"): 1,
                 ("how", "are", "you"): 1,
                 ("are", "you", "</s>"): 1,
                 ("you", "</s>", "</s>"): 1,
                 ("<s>", "hello", "world"): 1,
                 ("hello", "world", "</s>"): 1,
                 ("world", "</s>", "</s>"): 1,
                }]
  
  def test_train_unigram(self):
    n1 = Ngram(1)
    
    n1.train(self.sentences[0], 0)
    n1.train(self.sentences[1], 1)

    assert n1.grams_arr == self.grams_arr[0:1]

  def test_train_bigram(self):
    n2 = Ngram(2)
    n2.train(self.sentences[0], 0)
    n2.train(self.sentences[1], 1)

    assert n2.grams_arr == self.grams_arr[0:2]

  def test_train_trigram(self):
    n3 = Ngram(3)
    n3.train(self.sentences[0], 0)
    n3.train(self.sentences[1], 1)

    assert n3.grams_arr == self.grams_arr
