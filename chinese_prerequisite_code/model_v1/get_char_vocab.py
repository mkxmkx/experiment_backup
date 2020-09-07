from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

'''
将json文件读入，将其中sentences字段sentence取出，将所有单词取出，不重复的放进一个集合中。
生成txt的单词词表
所生成文件名： char_vocab.english.txt
'''

def get_char_vocab(input_filenames, output_filename):
  vocab = set()
  f = json.load(open(input_filenames, "r"))
  for topic_num in range(len(f)):
      sentences = f[topic_num]["sentences"]
      for sentence in sentences:
          for word in sentence:
              vocab.update(word)  # set update方法可以添加新的元素或集合到当前集合中，如果添加的元素在集合中已存在，则该元素只会出现一次，重复的会忽略。
  vocab = sorted(list(vocab))
  with open(output_filename, "wb") as f:
    for char in vocab:
      f.write(u"{}\n".format(char).encode("utf8"))
  print("Wrote {} characters to {}".format(len(vocab), output_filename))

def get_char_vocab_language():
  get_char_vocab("/home/makexin/makexin/Experiment/data/Chinese/dataStructure/topic_sentences.json",
                 "/home/makexin/makexin/Experiment/data/Chinese/dataStructure/char_vocab.txt")

get_char_vocab_language()
