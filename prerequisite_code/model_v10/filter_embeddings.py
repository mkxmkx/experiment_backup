from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import os

'''
保存训练数据和测试数据中所有句子的embedding
'''

if __name__ == "__main__":
  if len(sys.argv) < 3:   #输入参数个数大于等于3
    sys.exit("Usage: {} <embeddings> <json1> <json2> ...".format(sys.argv[0]))

  # words_to_keep = set()
  # for json_filename in sys.argv[2:]:   #读入第三个参数文件，将文件中sentence取出放到集合中
  #   if(os.path.exists("/home/makexin/makexin/Experiment/data/v4/Global_warming")):
  #     print("dir exist")
  #   if(os.path.exists(json_filename)):
  #     print(json_filename)
  #   file_list = os.listdir("/home/makexin/makexin/Experiment/data/v4/Global_warming")
  #   print(file_list)
  #   filename = "/home/makexin/makexin/Experiment/data/v4/Global_warming/" + file_list[1]
  #   print(filename)
  #
  #   with open(filename) as json_file:
  #     for line in json_file.readlines():
  #       for sentence in json.loads(line)["pre_sentences"]:
  #         words_to_keep.update(sentence)  # 将所有训练和测试句子加入到集合中，  "words"是一个句子
  #       for sentence in json.loads(line)["post_sentences"]:
  #         words_to_keep.update(sentence)

  words_to_keep = set()
  for json_filename in sys.argv[2:]:  # 读入第三个参数文件，将文件中sentence取出放到集合中
    with open(json_filename) as json_file:
      for line in json_file.readlines():
        for sentence in json.loads(line)["pre_sentences"]:
          words_to_keep.update(sentence)  # 将所有训练和测试句子加入到集合中，  "words"是一个句子
        for sentence in json.loads(line)["post_sentences"]:
          words_to_keep.update(sentence)

  print("Found {} words in {} dataset(s).".format(len(words_to_keep), len(sys.argv) - 2))

  total_lines = 0
  kept_lines = 0
  out_filename = "{}.filtered".format(sys.argv[1])   #写入的文件名
  with open(sys.argv[1]) as in_file:   #读取预训练好的embedding文件
    with open(out_filename, "w") as out_file:
      for line in in_file.readlines():
        total_lines += 1
        word = line.split()[0]
        if word in words_to_keep:   #若该句子在测试句子集合中出现过，则保存。保存内容为：单词 embedding
          kept_lines += 1
          out_file.write(line)

  print("Kept {} out of {} lines.".format(kept_lines, total_lines))
  print("Wrote result to {}.".format(out_filename))
