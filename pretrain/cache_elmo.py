from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py
import json
import sys
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = "0,2"

def build_elmo():
  with tf.device("/cpu:0"):
    token_ph = tf.placeholder(tf.string, [None, None])
    len_ph = tf.placeholder(tf.int32, [None])
    elmo_module = hub.Module("/home/makexin/makexin/elmo/modle/2")
    lm_embeddings = elmo_module(
      inputs={"tokens": token_ph, "sequence_len": len_ph},
      signature="tokens", as_dict=True)
    word_emb = lm_embeddings["word_emb"]
    lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                       lm_embeddings["lstm_outputs1"],
                       lm_embeddings["lstm_outputs2"]], -1)

  return token_ph, len_ph, lm_emb

def cache_dataset(data_path, session, token_ph, len_ph, lm_emb, out_file):
  in_file = json.load(open(data_path, 'r'))
  print("topic num :" ,len(in_file))
  topic_set = set()
  for topic_num in range(len(in_file)):
    sentences = in_file[topic_num]["sentences"]
    print("sentences num : ",len(sentences))
    topic = in_file[topic_num]["topic"]
    if topic in topic_set:
      continue
    else:
      topic_set.add(topic)
    group = out_file.create_group(topic)

    max_sentence_length = max(len(s) for s in sentences)
    print("max sentence length ：", max_sentence_length)
    tokens = [[""] * max_sentence_length for _ in sentences]
    text_len = np.array([len(s) for s in sentences])

    for i, sentence in enumerate(sentences):
      for j, word in enumerate(sentence):
        tokens[i][j] = word
    tokens = np.array(tokens)
    tf_lm_emb = session.run(lm_emb, feed_dict={
      token_ph: tokens,
      len_ph: text_len
    })
    for i, (e, l) in enumerate(zip(tf_lm_emb, text_len)):
      e = e[:l, :, :]
      group[str(i)] = e  # senterence id ，此处id为顺序编号
    print("Cached {} topic , topic is {}".format(topic_num, topic))



if __name__ == "__main__":
  token_ph, len_ph, lm_emb = build_elmo()
  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    hdf5_file_name = "/home/makexin/makexin/Experiment/data/v1/elmo_cache.hdf5"
    with h5py.File(hdf5_file_name, "w") as out_file:
      json_filename = "/home/makexin/makexin/Experiment/data/v1/topic_sentences.json"
      cache_dataset(json_filename,session,token_ph,len_ph,lm_emb,out_file)