from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py
import tensorflow.contrib.eager as tfe

from model_v2 import util
from model_v2 import coref_ops
import metrics
from model_v2 import evaluate


class CorefModel(object):
  def __init__(self, config):
    self.config = config
    self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
    #self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
    #self.char_embedding_size = config["char_embedding_size"]
    #self.char_dict = util.load_char_dict(config["char_vocab_path"])
    self.max_span_width = config["max_span_width"]
    #self.genres = { g:i for i,g in enumerate(config["genres"]) }
    if config["lm_path"]:
      self.lm_file = h5py.File(self.config["lm_path"], "r")
    else:
      self.lm_file = None
    self.lm_layers = self.config["lm_layers"]
    self.lm_size = self.config["lm_size"]
    self.eval_data = None # Load eval data lazily.

    input_props = []
    input_props.append((tf.int32, [None]))  #pre_topic start index
    input_props.append((tf.int32, [None]))  #pre topic end index
    input_props.append((tf.int32, [None]))  # post topic start index
    input_props.append((tf.int32, [None]))  # post topic end index
    input_props.append((tf.string, [None, None])) # pre_Tokens.
    input_props.append((tf.string, [None, None]))  # post_Tokens.
    input_props.append((tf.float32, [None, None, self.context_embeddings.size])) # pre Context embeddings.
    input_props.append((tf.float32, [None, None, self.context_embeddings.size]))  # post Context embeddings.
    #input_props.append((tf.float32, [None, None, self.head_embeddings.size])) # Head embeddings.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers])) # pre LM embeddings. ELMo   lm_file
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers]))  # post LM embeddings. ELMo   lm_file
    input_props.append((tf.int32, [None])) # pre Text lengths.
    input_props.append((tf.int32, [None]))  # post Text lengths.
    input_props.append((tf.bool, [])) # Is training.
    input_props.append((tf.float32, []))  # label


    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]   #输入数据初始化

    dtypes, shapes = zip(*input_props)
    try:
        queue = tf.PaddingFIFOQueue(capacity=self.config["batch_size"] * 2, dtypes=dtypes, shapes=shapes)  # 创建先进先出队列
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)  # 入队操作
        self.input_tensors = queue.dequeue_many(self.config["batch_size"])  # 出队操作
    except tf.errors.OutOfRangeError :
        pass


    print("input tensor: ", self.input_tensors)
    # min_after_dequeue = 1000
    # capacity = min_after_dequeue + 3 * self.config["batch_size"]
    # input_queue = tf.train.slice_input_producer(self.queue_input_tensors, shuffle=True, capacity= capacity)
    # data_batch = tf.train.batch(input_queue, batch_size= self.config["batch_size"], capacity = capacity, dynamic_pad=True, allow_smaller_final_batch=False)

    self.loss, self.post_topic_relation_score, self.pre_topic_relation_score, self.predictions= self.get_predictions_and_loss(*self.input_tensors)    #获得了top span antecedent以及embedding表示、score, loss
    #self.loss, self.post_topic_relation_score, self.pre_topic_relation_score, self.predictions = self.get_predictions_and_loss(data_batch)
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)
    learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               self.config["decay_frequency"], self.config["decay_rate"], staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam" : tf.train.AdamOptimizer,
      "sgd" : tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

  def start_enqueue_thread(self, session):
    with open(self.config["train_path"]) as f:
      train_examples = [json.loads(jsonline) for jsonline in f.readlines()]   #载入训练数据
    def _enqueue_loop():
      while True:

        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord, sess=session)

        random.shuffle(train_examples)
        for example in train_examples:
          tensorized_example = self.tensorize_example(example, is_training=True)
          feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
          #print("feed_dict: ", feed_dict)
          session.run(self.enqueue_op, feed_dict=feed_dict)    # feed_dict={ self.queue_input_tensors : tensorized_example}，初始化self.queue_input_tensors
          # coord.request_stop()
          # coord.join(threads)


    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def restore(self, session):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
    saver = tf.train.Saver(vars_to_restore)
    checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)

  def load_lm_embeddings(self, topic):
    if self.lm_file is None:
      return np.zeros([0, 0, self.lm_size, self.lm_layers])
    file_key = topic
    group = self.lm_file[file_key]
    num_sentences = len(list(group.keys()))
    sentences = [group[str(i)][...] for i in range(num_sentences)]
    lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
    for i, s in enumerate(sentences):
      lm_emb[i, :s.shape[0], :, :] = s
    return lm_emb

  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = zip(*mentions)
    else:
      starts, ends = [], []
    return np.array(starts), np.array(ends)

  def tensorize_span_labels(self, tuples, label_dict):
    if len(tuples) > 0:
      starts, ends, labels = zip(*tuples)
    else:
      starts, ends, labels = [], [], []
    return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

  def tensorize_example(self, example, is_training):   #将所读取的训练数据转换为tensor
    #clusters = example["clusters"]

    #gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    #gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
    #cluster_ids = np.zeros(len(gold_mentions))
    '''

    for cluster_id, cluster in enumerate(clusters):
      for mention in cluster:
        cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1
    '''
    pre_topic = example["pre_topic"]
    post_topic = example["post_topic"]

    pre_topic_start_index = [example["pre_topic_start_index"]]
    pre_topic_end_index = [example["pre_topic_end_index"]]
    post_topic_start_index = [example["post_topic_start_index"]]
    post_topic_end_index = [example["post_topic_end_index"]]

    pre_sentences = example["pre_sentences"]
    post_sentences = example["post_sentences"]
    pre_num_words = sum(len(s) for s in pre_sentences)
    post_num_words = sum(len(s) for s in post_sentences)
    #speakers = util.flatten(example["speakers"])

    #assert num_words == len(speakers)

    pre_max_sentence_length = max(len(s) for s in pre_sentences)
    post_max_sentence_length = max(len(s) for s in post_sentences)

    pre_max_word_length = max(max(max(len(w) for w in s) for s in pre_sentences), max(self.config["filter_widths"]))
    post_max_word_length = max(max(max(len(w) for w in s) for s in post_sentences), max(self.config["filter_widths"]))

    pre_text_len = np.array([len(s) for s in pre_sentences])
    post_text_len = np.array([len(s) for s in post_sentences])

    pre_tokens = [[""] * pre_max_sentence_length for _ in pre_sentences]
    post_tokens = [[""] * post_max_sentence_length for _ in post_sentences]

    pre_context_word_emb = np.zeros([len(pre_sentences), pre_max_sentence_length, self.context_embeddings.size])    #self.context_embeddings.size针对不同的训练数据应该是不一样的，pre和post
    post_context_word_emb = np.zeros([len(post_sentences), post_max_sentence_length, self.context_embeddings.size])
    #head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
    #char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])

    for i, sentence in enumerate(pre_sentences):
      for j, word in enumerate(sentence):
        pre_tokens[i][j] = word
        pre_context_word_emb[i, j] = self.context_embeddings[word]
        #head_word_emb[i, j] = self.head_embeddings[word]
        #char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
    pre_tokens = np.array(pre_tokens)

    for i, sentence in enumerate(post_sentences):
      for j, word in enumerate(sentence):
        post_tokens[i][j] = word
        post_context_word_emb[i, j] = self.context_embeddings[word]
        #head_word_emb[i, j] = self.head_embeddings[word]
        #char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
    post_tokens = np.array(post_tokens)

    #speaker_dict = { s:i for i,s in enumerate(set(speakers)) }
    #speaker_ids = np.array([speaker_dict[s] for s in speakers])

    #doc_key = example["doc_key"]
    #genre = self.genres[doc_key[:2]]

    #gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

    pre_lm_emb = self.load_lm_embeddings(pre_topic)
    post_lm_emb = self.load_lm_embeddings(post_topic)

    #print("example label: ", example["label"])

    if example["label"] != "1":
      label = float(0)
      #print("false label")
    else:
      label = float(example["label"])
      #print("true label")


    #example_tensors = (tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids)
    example_tensors = (pre_topic_start_index, pre_topic_end_index, post_topic_start_index, post_topic_end_index, pre_tokens, post_tokens, pre_context_word_emb, post_context_word_emb, pre_lm_emb, post_lm_emb, pre_text_len, post_text_len, is_training, label)

    if is_training and len(pre_sentences) > self.config["max_training_sentences"] and len(post_sentences) > self.config["max_training_sentences"]:
      return self.truncate_example(*example_tensors)
    else:
      return example_tensors

  def truncate_example(self, pre_topic_start_index, pre_topic_end_index, post_topic_start_index, post_topic_end_index, pre_tokens, post_tokens, pre_context_word_emb, post_context_word_emb,  pre_lm_emb, post_lm_emb, pre_text_len, post_text_len, is_training, label):
    max_training_sentences = self.config["max_training_sentences"]
    pre_num_sentences = pre_context_word_emb.shape[0]
    post_num_sentences = post_context_word_emb.shape[0]

    assert pre_num_sentences > max_training_sentences
    assert post_num_sentences > max_training_sentences

    pre_sentence_offset = random.randint(0, pre_num_sentences - max_training_sentences)
    pre_word_offset = pre_text_len[:pre_sentence_offset].sum()
    pre_num_words = pre_text_len[pre_sentence_offset:pre_sentence_offset + max_training_sentences].sum()
    pre_tokens = pre_tokens[pre_sentence_offset:pre_sentence_offset + max_training_sentences, :]
    pre_context_word_emb = pre_context_word_emb[pre_sentence_offset:pre_sentence_offset + max_training_sentences, :, :]
    pre_lm_emb = pre_lm_emb[pre_sentence_offset:pre_sentence_offset + max_training_sentences, :, :, :]
    pre_text_len = pre_text_len[pre_sentence_offset:pre_sentence_offset + max_training_sentences]

    post_sentence_offset = random.randint(0, post_num_sentences - max_training_sentences)
    post_word_offset = post_text_len[:post_sentence_offset].sum()
    post_num_words = post_text_len[post_sentence_offset:post_sentence_offset + max_training_sentences].sum()
    post_tokens = post_tokens[post_sentence_offset:post_sentence_offset + max_training_sentences, :]
    post_context_word_emb = post_context_word_emb[post_sentence_offset:post_sentence_offset + max_training_sentences, :, :]
    post_lm_emb = post_lm_emb[post_sentence_offset:post_sentence_offset + max_training_sentences, :, :, :]
    post_text_len = post_text_len[post_sentence_offset:post_sentence_offset + max_training_sentences]

    return pre_topic_start_index, pre_topic_end_index, post_topic_start_index, post_topic_end_index, pre_tokens, post_tokens, pre_context_word_emb, post_context_word_emb, pre_lm_emb, post_lm_emb, pre_text_len, post_text_len, is_training, label

  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
    same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0)) # [num_labeled, num_candidates]
    same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) # [num_labeled, num_candidates]
    same_span = tf.logical_and(same_start, same_end) # [num_labeled, num_candidates]
    candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span)) # [1, num_candidates]
    candidate_labels = tf.squeeze(candidate_labels, 0) # [num_candidates]
    return candidate_labels

  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)

  def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
    top_span_emb = tf.debugging.check_numerics(top_span_emb, "top span emb")   # [batch_size, k, emb]
    top_span_mention_scores = tf.debugging.check_numerics(top_span_mention_scores, "top_span_mention_scores")   # [batch_size, k]

    k = util.shape(top_span_emb, 1)   #获得第0维的大小
    #k_max = top_span_emb.shape[0].value
    top_span_range = tf.range(k) # [k]
    antecedent_offsets = tf.tile(tf.expand_dims(tf.expand_dims(top_span_range, 1), 0), [self.config["batch_size"], 1, 1]) \
                         - tf.tile(tf.expand_dims(tf.expand_dims(top_span_range, 0), 0), [self.config["batch_size"], 1, 1]) # [batch_size, k, k]
    #antecedents_mask = antecedent_offsets >= 1 # [k, k]
    antecedents_mask = antecedent_offsets >= 0  # [batch_size, k, k]   此处修改为大于等于0，不知道有作用没
    fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 2) + tf.expand_dims(top_span_mention_scores, 1) # [batch_size, k, k]
    fast_antecedent_scores = tf.debugging.check_numerics(fast_antecedent_scores, "fast antecedent scores 1")
    fast_antecedent_scores += tf.log(tf.clip_by_value(tf.to_float(antecedents_mask), 1e-10, 1.0)) # [batch_size, k, k]    该步的作用？？？
    fast_antecedent_scores = tf.debugging.check_numerics(fast_antecedent_scores, "fast antecedent scores 2")
    fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb) # [batch_size, k, k]
    print("fast antecedent scores: ", fast_antecedent_scores.get_shape())

    _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False) # [batch_size, k, c]
    print("top antecedents: ", top_antecedents.get_shape())
    top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents) # [batch_size, k, c]
    print("top antecedents mask: ", top_antecedents_mask.get_shape())
    top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents) # [batch_size, k, c]
    print("top fast antecedent scores: ", top_fast_antecedent_scores.get_shape())
    top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
    print("top antecedent offsets: ", top_antecedent_offsets.get_shape())
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def distance_pruning(self, top_span_emb, top_span_mention_scores, c):
    top_span_emb = tf.debugging.check_numerics(top_span_emb, "top span emb")  # [batch_size, k, emb]
    top_span_mention_scores = tf.debugging.check_numerics(top_span_mention_scores,
                                                            "top_span_mention_scores")  # [batch_size, k]
    k = util.shape(top_span_emb, 1)
    top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1]) # [k, c]
    raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets # [k, c]
    top_antecedents_mask = raw_top_antecedents >= 0 # [k, c]
    top_antecedents = tf.maximum(raw_top_antecedents, 0) # [k, c]

    top_fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores, top_antecedents) # [k, c]
    top_fast_antecedent_scores += tf.log(tf.to_float(top_antecedents_mask)) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def get_predictions_and_loss(self, pre_topic_start_index, pre_topic_end_index, post_topic_start_index, post_topic_end_index, pre_tokens, post_tokens, pre_context_word_emb, post_context_word_emb, pre_lm_emb, post_lm_emb, pre_text_len, post_text_len, is_training, label):
    #原始参数：pre_topic_start_index, pre_topic_end_index, post_topic_start_index, post_topic_end_index, pre_tokens, post_tokens, pre_context_word_emb, post_context_word_emb, pre_lm_emb, post_lm_emb, pre_text_len, post_text_len, is_training, label
    #data_batch = tf.convert_to_tensor(data_batch)

    # pre_topic_start_index = tf.slice(data_batch, [0, 0], [-1, 1])
    # pre_topic_end_index = tf.slice(data_batch, [0, 1], [-1, 1])
    # post_topic_start_index = tf.slice(data_batch, [0, 2], [-1, 1])
    # post_topic_end_index = tf.slice(data_batch, [0, 3], [-1, 1])
    # pre_tokens = tf.slice(data_batch, [0, 4], [-1, 1])
    # post_tokens = tf.slice(data_batch, [0, 5], [-1, 1])
    # pre_context_word_emb = tf.slice(data_batch, [0, 6], [-1, 1])
    # post_context_word_emb = tf.slice(data_batch, [0, 7], [-1, 1])
    # pre_lm_emb = tf.slice(data_batch, [0, 8], [-1, 1])
    # post_lm_emb = tf.slice(data_batch, [0, 9], [-1, 1])
    # pre_text_len = tf.slice(data_batch, [0, 10], [-1, 1])
    # post_text_len = tf.slice(data_batch, [0, 11], [-1, 1])
    # is_training = tf.slice(data_batch, [0, 12], [-1, 1])
    # label = tf.slice(data_batch, [0, 13], [-1, 1])

    # pre_topic_start_index = data_batch[0]
    # pre_topic_end_index = data_batch[1]
    # post_topic_start_index = data_batch[2]
    # post_topic_end_index = data_batch[3]
    # pre_tokens = data_batch[4]
    # post_tokens = data_batch[5]
    # pre_context_word_emb = data_batch[6]
    # post_context_word_emb = data_batch[7]
    # pre_lm_emb = data_batch[8]
    # post_lm_emb = data_batch[9]
    # pre_text_len = data_batch[10]
    # post_text_len = data_batch[11]
    # is_training = data_batch[12]
    # label = data_batch[13]

    print("lm emb: ", pre_lm_emb.get_shape())



    with tf.variable_scope("pre_topic"):
      pre_topic_emb, pre_top_span_emb, pre_top_span_mention_scores, pre_top_antecedent_emb, pre_top_antecedent_scores = self.get_predictions(pre_topic_start_index, pre_topic_end_index, pre_tokens, pre_context_word_emb, pre_lm_emb, pre_text_len, is_training)
    with tf.variable_scope("post_topic"):
      post_topic_emb, post_top_span_emb, post_top_span_mention_scores, post_top_antecedent_emb, post_top_antecedent_scores = self.get_predictions(post_topic_start_index, post_topic_end_index, post_tokens, post_context_word_emb, post_lm_emb, post_text_len, is_training)


    '''
    topic_emb: [batch_size, 1, emb]
    top_span_emb: [batch_size, k, emb]
    top_span_mention_scores: [batch_size, k]
    top_antecedent_emb: [batch_size, k, c + 1, emb]    获得的可能作为后序词的span
    top_antecedent_scores: [batch_size, k, c + 1]
    '''

    print("topic_emb: ", pre_topic_emb.get_shape())
    print("top_span_emb: ", pre_top_span_emb.get_shape())
    print("top_span_mention_scores: ", pre_top_span_mention_scores.get_shape())
    print("top_antecedent_emb: ", pre_top_antecedent_emb.get_shape())
    print("top_antecedent_scores: ", pre_top_antecedent_scores.get_shape())

    # pre_topic_emb = tf.squeeze(pre_topic_emb)   #[emb]
    # post_topic_emb = tf.squeeze(post_topic_emb)

    #span_emb = tf.concat(0,[pre_top_span_emb,post_top_span_emb])  #[2k, emb]
    span_emb_set = set()

    pre_k = util.shape(pre_top_span_emb, 1)
    # pre_k = pre_top_span_emb.get_shape()[0]
    # print("pre_k 维度值: ", pre_k)
    pre_c = util.shape(pre_top_antecedent_emb, 2)
    #pre_c = pre_top_antecedent_emb.get_shape().as_list()[1]
    # pre_top_span_emb = tf.expand_dims(tf.tile(pre_top_span_emb, [1, pre_c]), 1)   #将emb复制c次后，增加一个维度，转变为[k,c,emb]的维度
    #pre_span_pair = [ [] * pre_c for _ in range(pre_k)]   #获得span对的embedding， pre_span_pair[i][j]的元素是span对的embedding表示，元组对的形式
    pre_temp_top_span_emb = tf.tile(tf.expand_dims(pre_top_span_emb, 2), [1, 1, pre_c, 1])  # [batch_size, k, c, emb]
    print("pre_top_span_emb: {}".format(pre_top_span_emb))
    pre_span_pair = tf.stack([pre_temp_top_span_emb, pre_top_antecedent_emb], 3)   # [batch_size, k, c, 2, emb], span 和对应先行词span组成的span对

    print("pre_span_pair: {}".format(pre_span_pair))

    #pre_span_pair = tf.tile(tf.expand_dims(tf.range(pre_k),1), [1, pre_c])
    #print("tf.range(pre_k): ", tf.range(pre_k))

    # for i in tf.range(pre_k):
    #   for j in tf.range(pre_c):
    #     pre_span_pair[i][j] = [pre_top_span_emb[i][j], pre_top_antecedent_emb[i][j]]
    #     span_emb_set.add(pre_top_span_emb[i][j])
    #     span_emb_set.add(pre_top_antecedent_emb[i][j])



    post_k = util.shape(post_top_span_emb, 1)
    post_c = util.shape(post_top_antecedent_emb, 2)
    post_temp_top_span_emb = tf.tile(tf.expand_dims(post_top_span_emb, 2), [1, 1, post_c, 1])  # [batch_size, k, c, emb]
    #post_temp_top_span_emb = tf.expand_dims(tf.tile(post_top_span_emb, [1, post_c]), 1)
    post_span_pair = tf.stack([post_temp_top_span_emb, post_top_antecedent_emb], 3)  # [batch_size, k, c, 2, emb], span 和对应先行词span组成的span对
    # post_span_pair = [ [] * post_c for _ in tf.range(post_k)]
    # for i in tf.range(post_k):
    #   for j in tf.range(post_c):
    #     post_span_pair[i][j] = [post_top_span_emb[i][j], post_top_antecedent_emb[i][j]]
    #     span_emb_set.add(post_top_span_emb[i][j])
    #     span_emb_set.add(post_top_antecedent_emb[i][j])

    # 给定topic，获得和span的余弦相似度值    pre_topic
    temp_pre_topic_emb = tf.tile(pre_topic_emb, [1, pre_k, 1])   #[batch_size, pre_k, emb]
    pre_topic_pre_span_cosin = self.topic_span_cosin(temp_pre_topic_emb, pre_top_span_emb, 2)   #[batch_size, pre_k]
    temp_pre_topic_emb = tf.tile(pre_topic_emb, [1, post_k, 1])   # [batch_size, post_k, emb]
    pre_topic_post_span_cosin = self.topic_span_cosin(temp_pre_topic_emb, post_top_span_emb, 2)   #[batch_size, post_k]

    #给定topic，获得和antecedent的余弦相似度值    pre_topic
    temp_pre_topic_emb = tf.tile(tf.expand_dims(pre_topic_emb, 1), [1, pre_k, pre_c, 1])    # [batch_size, pre_k, pre_c, emb]
    pre_topic_pre_antecedent_cosin = self.topic_span_cosin(temp_pre_topic_emb, pre_top_antecedent_emb, 3)    # [batch_size, pre_k, pre_c]
    temp_pre_topic_emb = tf.tile(tf.expand_dims(pre_topic_emb, 1), [1, post_k, post_c, 1])    # [batch_size, post_k, post_c, emb]
    pre_topic_post_antecedent_cosin = self.topic_span_cosin(temp_pre_topic_emb, post_top_antecedent_emb, 3)   # [batch_size, post_k, post_c]

    # 给定topic，获得和span的余弦相似度值    post_topic
    temp_post_topic_emb = tf.tile(post_topic_emb, [1, pre_k, 1])  # [batch_size, pre_k, emb]
    post_topic_pre_span_cosin = self.topic_span_cosin(temp_post_topic_emb, pre_top_span_emb, 2)  # [batch_size, pre_k]
    temp_post_topic_emb = tf.tile(post_topic_emb, [1, post_k, 1])  # [batch_size, post_k, emb]
    post_topic_post_span_cosin = self.topic_span_cosin(temp_post_topic_emb, post_top_span_emb, 2)  # [batch_size, post_k]

    # 给定topic，获得和antecedent的余弦相似度值    post_topic
    temp_post_topic_emb = tf.tile(tf.expand_dims(post_topic_emb, 1), [1, pre_k, pre_c, 1])  # [batch_size, pre_k, pre_c, emb]
    post_topic_pre_antecedent_cosin = self.topic_span_cosin(temp_post_topic_emb, pre_top_antecedent_emb,
                                                           3)  # [batch_size, pre_k, pre_c]
    temp_post_topic_emb = tf.tile(tf.expand_dims(post_topic_emb, 1), [1, post_k, post_c, 1])  # [batch_size, post_k, post_c, emb]
    post_topic_post_antecedent_cosin = self.topic_span_cosin(temp_post_topic_emb,
                                                            post_top_antecedent_emb, 3)  # [batch_size, post_k, post_c]

    # 统计两个主题分别作为先序词的得分
    pre_topic_relation_score = [ 0.0 for _ in range(self.config["batch_size"])]
    post_topic_relation_score = [ 0.0 for _ in range(self.config["batch_size"])]

    # antecedent为后序词，即先后序关系指向antecedent ， 后学antecedent


    # pre_topic作为后序词    pre_antecedent
    pre_topic_relation_score += self.get_topic_relation_score(pre_topic_pre_antecedent_cosin, pre_top_antecedent_scores, post_topic_post_span_cosin, post_k, pre_k, pre_c)
    # pre topic作为后序词    post antecedent
    pre_topic_relation_score += self.get_topic_relation_score(pre_topic_post_antecedent_cosin, post_top_antecedent_scores, post_topic_pre_span_cosin, pre_k, post_k, post_c)
    # post topic作为后序词    pre antecedent
    post_topic_relation_score += self.get_topic_relation_score(post_topic_pre_antecedent_cosin, pre_top_antecedent_scores, pre_topic_post_span_cosin, post_k, pre_k, pre_c)
    # post topic作为后序词     post antecedent
    post_topic_relation_score += self.get_topic_relation_score(post_topic_post_antecedent_cosin, post_top_antecedent_scores, pre_topic_pre_span_cosin, pre_k, post_k, post_c)


    # pre_topic_pre_relation = tf.multiply(pre_topic_pre_antecedent_cosin,
    #                                       pre_top_antecedent_scores)  # [pre_k, pre_c]
    # pre_topic_pre_relation = tf.tile(tf.expand_dims(pre_topic_pre_relation, 0),
    #                                   [post_k, 1, 1])  # [post_k, pre_k, pre_c]
    # temp_post_topic_post_span_cosin = tf.tile(tf.expand_dims(tf.expand_dims(post_topic_post_span_cosin, 1), 1),
    #                                          [1, pre_k, pre_c])  # [post_k, pre_k, pre_c]
    # temp_pre_topic_pre_antecedent_cosin = tf.tile(tf.expand_dims(pre_topic_pre_antecedent_cosin, 0),
    #                                                [post_k, 1, 1])  # [post_k, pre_k, pre_c]
    # pre_topic_pre_mask = tf.logical_and(
    #   tf.greater_equal(temp_post_topic_post_span_cosin, tf.fill([post_k, pre_k, pre_c], metric)),
    #   tf.greater_equal(temp_pre_topic_pre_antecedent_cosin, tf.fill([post_k, pre_k, pre_c], metric)))
    # pre_topic_as_antecedent = tf.boolean_mask(pre_topic_pre_relation, pre_topic_pre_mask)
    # pre_topic_relation_score += tf.reduce_sum(pre_topic_as_antecedent)





    # print("topic emb shape {}".format(pre_topic_emb.get_shape()))
    # pre_related_span = self.get_related_span(pre_topic_emb, span_emb_set)
    # post_related_span = self.get_related_span(post_topic_emb, span_emb_set)



    # for i in tf.range(pre_k):
    #   for j in tf.range(pre_c):
    #     [emb_1, emb_2] = pre_span_pair[i][j]
    #     if emb_1 in pre_related_span and emb_2 in post_related_span:    #由emb_1指向emb_2，先学emb_1，后学emb_2
    #       span_score = self.cosin_score(post_topic_emb, emb_2)   #span的相似度值
    #       relation_score = pre_top_antecedent_scores[i][j]     #span对的分值
    #       post_topic_relation_score += ( span_score * relation_score )
    #     if emb_2 in pre_related_span and emb_1 in post_related_span:
    #       span_score = self.cosin_score(pre_topic_emb, emb_2)
    #       relation_score = pre_top_antecedent_scores[i][j]
    #       pre_topic_relation_score += ( span_score * relation_score)
    # for i in tf.range(post_k):
    #   for j in tf.range(post_c):
    #     [emb_1, emb_2] = post_span_pair[i][j]
    #     if emb_1 in pre_related_span and emb_2 in post_related_span:
    #       span_score = self.cosin_score(post_topic_emb, emb_2)
    #       relation_score = post_top_antecedent_scores[i][j]
    #       post_topic_relation_score += ( span_score * relation_score)
    #     if emb_2 in pre_related_span and emb_1 in post_related_span:
    #       span_score = self.cosin_score(pre_topic_emb, emb_2)
    #       relation_score = post_top_antecedent_scores[i][j]
    #       pre_topic_relation_score += ( span_score * relation_score)

    # post_topic_relation_score = tf.sigmoid(post_topic_relation_score)
    # pre_topic_relation_score = tf.sigmoid(pre_topic_relation_score)
    post_score = post_topic_relation_score - pre_topic_relation_score     #post topic作为后序词
    pre_score = pre_topic_relation_score - post_topic_relation_score

    final_score = post_topic_relation_score - pre_topic_relation_score

    final_score = tf.sigmoid(final_score)
    print("final score: ", final_score.get_shape())
    print("label: ", label.get_shape())

    # if label != 1:

    #   label = 0

    #loss = self.corss_entropy_loss(tf.cast(label, tf.float32), tf.cast(final_score, tf.float32))
    #loss = self.corss_entropy_loss(label, final_score)
    loss = self.weighted_cross_entorpy_with_logits(label, final_score)




    return loss,post_topic_relation_score, pre_topic_relation_score, [pre_topic_emb, post_topic_emb, pre_top_span_emb, post_top_span_emb, pre_top_span_mention_scores, post_top_span_mention_scores, pre_top_antecedent_emb, post_top_antecedent_emb, pre_top_antecedent_scores, post_top_antecedent_scores, final_score]

  def get_topic_relation_score(self, antecedent_cosin, antecedent_scores, span_cosin, dim1, dim2, dim3):

    '''

    :param antecedent_cosin:    [batch_size, dim2, dim3]
    :param antecedent_scores:   [batch_size, dim2, dim3]
    :param span_cosin:   [batch_size, dim1]
    :param dim1:
    :param dim2:
    :param dim3:
    :return:
    '''

    metric = 0.0

    # 例子，pre_topic作为后序词    和pre_antecedent相关
    pre_topic_pre_relation = tf.multiply(antecedent_cosin,
                                         antecedent_scores)  # [batch_size, pre_k, pre_c]   [dim2, dim3]
    pre_topic_pre_relation = tf.tile(tf.expand_dims(pre_topic_pre_relation, 1),
                                     [1, dim1, 1, 1])  # [batch_size, post_k, pre_k, pre_c]   [dim1, dim2, dim3]
    temp_post_topic_post_span_cosin = tf.tile(tf.expand_dims(tf.expand_dims(span_cosin, 2), 2),
                                              [1, 1, dim2, dim3])  # [batch_size, post_k, pre_k, pre_c]   [dim1, dim2, dim3]
    temp_pre_topic_pre_antecedent_cosin = tf.tile(tf.expand_dims(antecedent_cosin, 1),
                                                  [1, dim1, 1, 1])  # [batch_size, post_k, pre_k, pre_c]   [dim1, dim2, dim3]
    #mask: pre和post topic和span的相关度都要大于零
    pre_topic_pre_mask = tf.logical_and(
      tf.greater_equal(temp_post_topic_post_span_cosin, tf.fill([self.config["batch_size"], dim1, dim2, dim3], metric)),
      tf.greater_equal(temp_pre_topic_pre_antecedent_cosin, tf.fill([self.config["batch_size"], dim1, dim2, dim3], metric)))
    pre_topic_as_antecedent = tf.boolean_mask(pre_topic_pre_relation, pre_topic_pre_mask)
    return tf.reduce_sum(pre_topic_as_antecedent)

  def topic_span_cosin(self, topic_emb, span_emb, dim):
    #求模
    topic_norm = tf.sqrt(tf.reduce_sum(tf.square(topic_emb), axis=dim))
    span_norm = tf.sqrt(tf.reduce_sum(tf.square(span_emb), axis=dim))
    #内积
    topic_span = tf.reduce_sum(tf.multiply(topic_emb, span_emb), axis=dim)
    cosin = tf.divide(topic_span, tf.multiply(topic_norm, span_norm))
    return cosin

  def corss_entropy_loss(self, label, score):
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=score)
      return tf.reduce_mean(loss)

  def weighted_cross_entorpy_with_logits(self, label, score):
      weight = 2.0
      loss = tf.nn.weighted_cross_entropy_with_logits(logits=score, targets=label, pos_weight=weight)
      loss += tf.add_n(tf.get_collection('losses'))
      return tf.reduce_mean(loss)

  def softmax_cross_entropy_with_logits(self, logits, label):
      if label == 1.0:
          label = tf.convert_to_tensor([0, 1], dtype=tf.int32)
      else:
          label = tf.convert_to_tensor([1, 0], tf.int32)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
      return tf.reduce_mean(loss)

  def cosin_score(self, emb_1, emb_2):
    #求模
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(emb_1), 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(emb_2), 1))
    #内积
    mul_emb = tf.reduce_sum(tf.multiply(emb_1, emb_2), 1)
    cosin = tf.divide(mul_emb, tf.multiply(pooled_len_1, pooled_len_2))
    return cosin
  def get_related_span(self, topic, span_emb_set):

    related_span = []
    for span in iter(span_emb_set):
      score = self.cosin_score(topic, span)
      if score > 0.5:
          related_span.append(span)
    return related_span


  def get_predictions(self, topic_start_index, topic_end_index, tokens, context_word_emb, lm_emb, text_len, is_training):    #contex embedding和lm embedding不一样
      '''
      :param topic_start_index:  [batch_size, 1]
      :param topic_end_index:   [batch_size, 1]
      :param tokens:   [batch_size, tokens]
      :param context_word_emb:   [batch_size, num_sentences, max_sentence_length, emb]
      :param lm_emb:   [batch_size, num_sentences, max_sentence_length, 1024, 3]
      :param text_len:   [batch_size, num_sentence] , [batch_size, num_sentence, max_sentence_length]
      :param is_training:   [batch_size, 1]
      :return:
      topic_start_index: [batch_size, 1]
      topic_end_index: [batch_size, 1]
      tokens: [batch_size, tokens]
      context_word_emb: [batch_size, num_sentences, max_sentence_length, emb]
      lm_emb: [batch_size, num_sentences, max_sentence_length, 1024, 3]
      text_len: [batch_size, num_sentence, max_sentence_length]
      is training: [batch_size, 1]
      '''


      # topic_start_index = tf.Print(topic_start_index, [topic_start_index], "topic start index value")
      # tokens = tf.Print(tokens, [tokens], "tokens value")
      # context_word_emb = tf.Print(context_word_emb, [context_word_emb], "context word embedding value")
      # lm_emb = tf.Print(lm_emb, [lm_emb], "lm embedding value")
      # text_len = tf.Print(text_len, [text_len], "text len value")
      batch_size = self.config["batch_size"]
      print("is training: ", is_training.get_shape())
      self.dropout = self.get_dropout(self.config["dropout_rate"], is_training[0])   #[batch_size]
      self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training[0])   #[batch_size]
      self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training[0])   #[batch_size]

      print("context word emb: ", context_word_emb.get_shape())
      # num_sentences = tf.shape(context_word_emb)[0]   #[batch_size]
      # max_sentence_length = tf.shape(context_word_emb)[1]   #[batch_size]
      num_sentences = self.batch_shape(context_word_emb, 0)   #[batch_size]
      max_sentence_length = self.batch_shape(context_word_emb, 1)    #[batch_size]
      print("num sentences: ", num_sentences.get_shape())

      print("text len: ", text_len.get_shape())



      '''
      if self.config["char_embedding_size"] > 0:
        char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), char_index) # [num_sentences, max_sentence_length, max_word_length, emb]
        flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
        flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
        aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
        context_emb_list.append(aggregated_char_emb)
        head_emb_list.append(aggregated_char_emb)   #获得head_emb_list
      '''
      '''
      if not self.lm_file:
        elmo_module = hub.Module("/home/makexin/makexin/elmo/modle/2")
        lm_embeddings = elmo_module(
          inputs={"tokens": tokens, "sequence_len": text_len},
          signature="tokens", as_dict=True)
        word_emb = lm_embeddings["word_emb"]  # [num_sentences, max_sentence_length, 512]
        lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                           lm_embeddings["lstm_outputs1"],
                           lm_embeddings["lstm_outputs2"]], -1)  # [num_sentences, max_sentence_length, 1024, 3]
      '''
      print("lm emb: ", lm_emb.get_shape())
      lm_emb_size = util.shape(lm_emb, 3)
      lm_num_layers = util.shape(lm_emb, 4)

      batch_max_sentence_length = max_sentence_length[tf.argmax(max_sentence_length, 0)]
      batch_max_num_sentence = num_sentences[tf.argmax(num_sentences, 0)]

      #padding context word emb
      context_word_emb_list = []
      for i in range(batch_size):
          context_word_sentence_length = tf.shape(context_word_emb[i])[1]
          context_word_num_sentences_length = tf.shape(context_word_emb[i])[0]
          context_word_emb_tmp = tf.pad(context_word_emb[i], [[0, batch_max_num_sentence-context_word_num_sentences_length], [0, batch_max_sentence_length-context_word_sentence_length], [0, 0]])
          context_word_emb_list.append(context_word_emb_tmp)
      context_word_emb = tf.convert_to_tensor(context_word_emb_list)

      context_emb_list = [context_word_emb]  # [batch_size, num_sentences, max_sentence_length, emb]

      #padding lm_emb
      lm_emb_list = []
      for i in range(batch_size):
          lm_sentence_length = tf.shape(lm_emb[i])[1]
          lm_num_sentence_length = tf.shape(lm_emb[i])[0]
          lm_tmp = tf.pad(lm_emb[i], [[0, batch_max_num_sentence-lm_num_sentence_length], [0, batch_max_sentence_length-lm_sentence_length], [0, 0], [0, 0]])
          lm_emb_list.append(lm_tmp)
      lm_emb_list = tf.convert_to_tensor(lm_emb_list)
      lm_emb = lm_emb_list   #[batch_size, num_sentences, batch_max_sentences_length, 1024, 3]

      #padding text len
      text_len_list = []
      for i in range(batch_size):
          text_len_num_sentence = tf.shape(text_len[i])[0]
          text_len_tmp = tf.pad([text_len[i]], [[0, 0], [0, batch_max_num_sentence-text_len_num_sentence]])
          text_len_list.append(text_len_tmp[0])
      text_len = tf.convert_to_tensor(text_len_list)  #[batch_size, max_num_sentence]

      with tf.variable_scope("lm_aggregation"):
        self.lm_weights = tf.nn.softmax(
          tf.get_variable("lm_scores", [batch_size, lm_num_layers], initializer=tf.constant_initializer(0.0)))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.config['lambda'])(self.lm_weights))
        self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
      #flattened_lm_emb = tf.reshape(lm_emb, [batch_size, num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
      flattened_lm_emb = tf.reshape(lm_emb, [batch_size, -1, lm_num_layers])
      flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights,
                                                                               -1))  # [batch_size, num_sentences * max_sentence_length * emb, 1]
      #aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [batch_size, num_sentences, batch_max_sentence_length, lm_emb_size])
      aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb,
                                     [batch_size, -1, batch_max_sentence_length, lm_emb_size])
      aggregated_lm_emb *= self.lm_scaling
      context_emb_list.append(aggregated_lm_emb)  # 获得context_emb_list      context_emb_list合并了context embedding 和 lm embedding

      context_emb = tf.concat(context_emb_list, 3)  # [batch_size, num_sentences, max_sentence_length, emb]  两个矩阵拼接
      print("context emb shape: ", context_emb.get_shape())
      print("lexical dropout: ", self.lexical_dropout)
      # head_emb = tf.concat(head_emb_list, 2) # [num_sentences, max_sentence_length, emb]
      context_emb = tf.nn.dropout(context_emb,
                                  self.lexical_dropout)  # [batch_size, num_sentences, max_sentence_length, emb]    进行dropout
      #context_emb = tf.debugging.check_numerics(context_emb, "check context_emb")
      #context_emb = tf.Print(context_emb, [context_emb], "context embedding")


      # head_emb = tf.nn.dropout(head_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]

      text_len_mask = tf.sequence_mask(text_len, maxlen=batch_max_sentence_length)  # [batch_size, num_sentence, max_sentence_length]

      context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask,
                                                "topic")  # [batch_size, num_words, emb]   经过一个双向lstm的输出    LSTM输出

      #context_outputs = tf.Print(context_outputs, [context_outputs], "context outputs")
      #context_outputs = tf.debugging.check_numerics(context_outputs, " check numbers context_outputs")

      # context_outputs = tf.reshape(context_outputs, [batch_size, -1, util.shape(context_outputs, 1)])   #[batch_size, num_words, emb]
      num_words = util.shape(context_outputs, 1)
      print("context outputs: ", context_outputs.get_shape())
      print("num words: ",num_words)

      # genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]), genre) # [emb]

      # sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1),
      #                            [1, max_sentence_length])  # [batch_size, num_sentence, max_sentence_length]
      sentence_indices_list = []
      for i in range(batch_size):
          sentence_indice_tmp = tf.tile(tf.expand_dims(tf.range(num_sentences[i]), 1), [1, batch_max_sentence_length])
          sentence_indices_list.append(sentence_indice_tmp)
      sentence_indices = tf.convert_to_tensor(sentence_indices_list)  # [batch_size, num_sentence, batch_max_sentence_length]
      flattened_sentence_indices = self.flatten_emb_by_batch_sentence(sentence_indices, text_len_mask)  # [batch_size, num_words]
      # flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask) # [num_words]

      # candidate start and end 指的是词的起始和结尾   划分span
      candidate_starts = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(num_words), 1), 0),
                                 [batch_size, 1, self.max_span_width])  # [batch_size, num_words, max_span_width]
      print("candidate start shape: ", candidate_starts.get_shape())
      candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width),
                                                         0)  # [batch_size, num_words, max_span_width]
      print("candidate ends shape: ", candidate_ends.get_shape())
      candidate_start_sentence_indices = tf.gather(flattened_sentence_indices,
                                                   candidate_starts)  # [batch_size, num_words, max_span_width]
      candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends,
                                                                                        num_words - 1))  # [batch_size, num_words, max_span_width]
      print("candidate  start sentence indices: ", candidate_start_sentence_indices.get_shape())
      candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices,
                                                                           candidate_end_sentence_indices))  # [batch_size, num_words, max_span_width]
      flattened_candidate_mask = tf.reshape(candidate_mask, [batch_size, -1])  # [batch_size, num_words * max_span_width]
      print("flattened candidate mask； ", flattened_candidate_mask.get_shape())
      candidate_starts_list = []
      candidate_ends_list = []
      for i in range(batch_size):
          candidate_starts_tmp = tf.boolean_mask(tf.reshape(candidate_starts[i], [-1]),
                                         flattened_candidate_mask[i])  # [num_candidates]
          candidate_starts_list.append(candidate_starts_tmp)
          candidate_ends_tmp = tf.boolean_mask(tf.reshape(candidate_ends[i], [-1]), flattened_candidate_mask[i])  # [num_candidates]
          candidate_ends_list.append(candidate_ends_tmp)
      candidate_starts = tf.convert_to_tensor(candidate_starts_list)
      candidate_ends = tf.convert_to_tensor(candidate_ends_list)
      # candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [batch_size, -1]),
      #                                           flattened_candidate_mask)  # [batch_size, num_candidates]

      #candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [batch_size, -1]), flattened_candidate_mask)  # [num_candidates]

      candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [batch_size, -1]),
                                                   flattened_candidate_mask)  # [num_candidates]

      topic_start = topic_start_index
      topic_end = topic_end_index
      print("topic start: {}, topic end {}".format(topic_start, topic_end))
      # 获得topic的embedding
      topic_emb = self.get_span_emb(context_outputs, topic_start, topic_end)  # [batch_size, 1, emb]
      # topic_emb = tf.Print(topic_emb, [topic_emb], "topic emb , after get span emb")
      print("topic emb: ", topic_emb.get_shape())
      print("candidate starts: ", candidate_starts)

      # candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids) # [num_candidates]

      # 获得span的embedding, 给出span的开始和结束索引
      candidate_span_emb = self.get_span_emb(context_outputs, candidate_starts, candidate_ends)  # [batch_size, num_candidates, emb]
      print("candidate span emb : ", candidate_span_emb.get_shape())
      candidate_mention_scores = self.get_mention_scores(candidate_span_emb, "topic")  # [batch_size, k, 1]   #通过一个前馈神经网络获得值
      print("candidate mention scores: ", candidate_mention_scores.get_shape())
      candidate_mention_scores = tf.squeeze(candidate_mention_scores, 2)  # [batch_size, k]

      k = tf.to_int32(tf.floor(tf.to_float(tf.shape(context_outputs)[1]) * self.config["top_span_ratio"]))  # 确定top k
      print("top k值: ", k)
      # 使用自定义的操作对span排序
      # top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
      #                                            tf.expand_dims(candidate_starts, 0),
      #                                            tf.expand_dims(candidate_ends, 0),
      #                                            tf.expand_dims(k, 0),
      #                                            util.shape(context_outputs, 0),
      #                                            True)  # [1, k]
      top_span_indices_list = []
      for i in range(batch_size):
          top_span_indices_tmp = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores[i], 0),
                                                 tf.expand_dims(candidate_starts[i], 0),
                                                 tf.expand_dims(candidate_ends[i], 0),
                                                 tf.expand_dims(k, 0),
                                                 util.shape(context_outputs[i], 0),
                                                 True)  # [1, k]
          top_span_indices_list.append(top_span_indices_tmp)
      top_span_indices = tf.convert_to_tensor(top_span_indices_list)
      print("top span indices shape: ", top_span_indices.get_shape())
      top_span_indices.set_shape([batch_size, 1, None])
      top_span_indices = tf.squeeze(top_span_indices, 1)  # [batch_size, k]
      print("top span indices shape: ", top_span_indices.get_shape())

      top_span_starts = tf.gather(candidate_starts, top_span_indices)  # [k]   根据选出的top k span，将对应span的start位置取出
      top_span_ends = tf.gather(candidate_ends, top_span_indices)  # [k]
      top_span_emb = self.change_batch_gather(candidate_span_emb, top_span_indices)  # [batch_size, k, emb]
      #top_span_emb = tf.gather(candidate_span_emb, top_span_indices)  # [batch_size, k, emb]
      print("top span emb: ", top_span_emb.get_shape())

      top_span_mention_scores = self.change_batch_gather(candidate_mention_scores, top_span_indices)   # [batch_size, k]
      print(" top span mention scores: ", top_span_mention_scores.get_shape())
      #top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)  # [k]
      top_span_sentence_indices = tf.gather(candidate_sentence_indices, top_span_indices)  # [k]

      # top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts) # [k]      speaker 用不到

      c = tf.minimum(self.config["max_top_antecedents"], k)  # 每个span，最多c个先行词

      # 剪枝
      if self.config["coarse_to_fine"]:
        top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(
          top_span_emb, top_span_mention_scores, c)
      else:
        top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.distance_pruning(
          top_span_emb, top_span_mention_scores, c) # distance_pruning 函数还没有改

      dummy_scores = tf.zeros([batch_size, k, 1])  # [batch_size, k, 1]
      for i in range(self.config["coref_depth"]):
        with tf.variable_scope("coref_layer", reuse=(i > 0)):
          #top_antecedent_emb = tf.gather(top_span_emb, top_antecedents)  # [batch_size, k, c, emb]
          top_antecedent_emb = util.bach_gather_with_equal_shape(top_span_emb, top_antecedents)  # [batch_size, k, c, emb]
          print("top antecedent emb: ", top_antecedent_emb.get_shape())
          # top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb,
          #                                                                                      top_antecedents,
          #                                                                                      top_antecedent_emb,
          #                                                                                      top_antecedent_offsets)  # [k, c]  考虑了距离、相似性、元信息等特征计算的得分
          top_antecedent_scores = top_fast_antecedent_scores  # [batch_size, k, c]
          top_antecedent_scores = tf.debugging.check_numerics(top_antecedent_scores, "top antecedent_scores 1")
          top_antecedent_scores = top_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb,
                                                                                                top_antecedents,
                                                                                                top_antecedent_emb,
                                                                                                top_antecedent_offsets)   # [batch_size, k, c]
          #top_antecedent_scores = tf.debugging.check_numerics(top_antecedent_scores, "top antecedent_scores 2")
          top_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, top_antecedent_scores], 2))  # [batch_size, k, c + 1]
          tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.config['lambda'])(top_antecedent_weights))
          top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 2), top_antecedent_emb], 2)  # [batch_size, k, c + 1, emb]
          attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 3) * top_antecedent_emb,
                                            2)  # [batch_size, k, emb]
          with tf.variable_scope("f"):
            f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 2),
                                           util.shape(top_span_emb, -1)))  # [batch_size, k, emb]
            top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb  # [batch_size, k, emb]

      top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 2)  # [batch_size, k, c + 1]
      '''

      top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents) # [k, c]   top antecedent对应的cluster id
      top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask))) # [k, c]
      same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1)) # [k, c]
      non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1) # [k, 1]
      pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator) # [k, c]
      dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True)) # [k, 1]
      top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1) # [k, c + 1]

      '''

      # loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels) # [k]   计算损失函数，但是没有用到实际的label呀
      # loss = tf.reduce_sum(loss) # []

      # topic_emb = tf.Print(topic_emb, [topic_emb], "topic embedding")
      # top_span_emb = tf.Print(top_span_emb, [top_span_emb], "top span embedding value")
      # top_span_mention_scores = tf.Print(top_span_mention_scores, [top_span_mention_scores], "top span mention scores")
      # top_antecedent_emb = tf.Print(top_antecedent_emb, [top_antecedent_emb], "top antecedent embedding value")
      # top_antecedent_scores = tf.Print(top_antecedent_scores, [top_antecedent_scores], "top antecedent scores")

      return topic_emb, top_span_emb, top_span_mention_scores, top_antecedent_emb, top_antecedent_scores

  def change_batch_gather(self, x, indice):
      tensor_list = []
      for i in range(self.config["batch_size"]):
          tmp = tf.gather(x[i], indice[i])
          tensor_list.append(tmp)
      return tf.convert_to_tensor(tensor_list)

  def get_span_emb(self, context_outputs, span_starts, span_ends):
    print("span starts shape: ", span_starts.get_shape())
    print("span ends shape: ", span_ends.get_shape())
    span_emb_list = []

    span_tmp_emb = []
    for i in range(self.config["batch_size"]):
        span_start_emb = tf.gather(context_outputs[i], span_starts[i])  # [k, emb]   此处输入context_output是一个句子的集合？还是句子中每个单词的集合
        span_tmp_emb.append(span_start_emb)
    span_emb_list.append(tf.convert_to_tensor(span_tmp_emb))
    print("span tmp emb: ", tf.convert_to_tensor(span_tmp_emb))

    span_tmp_emb = []
    for i in range(self.config["batch_size"]):
        span_end_emb = tf.gather(context_outputs[i],
                                   span_ends[i])  # [k, emb]   此处输入context_output是一个句子的集合？还是句子中每个单词的集合
        span_tmp_emb.append(span_end_emb)
    span_emb_list.append(tf.convert_to_tensor(span_tmp_emb))


    # span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]   此处输入context_output是一个句子的集合？还是句子中每个单词的集合
    # span_emb_list.append(span_start_emb)
    #
    # span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]
    # span_emb_list.append(span_end_emb)

    #span_width = 1 + span_ends - span_starts # [k]
    #span_width = list(map(operator.sub, [x+1 for x in span_ends], span_starts))


    '''
    if self.config["use_features"]:    #span也考虑特征
      span_width_index = span_width - 1 # [k]
      span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]), span_width_index) # [k, emb]
      span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
      span_emb_list.append(span_width_emb)
    '''

    '''
    if self.config["model_heads"]:   #head attention部分  猜测
      span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts, 1) # [k, max_span_width]
      span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices) # [k, max_span_width]
      span_text_emb = tf.gather(head_emb, span_indices) # [k, max_span_width, emb]
      with tf.variable_scope("head_scores"):
        self.head_scores = util.projection(context_outputs, 1) # [num_words, 1]
      span_head_scores = tf.gather(self.head_scores, span_indices) # [k, max_span_width, 1]
      span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32), 2) # [k, max_span_width, 1]
      span_head_scores += tf.log(span_mask) # [k, max_span_width, 1]
      span_attention = tf.nn.softmax(span_head_scores, 1) # [k, max_span_width, 1]
      span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1) # [k, emb]
      span_emb_list.append(span_head_emb)
    '''
    print("span_emb_list shap： ", np.array(span_emb_list).shape)
    span_emb = tf.concat(span_emb_list, 2) # [batch_size, k, emb]   生成的各个embedding拼接起来
    return span_emb # [batch_size, k, emb]


  def get_mention_scores(self, span_emb, topic):
    with tf.variable_scope("{}".format(topic)):
      with tf.variable_scope("mention_scores"):
        return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout)  # [k, 1]

  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [k, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [k]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [k]
    return log_norm - marginalized_gold_scores # [k]

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)

  def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets):
    '''

    :param top_span_emb:  [batch_size, k, emb]
    :param top_antecedents: [batch_size, k, c]
    :param top_antecedent_emb: [batch_size, k, c, emb]
    :param top_antecedent_offsets: [batch_size, k, c]
    :return:
    '''
    k = util.shape(top_span_emb, 1)
    c = util.shape(top_antecedents, 2)

    feature_emb_list = []

    '''
    if self.config["use_metadata"]:   #该方法中所用到speaker、genre信息都没有
      top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents) # [k, c]
      same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids) # [k, c]   #获得speaker信息
      speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]), tf.to_int32(same_speaker)) # [k, c, emb]
      feature_emb_list.append(speaker_pair_emb)

      tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1]) # [k, c, emb]
      feature_emb_list.append(tiled_genre_emb)
    '''

    if self.config["use_features"]:   #考虑特征信息，距离特征
      print("top antecedent offsets: ", top_antecedent_offsets.get_shape())
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [batch_size, k, c]
      print("antecedent distance buckets: ", antecedent_distance_buckets.get_shape())
      #antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [self.config["batch_size"], 10, self.config["feature_size"]]), antecedent_distance_buckets) # [batch_size, k, c]
      antecedent_distance_emb = util.bach_gather_with_equal_shape(tf.get_variable("antecedent_distance_emb", [self.config["batch_size"], 10, self.config["feature_size"]]), antecedent_distance_buckets)
      feature_emb_list.append(antecedent_distance_emb)

    print(" feature embedding list shape: ", np.array(feature_emb_list).shape)
    feature_emb = tf.concat(feature_emb_list, 3) # [batch_size, k, c, emb]
    print("feature emb: ", feature_emb.get_shape())
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [batch_size, k, c, emb]
    print("feature emb 2: ", feature_emb.get_shape())

    target_emb = tf.expand_dims(top_span_emb, 2) # [batch_size, k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb # [batch_size, k, c, emb]    考虑两个span的相似性
    target_emb = tf.tile(target_emb, [1, 1, c, 1]) # [batch_size, k, c, emb]

    pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 3) # [batch_size, k, c, emb]    span对的embedding

    with tf.variable_scope("slow_antecedent_scores"):
      slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [batch_size, k, c, 1]
    slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 3) # [batch_size, k, c]
    return slow_antecedent_scores # [batch_size, k, c]

  def get_fast_antecedent_scores(self, top_span_emb):
      # top_span_emb: [batch_size, k, emb]

    with tf.variable_scope("src_projection"):
      source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)), self.dropout) # [batch_size, k, emb]
    target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout) # [batch_size, k, emb]
    return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [batch_size, k, k]   两个矩阵相乘

  def flatten_emb_by_sentence(self, emb, text_len_mask):   #改变多维形状
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())

    if emb_rank == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    elif emb_rank == 4:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2), util.shape(emb, 3) ])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))

    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))
  def flatten_emb_with_rank_4(self, emb, text_len_mask):
      batch_size = self.config["batch_size"]
      result = []
      max_num_length = 0
      max_num_length = tf.convert_to_tensor(max_num_length)
      for i in range(batch_size):
          tmp = self.flatten_emb_by_sentence(emb[i], text_len_mask[i])
          result.append(tmp)
          print("max : ", util.shape(tmp, 0))
          max_num_length = tf.reduce_max([max_num_length, util.shape(tmp, 0)], 0)

      pad_result_list = []
      for i in range(batch_size):
          num_length = util.shape(result[i], 0)
          result_tmp = tf.pad(result[i], [[0, max_num_length-num_length], [0, 0]])
          pad_result_list.append(result_tmp)
      pad_result = tf.convert_to_tensor(pad_result_list)  # [batch_size, num_words, emb]
      return tf.convert_to_tensor(pad_result)
  def flatten_emb_by_batch_sentence(self, emb, text_len_mask):
    batch_size = tf.shape(emb)[0]
    num_sentences = tf.shape(emb)[1]
    max_sentence_length = tf.shape(emb)[2]

    emb_rank = len(emb.get_shape())

    if emb_rank == 3:
      flattened_emb = tf.reshape(emb, [batch_size , num_sentences * max_sentence_length])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [batch_size , num_sentences * max_sentence_length]))

  def lstm_contextualize(self, text_emb, text_len, text_len_mask, topic):
      #text_len_mask: [batch_size, num_sentence]
    batch_size = text_emb.get_shape()[0].value
    print("batch size :", batch_size)
    num_sentences = tf.shape(text_emb)[1]
    batch_max_sentence_length = tf.shape(text_emb)[2]
    emb = text_emb.get_shape()[3].value
    print("emb: ", emb)
    #num_sentences = self.batch_shape(text_emb, 0)  #[batch_size]
    #num_sentences = tf.shape(text_emb)[0]

    current_inputs = text_emb # [batch_size, num_sentences, max_sentence_length, emb]
    current_inputs = tf.reshape(current_inputs, [batch_size*num_sentences, batch_max_sentence_length, emb])   #[batch_size*num_sentences, batch_max_sentence_length, emb]
    print("text emb: ", text_emb.get_shape())

    current_num_sentences = tf.shape(current_inputs)[0]


    for layer in range(self.config["contextualization_layers"]):
      with tf.variable_scope("{}".format(topic)):
        with tf.variable_scope("layer_{}".format(layer)):
          with tf.variable_scope("fw_cell"):
            ## cell_fw 前向
            cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], current_num_sentences, self.lstm_dropout)
          with tf.variable_scope("bw_cell"):
            #  cell_bw  后向
            cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], current_num_sentences, self.lstm_dropout)
          # LSTMStateTuple   用于存储LSTM单元的state_size,zero_state和output state的元组。按顺序存储两个元素(c,h),其中c是隐藏状态，h是输出。
          state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [current_num_sentences, 1]),
                                                   tf.tile(cell_fw.initial_state.h, [current_num_sentences, 1]))
          state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [current_num_sentences, 1]),
                                                   tf.tile(cell_bw.initial_state.h, [current_num_sentences, 1]))

          current_text_len = tf.reshape(text_len, [-1])

          print("current input: ", current_inputs.get_shape())
          # 双向LSTM
          (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=current_inputs,
            sequence_length=current_text_len,
            initial_state_fw=state_fw,
            initial_state_bw=state_bw)

          print("fw outputs: ", fw_outputs.get_shape())
          print("bw outputs: ", bw_outputs.get_shape())
          text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [batch_size * num_sentences, max_sentence_length, emb]  将输出拼接
          text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)


          if layer > 0:
            highway_gates = tf.sigmoid(
              util.projection(text_outputs, util.shape(text_outputs, 2)))  # [batch_size, num_sentences, max_sentence_length, emb]
            text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
          emb_size = util.shape(current_inputs, 2)
          current_inputs = text_outputs
          text_outputs = tf.reshape(text_outputs, [batch_size, num_sentences, batch_max_sentence_length, -1])# [batch_size, num_sentences, max_sentence_length, emb]
          print("text outputs: ", text_outputs.get_shape())
          print("text len mask: ", text_len_mask.get_shape())


          outputs = self.flatten_emb_with_rank_4(text_outputs, text_len_mask)
          outputs = tf.reshape(outputs, [batch_size, -1, emb_size])
          print("outputs: ", outputs.get_shape())
          # outputs_emb = util.shape(outputs, 1)
          # outputs = tf.reshape(outputs, [batch_size, -1])
          # outputs = tf.reshape(outputs, [batch_size, -1, outputs_emb])
          # print("outputs: ", outputs.get_shape())
    return outputs

  def get_predicted_antecedents(self, antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
      if index < 0:
        predicted_antecedents.append(-1)
      else:
        predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents

  def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index
      predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(top_span_starts[i]), int(top_span_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

    return predicted_clusters, mention_to_predicted

  def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters

  def load_eval_data(self):
    if self.eval_data is None:
      def load_line(line):
        example = json.loads(line)
        return self.tensorize_example(example, is_training=False), example
      with open(self.config["eval_path"]) as f:    #加载测试数据集
        self.eval_data = [load_line(l) for l in f.readlines()]
      #num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
      print("Loaded {} eval examples.".format(len(self.eval_data)))

  def evaluate(self, session, official_stdout=False):
    self.load_eval_data()

    coref_predictions = {}
    #coref_evaluator = metrics.CorefEvaluator()

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):    # example为载入的原始json数据
      pre_topic_start_index, pre_topic_end_index, post_topic_start_index, post_topic_end_index, pre_tokens, post_tokens, pre_context_word_emb, post_context_word_emb, pre_lm_emb, post_lm_emb, pre_text_len, post_text_len, is_training, label = tensorized_example
      #_, _, _, _, _, _, _, _, _, gold_starts, gold_ends, _ = tensorized_example
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      #candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(self.predictions, feed_dict=feed_dict)

      pre_topic_emb, post_topic_emb, pre_top_span_emb, post_top_span_emb, pre_top_span_mention_scores, post_top_span_mention_scores, pre_top_antecedent_emb, post_top_antecedent_emb, pre_top_antecedent_scores, post_top_antecedent_scores, final_score = session.run(self.predictions, feed_dict=feed_dict)

      #pre_predicted_antecedents = self.get_predicted_antecedents(pre_top_antecedent_emb, pre_top_antecedent_scores)

      file_key = example["pre_topic"] + "_" + example["post_topic"]
      original_label = int(example["label"])
      if original_label != 1:
        original_label = 0
      #coref_predictions[file_key] = self.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"], coref_evaluator)
      coref_predictions[file_key] = [final_score, original_label, example["pre_topic"], example["post_topic"]]
      if example_num % 10 == 0:
        print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

    summary_dict = {}
    #conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
    #average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    precision, recall, f = evaluate.evaluate(coref_predictions, self.config)
    summary_dict["Average F1 (py)"] = f
    print("Average F1 (py): {:.2f}%".format(f))

    #p,r,f = coref_evaluator.get_prf()

    summary_dict["Average precision (py)"] = precision
    print("Average precision (py): {:.2f}%".format(precision))
    summary_dict["Average recall (py)"] = recall
    print("Average recall (py): {:.2f}%".format(recall))

    return util.make_summary(summary_dict), f, precision, recall

  def batch_shape(self, tensor, dim):
      batch_size = self.config["batch_size"]
      result_shape = [ 0 for _ in range(batch_size)]
      for i in range(batch_size):
          result_shape[i] = tf.shape(tensor[i])[dim]
      return tf.convert_to_tensor(result_shape)