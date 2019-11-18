#coding:utf-8
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import coref_model as cm
import util
from tensorflow.python import debug as tf_debug
from sklearn import metrics
from tensorboard import summary as summary_lib
from datetime import datetime



if __name__ == "__main__":
  config = util.initialize_from_env()

  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]


  # cluster_config = config["cluster"]
  # util.set_gpus(*cluster_config["gpus"])

  model = cm.CorefModel(config)
  saver = tf.train.Saver()

  TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
  log_dir = config["log_dir"]
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_f1 = 0
  max_presission = 0
  max_recall = 0
  coord = tf.train.Coordinator()
  with tf.Session() as session:

    # session = tf_debug.LocalCLIDebugWrapperSession(session)
    # session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    #session.run(tf.initialize_all_variables())
    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0

    writer.add_graph(session.graph)
    merged_summary = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)

    initial_time = time.time()
    pred = []  # 预测结果
    true = []  # 实际标签
    while True:

      tf_loss, post_topic_relation_score, pre_topic_relation_score, label, predict, tf_global_step, _ = session.run([model.loss, model.post_topic_relation_score, model.pre_topic_relation_score, model.label, model.predict, model.global_step, model.train_op])

      print("loss: {}, label: {}, predict: {}, post_topic_relation_score: {}, pre_topic_relation_score: {}".format(tf_loss, label, predict, post_topic_relation_score, pre_topic_relation_score))

      train_summary = session.run(merged_summary)
      writer.add_summary(train_summary)
      accumulated_loss += tf_loss

      #训练数据结果
      if predict > config["result_metric"]:
        pred.append(1)
      else:
        pred.append(0)
      true.append(label)

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)

        accumulated_loss = 0.0

      if tf_global_step % eval_frequency == 0:

        #训练集评价结果
        train_accuracy = metrics.accuracy_score(true, pred)
        train_precision_macro = metrics.precision_score(true, pred, average='macro')
        train_recall_macro = metrics.recall_score(true, pred, average='macro')
        train_f = metrics.f1_score(true, pred, average='macro')
        summary_dict = {}
        summary_dict["train F1"] = train_f
        summary_dict["train accuracy"] = train_accuracy
        summary_dict["train precision"] = train_precision_macro
        summary_dict["train recall"] = train_recall_macro
        writer.add_summary(util.make_summary(summary_dict), tf_global_step)
        print(
          "[{}] train_f1={:.4f}, train_accuracy={:.4f}, train_precision={:.4f}, train_recall={:.4f}".format(
            tf_global_step, train_f, train_accuracy, train_precision_macro, train_recall_macro))
        #重新置为空
        pred = []  # 预测结果
        true = []  # 实际标签

        #测试集结果
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        eval_summary, accuracy, f, precision, recall = model.evaluate(session)

        if f > max_f1:
          max_f1 = f
          max_presission = precision
          max_recall = recall
          max_accuracy = accuracy
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)),
                               os.path.join(log_dir, "model.max.ckpt"))

        writer.add_summary(eval_summary, tf_global_step)

        print(
          "[{}] max_accuracy={:.4f} max_f1={:.4f}, max_precision={:.4f}, max recall={:.4f}".format(tf_global_step, max_accuracy, max_f1, max_presission,
                                                                               max_recall))
        print(
          "[{}] eval_accuracy={:.4f}, evaL_f1_macro={:.4f}, eval_precision_macro={:.4f}, eval_recall_macro={:.4f}".format(
            tf_global_step, accuracy, f, precision, recall))
