#coding:utf-8
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import coref_model as cm
import util
from tensorflow.python import debug as tf_debug



if __name__ == "__main__":
  config = util.initialize_from_env()

  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]


  # cluster_config = config["cluster"]
  # util.set_gpus(*cluster_config["gpus"])

  model = cm.CorefModel(config)
  saver = tf.train.Saver()

  log_dir = config["log_dir"]
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_f1 = 0
  coord = tf.train.Coordinator()
  with tf.Session() as session:

    # session = tf_debug.LocalCLIDebugWrapperSession(session)
    # session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    #session.run(tf.initialize_all_variables())
    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0

    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)

    initial_time = time.time()
    while True:
      tf_loss, post_topic_relation_score, pre_topic_relation_score, label, tf_global_step, _ = session.run([model.loss, model.post_topic_relation_score, model.pre_topic_relation_score, model.label, model.global_step, model.train_op])
      accumulated_loss += tf_loss
      print("loss: {}, label: {}, post_topic_relation_score: {}, pre_topic_relation_score: {}".format(tf_loss, label, post_topic_relation_score, pre_topic_relation_score))


      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0

      if tf_global_step % eval_frequency == 0:
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        eval_summary, eval_f1, eval_precision, eval_recall = model.evaluate(session)

        if eval_f1 > max_f1:
          max_f1 = eval_f1
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

        writer.add_summary(eval_summary, tf_global_step)


        print("[{}] evaL_f1={:.2f}, max_f1={:.2f}, eval_precision={:.2f}, eval_recall={:.2f}".format(tf_global_step, eval_f1, max_f1, eval_precision, eval_recall))
