import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.python.keras import backend as K

'''
计算elmo词向量
并计算余弦相似性
'''

class Simility():

    def elmo_vector(self,x):
        elmo = hub.Module("D:/Experiment/elmo/modle/2")
        embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
        print(embeddings.shape)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            array = K.get_session().run(embeddings)
            for i in range(x.size):
                print(array[0, i, :])
            return sess.run(tf.reduce_mean(embeddings, 1))  # 求所有行向量之间的平均值

    def cosine_simility_for_word(self,x, y):
        with tf.Session() as sess:
            # 求模
            x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=0))
            y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=0))
            # 内积
            xy = tf.reduce_sum(tf.multiply(x, y), axis=0)
            cosin = tf.divide(xy, tf.multiply(x_norm, y_norm))
            sess.run(tf.global_variables_initializer())
            return sess.run(cosin)

    def get_simility(self,topic1,topic2):
        nptopic1 = np.array(topic1)
        nptopic2 = np.array(topic2)
        score = self.cosine_simility_for_word(self.elmo_vector(nptopic1)[0,:],self.elmo_vector(nptopic2)[0,:])
        return score