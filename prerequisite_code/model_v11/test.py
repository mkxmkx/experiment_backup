import codecs
import tensorflow as tf

def get_glove_embedding():
    input_file = codecs.open("/home/makexin/makexin/Experiment/data/coreference_data/glove.840B.300d.txt",'r','utf-8')
    count = 0
    while 1:
        if count<100:
            line = input_file.readline().strip()
            print("in while")
            print(line)
            print("\n")
            count += 1
        else:
            break
    input_file.close()

a = tf.constant([1, 2, 3])
b = tf.constant([1, 2, 3])
c = set()

c.add(a)
c.add(b)

with tf.Session() as sess:
    print(sess.run(c))
