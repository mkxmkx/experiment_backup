import codecs
import random

def split_train_eval():
    readfile = codecs.open("/home/makexin/makexin/Experiment/data/Chinese/dataStructure/all_topic_pair_sentences_data.json", 'r', 'utf-8')
    train_file = codecs.open("/home/makexin/makexin/Experiment/data/Chinese/dataStructure/train_topic_pair_sentences_train_data.json", 'w', 'utf-8')
    eval_file = codecs.open("/home/makexin/makexin/Experiment/data/Chinese/dataStructure/eval_topic_pair_sentences_train_data.json", 'w', 'utf-8')
    # readfile = codecs.open(
    #     "/home/makexin/makexin/Experiment/data/v2/topic_pair_sentences_train_data.json", 'r', 'utf-8')
    # train_file = codecs.open(
    #     "/home/makexin/makexin/Experiment/data/v2/train_topic_pair_sentences_train_data.json", 'w',
    #     'utf-8')
    # eval_file = codecs.open(
    #     "/home/makexin/makexin/Experiment/data/v2/eval_topic_pair_sentences_train_data.json", 'w',
    #     'utf-8')
    while True:
        line = readfile.readline().strip()
        if not line:
            break
        train_file.write(line)
        train_file.write('\n')
        x = random.random()
        if x>=0.7:
            eval_file.write(line)
            eval_file.write('\n')
        # else:
        #     train_file.write(line)
        #     train_file.write('\n')
    readfile.close()
    train_file.close()
    eval_file.close()

split_train_eval()