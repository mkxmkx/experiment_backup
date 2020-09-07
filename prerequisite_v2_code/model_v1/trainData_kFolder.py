import codecs
import random
##将训练数据划分为k份（k根据比例划分），方便进行k折交叉验证
def kFolder():
    readfile = codecs.open("/home/makexin/makexin/Experiment/data/v2/train_topic_pair_sentences_train_data.json", 'r', 'utf-8')
    train_file_1 = codecs.open("/home/makexin/makexin/Experiment/data/v2/K_folder/2_k/train_topic_pair_sentences_train_data_1.json", 'w', 'utf-8')
    train_file_2 = codecs.open(
        "/home/makexin/makexin/Experiment/data/v2/K_folder/2_k/train_topic_pair_sentences_train_data_2.json", 'w',
        'utf-8')
    # train_file_3 = codecs.open(
    #     "/home/makexin/makexin/Experiment/data/v2/K_folder/4_k/train_topic_pair_sentences_train_data_3.json", 'w',
    #     'utf-8')
    # train_file_4 = codecs.open(
    #     "/home/makexin/makexin/Experiment/data/v2/K_folder/4_k/train_topic_pair_sentences_train_data_4.json", 'w',
    #     'utf-8')
    # train_file_5 = codecs.open(
    #     "/home/makexin/makexin/Experiment/data/v2/K_folder/5_k/train_topic_pair_sentences_train_data_5.json", 'w',
    #     'utf-8')
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
        x = random.random()
        if(x<=0.5):
            train_file_1.write(line)
            train_file_1.write('\n')
        else:
            train_file_2.write(line)
            train_file_2.write('\n')
        # elif(x<=0.75):
        #     train_file_3.write(line)
        #     train_file_3.write('\n')
        # else:
        #     train_file_4.write(line)
        #     train_file_4.write('\n')


    readfile.close()
    train_file_1.close()
    train_file_2.close()
    # train_file_3.close()
    # train_file_4.close()



kFolder()