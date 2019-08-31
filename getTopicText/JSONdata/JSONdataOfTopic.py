import codecs
import json
import os
import nltk
from JSONdata.getTopicFromCSV import CSVURL

'''
生成用于训练elmo词向量的json文件
'''

def getJSONdata():


    JSONfile = codecs.open("D:/Experiment/CrowdComp_spiderData/topic_sentences.json","w","utf-8")
    #JSONfile = codecs.open("D:/Experiment/spiderData_v4/Global_warming/topic_sentences.json", "w", "utf-8")

    csv = CSVURL()
    id_topic_file_0 = "D:/Experiment/CrowdComp_spiderData/Global_warming/id_topic.csv"
    id_topic_file_1 = "D:/Experiment/CrowdComp_spiderData/Meiosis/id_topic.csv"
    id_topic_file_2 = "D:/Experiment/CrowdComp_spiderData/Newton_Laws/id_topic.csv"
    id_topic_file_3 = "D:/Experiment/CrowdComp_spiderData/Parallel_Postulate/id_topic.csv"
    id_topic_file_4 = "D:/Experiment/CrowdComp_spiderData/PublicKeyCryptography/id_topic.csv"
    result = []
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for i in range(5):
        if i == 0:
            csv_read_file = id_topic_file_0
            filename_dir = "D:/Experiment/CrowdComp_spiderData/Global_warming/"
        elif i == 1:
            csv_read_file = id_topic_file_1
            filename_dir = "D:/Experiment/CrowdComp_spiderData/Meiosis/"
        elif i == 2:
            csv_read_file = id_topic_file_2
            filename_dir = "D:/Experiment/CrowdComp_spiderData/Newton_Laws/"
        elif i == 3:
            csv_read_file = id_topic_file_3
            filename_dir = "D:/Experiment/CrowdComp_spiderData/Parallel_Postulate/"
        else:
            csv_read_file = id_topic_file_4
            filename_dir = "D:/Experiment/CrowdComp_spiderData/PublicKeyCryptography/"
        topic_id_list = csv.csv_read(csv_read_file)
        for id in topic_id_list:
            filename = filename_dir + str(id) + ".txt"

            if (os.path.exists(filename)):
                print("filename : ", filename, " exist")
                readFile = codecs.open(filename, 'r', 'utf-8')
                temp = {}
                temp['topic'] = topic_id_list[id]
                all_sentence = []
                paragraph = []
                while True:  # 对于每一个段落，即读入的一行字符。进行分句
                    line = readFile.readline().strip()
                    if not line:
                        break
                    sentence = sen_tokenizer.tokenize(line)  # 进行分句
                    # sentence_list = []
                    for s in sentence:
                        word_list = nltk.word_tokenize(s)  # 每一句分词，将标点符号与单词分开
                        # sentence_list.append(word_list)
                        all_sentence.append(word_list)
                    # paragraph.append(sentence_list)   #一个段落为一个list，其中每个元素为一句话
                temp['sentences'] = all_sentence
                result.append(temp)
                readFile.close()
    JSONfile.write(json.dumps(result,ensure_ascii=False))
    JSONfile.close()

getJSONdata()





