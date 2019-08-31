import codecs
import json
import os
import nltk
from JSONdata.getTopicFromCSV import CSVURL
import jsonlines

#生成模型输入数据， 每一对需判断的topic

def getJSONLinesdata():

    jsonlines_file_name = "D:\Experiment\CrowdComp_spiderData/all_topic_pair_sentences_train_data.json"
    #jsonlines_file_name = "D:/Experiment/spiderData_v2/topic_pair_sentences_train_data.json"
    #error_topic_file = "D:/Experiment/spiderData_v2/error_topic.json"   #记录无法在文本中精确匹配到topic的topic
    JSONfile = codecs.open(jsonlines_file_name,"w","utf-8")
    #error_topic = codecs.open(error_topic_file,'w','utf-8')
    id_topic_file_0 = "D:/Experiment/CrowdComp_spiderData/Global_warming/id_topic.csv"
    id_topic_file_1 = "D:/Experiment/CrowdComp_spiderData/Meiosis/id_topic.csv"
    id_topic_file_2 = "D:/Experiment/CrowdComp_spiderData/Newton_Laws/id_topic.csv"
    id_topic_file_3 = "D:/Experiment/CrowdComp_spiderData/Parallel_Postulate/id_topic.csv"
    id_topic_file_4 = "D:/Experiment/CrowdComp_spiderData/PublicKeyCryptography/id_topic.csv"
    count = 0
    for i in range(5):
        if i == 0:
            csv_read_file = "D:\Experiment\CrowdComp_spiderData/Meiosis/id_topic.csv"
            csv_pair_file = "D:\Experiment\CrowdComp_spiderData/Meiosis/CrowdComp_Meiosis_topicPair.csv"
            by_name = "CrowdComp_Meiosis_topicPair"
            file_dir = "D:\Experiment\CrowdComp_spiderData/Meiosis/"
        if i == 1:
            csv_read_file = "D:\Experiment\CrowdComp_spiderData/Newton_Laws/id_topic.csv"
            csv_pair_file = "D:\Experiment\CrowdComp_spiderData/Newton_Laws/CrowdComp_Newton_Laws_topicPair.csv"
            by_name = "CrowdComp_Newton_Laws_topicPair"
            file_dir = "D:\Experiment\CrowdComp_spiderData/Newton_Laws/"
        elif i == 2:
            csv_read_file = "D:\Experiment\CrowdComp_spiderData/Parallel_Postulate/id_topic.csv"
            csv_pair_file = "D:\Experiment\CrowdComp_spiderData/Parallel_Postulate/CrowdComp_Parallel_Postulate_topicPair.csv"
            by_name = "CrowdComp_Parallel_Postulate_to"
            file_dir = "D:\Experiment\CrowdComp_spiderData/Parallel_Postulate/"
        elif i == 3:
            csv_read_file = "D:\Experiment\CrowdComp_spiderData\PublicKeyCryptography/id_topic.csv"
            csv_pair_file = "D:\Experiment\CrowdComp_spiderData/PublicKeyCryptography/CrowdComp_PublicKeyCryptography_topicPair.csv"
            by_name = "CrowdComp_PublicKeyCryptography"
            file_dir = "D:\Experiment\CrowdComp_spiderData/PublicKeyCryptography/"
        else:
            csv_read_file = "D:\Experiment\CrowdComp_spiderData\Global_warming/id_topic.csv"
            csv_pair_file = "D:\Experiment\CrowdComp_spiderData/Global_warming/CrowdComp_Global_warming_topicPair.csv"
            by_name = "CrowdComp_Global_warming_topicP"
            file_dir = "D:\Experiment\CrowdComp_spiderData/Global_warming/"


        csv = CSVURL()
        topic_id_list = csv.csv_read(csv_read_file)
        prerequisite_pair_result = csv.csv_get_prerequisite_pair(csv_pair_file, by_name)
        # final_result = []

        for pair in prerequisite_pair_result:
            pre_topic = pair["pre_topic"]
            post_topic = pair["post_topic"]
            label = pair["label"]

            pre_topic_filename = file_dir + str(pre_topic) + ".txt"
            post_topic_filename = file_dir + str(post_topic) + ".txt"

            if (file_exist(pre_topic_filename) and file_exist(
                    post_topic_filename) and pre_topic in topic_id_list and post_topic in topic_id_list):
                result = {}
                result["pre_topic"] = topic_id_list[pre_topic]
                print(post_topic)
                result["post_topic"] = topic_id_list[post_topic]
                result["pre_sentences"] = file_text_to_list(pre_topic_filename)
                result["post_sentences"] = file_text_to_list(post_topic_filename)
                result["label"] = label

                result["pre_topic_start_index"] = 0
                result["pre_topic_end_index"] = get_topic_end_index(topic_id_list[pre_topic])
                result["post_topic_start_index"] = 0
                result["post_topic_end_index"] = get_topic_end_index(topic_id_list[post_topic])

                # pre_topic_equal_flag, pre_sentence_num, pre_topic_start_index, pre_topic_end_index = get_topic_index(topic_id_list[pre_topic], pre_topic_filename)
                # post_topic_equal_flag, post_sentence_num, post_topic_start_index, post_topic_end_index = get_topic_index(topic_id_list[post_topic], post_topic_filename)
                # if pre_topic_equal_flag == True:
                #     pre_start_index = [pre_sentence_num, pre_topic_start_index]
                #     pre_end_index = [pre_sentence_num, pre_topic_end_index]
                # else:
                #     continue
                #     error_topic.write(topic_id_list[pre_topic])
                #     error_topic.write('\n')
                #     pre_start_index = [pre_sentence_num-1, pre_topic_start_index]
                #     pre_end_index = [pre_sentence_num-1, pre_topic_end_index]
                #
                # if post_topic_equal_flag == True:
                #     post_start_index = [post_sentence_num, post_topic_start_index]
                #     post_end_index = [post_sentence_num, post_topic_end_index]
                # else:
                #     continue
                #     error_topic.write(topic_id_list[post_topic])
                #     error_topic.write('\n')
                #     post_start_index = [post_sentence_num-1, post_topic_start_index]
                #     post_end_index = [post_sentence_num-1, post_topic_end_index]
                #
                # result["pre_topic_start_index"] = pre_start_index   #sentences * num_word + index
                # result["pre_topic_end_index"] = pre_end_index
                # result["post_topic_start_index"] = post_start_index
                # result["post_topic_end_index"] = post_end_index

                count += 1

                JSONfile.write(json.dumps(result))
                JSONfile.write("\n")
                # final_result.append(result)

                print("pre topic:", result["pre_topic"], "and post topic；", result["post_topic"],
                      " have write into json file ")


    #JSONfile.write(json.dumps(final_result,ensure_ascii=False))
    JSONfile.close()
    print("label number: ",count)


def getTestJSONLinesdata():

    jsonlines_file_name = "D:\Experiment\CrowdComp_spiderData/PublicKeyCryptography/topic_pair_sentences_test_data.json"
    #jsonlines_file_name = "D:/Experiment/spiderData_v2/topic_pair_sentences_train_data.json"
    #error_topic_file = "D:/Experiment/spiderData_v2/error_topic.json"   #记录无法在文本中精确匹配到topic的topic
    JSONfile = codecs.open(jsonlines_file_name,"w","utf-8")
    #error_topic = codecs.open(error_topic_file,'w','utf-8')
    id_topic_file_0 = "D:/Experiment/CrowdComp_spiderData/Global_warming/id_topic.csv"
    id_topic_file_1 = "D:/Experiment/CrowdComp_spiderData/Meiosis/id_topic.csv"
    id_topic_file_2 = "D:/Experiment/CrowdComp_spiderData/Newton_Laws/id_topic.csv"
    id_topic_file_3 = "D:/Experiment/CrowdComp_spiderData/Parallel_Postulate/id_topic.csv"
    id_topic_file_4 = "D:/Experiment/CrowdComp_spiderData/PublicKeyCryptography/id_topic.csv"
    count = 0

    csv_read_file = "D:\Experiment\CrowdComp_spiderData\PublicKeyCryptography/id_topic.csv"
    csv_pair_file = "D:\Experiment\CrowdComp_spiderData/PublicKeyCryptography/CrowdComp_PublicKeyCryptography_topicPair.csv"
    by_name = "CrowdComp_PublicKeyCryptography"
    file_dir = "D:\Experiment\CrowdComp_spiderData/PublicKeyCryptography/"

    csv = CSVURL()
    topic_id_list = csv.csv_read(csv_read_file)
    prerequisite_pair_result = csv.csv_get_prerequisite_pair(csv_pair_file, by_name)
    # final_result = []

    for pair in prerequisite_pair_result:
        pre_topic = pair["pre_topic"]
        post_topic = pair["post_topic"]
        label = pair["label"]

        pre_topic_filename = file_dir + str(pre_topic) + ".txt"
        post_topic_filename = file_dir + str(post_topic) + ".txt"

        if (file_exist(pre_topic_filename) and file_exist(
                post_topic_filename) and pre_topic in topic_id_list and post_topic in topic_id_list):
            result = {}
            result["pre_topic"] = topic_id_list[pre_topic]
            print(post_topic)
            result["post_topic"] = topic_id_list[post_topic]
            result["pre_sentences"] = file_text_to_list(pre_topic_filename)
            result["post_sentences"] = file_text_to_list(post_topic_filename)
            result["label"] = label

            result["pre_topic_start_index"] = 0
            result["pre_topic_end_index"] = get_topic_end_index(topic_id_list[pre_topic])
            result["post_topic_start_index"] = 0
            result["post_topic_end_index"] = get_topic_end_index(topic_id_list[post_topic])


            count += 1

            JSONfile.write(json.dumps(result))
            JSONfile.write("\n")
            # final_result.append(result)

            print("pre topic:", result["pre_topic"], "and post topic；", result["post_topic"],
                  " have write into json file ")

    # for i in range(4):
    #     if i == 0:
    #         csv_read_file = "D:\Experiment\CrowdComp_spiderData/Meiosis/id_topic.csv"
    #         csv_pair_file = "D:\Experiment\CrowdComp_spiderData/Meiosis/CrowdComp_Meiosis_topicPair.csv"
    #         by_name = "CrowdComp_Meiosis_topicPair"
    #         file_dir = "D:\Experiment\CrowdComp_spiderData/Meiosis/"
    #     if i == 1:
    #         csv_read_file = "D:\Experiment\CrowdComp_spiderData/Newton_Laws/id_topic.csv"
    #         csv_pair_file = "D:\Experiment\CrowdComp_spiderData/Newton_Laws/CrowdComp_Newton_Laws_topicPair.csv"
    #         by_name = "CrowdComp_Newton_Laws_topicPair"
    #         file_dir = "D:\Experiment\CrowdComp_spiderData/Newton_Laws/"
    #     elif i == 2:
    #         csv_read_file = "D:\Experiment\CrowdComp_spiderData/Parallel_Postulate/id_topic.csv"
    #         csv_pair_file = "D:\Experiment\CrowdComp_spiderData/Parallel_Postulate/CrowdComp_Parallel_Postulate_topicPair.csv"
    #         by_name = "CrowdComp_Parallel_Postulate_to"
    #         file_dir = "D:\Experiment\CrowdComp_spiderData/Parallel_Postulate/"
    #     # elif i == 3:
    #         csv_read_file = "D:\Experiment\CrowdComp_spiderData\PublicKeyCryptography/id_topic.csv"
    #         csv_pair_file = "D:\Experiment\CrowdComp_spiderData/PublicKeyCryptography/CrowdComp_PublicKeyCryptography_topicPair.csv"
    #         by_name = "CrowdComp_PublicKeyCryptography"
    #         file_dir = "D:\Experiment\CrowdComp_spiderData/PublicKeyCryptography/"
    #     else:
    #         csv_read_file = "D:\Experiment\CrowdComp_spiderData\Global_warming/id_topic.csv"
    #         csv_pair_file = "D:\Experiment\CrowdComp_spiderData/Global_warming/CrowdComp_Global_warming_topicPair.csv"
    #         by_name = "CrowdComp_Global_warming_topicP"
    #         file_dir = "D:\Experiment\CrowdComp_spiderData/Global_warming/"





    #JSONfile.write(json.dumps(final_result,ensure_ascii=False))
    JSONfile.close()
    print("label number: ",count)


def file_exist(filename):
    if(os.path.exists(filename)):
        return True
    else:
        return False

def file_text_to_list(filename):
    readFile = codecs.open(filename, 'r', 'utf-8')
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    all_sentence = []
    while True:  # 对于每一个段落，即读入的一行字符。进行分句
        line = readFile.readline().strip()
        if not line:
            break
        sentence = sen_tokenizer.tokenize(line)  # 进行分句
        # sentence_list = []
        for s in sentence:
            word_list = nltk.word_tokenize(s)  # 每一句分词，将标点符号与单词分开
            all_sentence.append(word_list)
    readFile.close()
    return all_sentence

def get_topic_index(topic, filename):
    readFile = codecs.open(filename, 'r', 'utf-8')
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    topic_start_index = 0
    topic_end_index = 0

    sentence = 0
    topic_equal_flag = False
    while True:
        line = readFile.readline().strip()
        if not line:
            break
        sentences = sen_tokenizer.tokenize(line)  #进行分句
        for s in sentences:
            if topic in s:
                word_list = nltk.word_tokenize(s)  # 每一句分词，将标点符号与单词分开
                topic_list = nltk.word_tokenize(topic)
                for i in range(len(word_list)):
                    if word_list[i] == topic_list[0]:
                        topic_equal_flag = False
                        for j in range(len(topic_list)):
                            if topic_list[j] == word_list[i+j]:
                                topic_equal_flag = True
                            else:
                                topic_equal_flag = False
                                break
                        if topic_equal_flag == True:
                            topic_start_index = i
                            topic_end_index = i+len(topic_list)-1
                            readFile.close()
                            return topic_equal_flag, sentence, topic_start_index, topic_end_index

            sentence += 1
    return topic_equal_flag, sentence, topic_start_index, topic_end_index

def get_topic_end_index(topic):
    topic_list = nltk.word_tokenize(topic)
    return len(topic_list)-1

getJSONLinesdata()
# getTestJSONLinesdata()





