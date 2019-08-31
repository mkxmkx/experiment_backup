import xlrd
import csv
import codecs
import random

'''
读csv文件，并返回主题id和主题的字典
'''

class CSVURL():


    def __init__(self):
        #self.filename = "D:/Experiment/spiderData/prerequisite_topics_v2.csv"
        self.filename = "D:/Experiment/spiderData_v4/Global_warming/id_topic.csv"
        self.colnameindex = 0  # 表头列名所在行的索引
        self.by_name = 'id_topic'  #sheet名称
    def csv_read(self, filename):
        result = {}
        csvFile = open(filename,'r')
        reader = csv.reader(csvFile)
        for item in reader:
            '''
            忽略第一行
            if (reader.line_num ==1):
                continue
            '''
            if(item[0] == 199):# 199号主题重复，去掉
                continue
            print("主题id：",item[0],". topic：",item[1])
            result[item[0]] = item[1]
        csvFile.close()
        return result

    def csv_get_prerequisite_pair(self, filename, by_name):
        # self.filename = "D:/Experiment/spiderData_v2/prerequisite_annotations.csv"
        #self.filename = "D:\Experiment\spiderData_v4\Global_warming/CrowdComp_Global_warming_topicPair.csv"
        self.colnameindex = 0
        #self.by_name = 'CrowdComp_Global_warming_topicP'
        self.by_name = by_name
        result = []

        trueLabelCount = 0
        falseLabelCount = 0

        csvFile = open(filename, 'r')
        reader = csv.reader(csvFile)
        for item in reader:

            #忽略第一行
            if (reader.line_num ==1):
                continue
            # if(item[0] == 199 or item[1] == 199):
            #     continue
            print("source topic id: ", item[0], ". target topic id ：", item[1], ". label : ", item[2])
            temp = {}
            temp["pre_topic"] = item[0]
            temp["post_topic"] = item[1]
            temp["label"] = item[2]
            # result.append(temp)
            if temp["label"] == "1":
                result.append(temp)
                trueLabelCount += 1
            else:
                x = random.random()
                if x > 0.95:
                    result.append(temp)
                    falseLabelCount += 1
                else:
                    continue
        csvFile.close()
        print("true label {}, false label {}".format(trueLabelCount, falseLabelCount))
        return result

# filename = "D:/Experiment/spiderData_v2/prerequisite_annotations.csv"
# by_name = "prerequisite_annotations"
# csv1 = CSVURL()
# result = csv1.csv_get_prerequisite_pair(filename, by_name)

