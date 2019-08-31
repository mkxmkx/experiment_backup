import xlrd
import csv

'''
读csv文件，并返回主题id和url的字典
'''

class CSVURL():


    def __init__(self):
        #self.filename = "D:/Experiment/Neural Relation Extraction with Selective Attention over Instances/original/TutorialBank/data/prerequisite_topics.csv"
        self.filename = "D:\Experiment\CrowdComp_spiderData\Global_warming/CrowdComp_Global_warming_topic.csv"
        self.colnameindex = 0  # 表头列名所在行的索引
        self.by_name = 'CrowdComp_Global_warming_topic'  #sheet名称
    def csv_read(self):
        result = {}
        csvFile = open(self.filename,'r')
        reader = csv.reader(csvFile)
        for item in reader:
            '''
            忽略第一行
            if (reader.line_num ==1):
                continue
            '''
            print("主题id：",item[0],". 链接：",item[1])
            result[item[0]] = item[1]
        return result
    def my_open_excel(self):
        try:
            data = xlrd.open_workbook(self.filename)
            return data
        except Exception as e:
            print(str(e))

    def excel_table_byname(self):
        data = self.my_open_excel()
        table = data.sheet_by_name(self.by_name)
        nrows = table.nrows  # 行数
        colnames = table.row_values(self.colnameindex)  # 某一行数据
        list = []
        for rownum in range(1, nrows):
            row = table.row_values(rownum)
            # print(row)
            list.append(row)
        return list

    def get_url(self):
        list = self.excel_table_byname()
        dict = {}
        for topic in list:
            dict[topic[1]] = topic[2]
        return dict

