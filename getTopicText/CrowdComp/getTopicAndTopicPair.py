import csv

class CrowdCompCSVread():
    def __init__(self):
        self.filename = "D:\Experiment\spiderData_v4\CrowdComp/CrowdComp_PublicKeyCryptography_Prerequisite_HIT.csv"
        self.colnameindex = 0  # 表头列名所在行的索引
        self.by_name = 'CrowdComp_PublicKeyCryptography'  #sheet名称

    def csv_read(self):
        result = {}
        csvFile = open(self.filename,'r')
        reader = csv.reader(csvFile)
        sourceList = []
        targetList = []
        label = []
        for item in reader:
            #忽略第一行
            if (reader.line_num ==1):
                continue
            sourceList.append(item[27])
            targetList.append(item[29])
            # 0代表DoNotKnow， 1代表Source2Target， 2代表Target2Source， 3代表Unrelated
            if item[31]:
                label.append(0)
            if item[32]:
                label.append(1)
            elif item[33]:
                label.append(2)
            else:
                label.append(3)
        result["sourceURL"] = sourceList
        result["targetURL"] = targetList
        result["label"] = label
        csvFile.close()
        return result

csvReader = CrowdCompCSVread()
csvResult = csvReader.csv_read()
sourceURL = csvResult["sourceURL"]
targetURL = csvResult["targetURL"]
label = csvResult["label"]

topicSet = set()
source = []
target = []
label_result = []

for i in range(0, len(sourceURL), 3):
    topicSet.add(sourceURL[i])
    topicSet.add(targetURL[i])
    source.append(sourceURL[i])
    target.append(targetURL[i])
    labelTag = [0, 0, 0]
    for j in range(3):
        # if label[i+j] == 0:
        #     labelTag[0] += 1
        if label[i+j] == 1:
            labelTag[1] += 1
        if label[i+j] == 2:
            labelTag[2] += 1
        if label[i+j] == 3:
            labelTag[0] += 1
    label_result.append(labelTag.index(max(labelTag)))   # 0代表无， 1代表Source2Target， 2代表Target2Source

topicDICT = {}
topicResult = []
for i, element in enumerate(topicSet):
    topicDICT[element] = i
    topicTemp = [i, element]
    topicResult.append(topicTemp)

topicFile = "D:\Experiment\CrowdComp_spiderData/PublicKeyCryptography/CrowdComp_PublicKeyCryptography_topic2.csv"
with open(topicFile, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(topicResult)

csvfile.close()

topicPair = "D:\Experiment\CrowdComp_spiderData/PublicKeyCryptography/CrowdComp_PublicKeyCryptography_topicPair2.csv"
with open(topicPair, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["source", "target", "prerequisite"])
    topicPairResult = []
    for i in range(len(source)):
        sourceConcept = source[i]
        sourceID = topicDICT[sourceConcept]
        targetConcept = target[i]
        targetID = topicDICT[targetConcept]
        labelID = label_result[i]

        if labelID == 0:
            topicPairResult.append([sourceID, targetID, 0])
        elif labelID == 1:
            topicPairResult.append([sourceID, targetID, 1])
        else:
            topicPairResult.append([sourceID, targetID, 0])
            # topicPairResult.append([targetID, sourceID, 1])
    writer.writerows(topicPairResult)
csvfile.close()



