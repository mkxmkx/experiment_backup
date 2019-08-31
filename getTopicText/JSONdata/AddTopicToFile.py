import codecs
import os

from JSONdata.getTopicFromCSV import CSVURL

def addTopic():
    csv = CSVURL()
    topic_id_list = csv.csv_read()

    for id in topic_id_list:
        filename = "D:/Experiment/spiderData/" + str(id) + ".txt"
        write_filename = "D:/Experiment/spiderData/" + str(id) + "withtopic.txt"
        if (os.path.exists(filename)):
            print("filename : ", filename, " exist")
            readFile = codecs.open(filename, 'r', 'utf-8')
            writeFile = codecs.open(write_filename, 'w', 'utf-8')
            topic = topic_id_list[id]
            writeFile.write(topic)
            writeFile.write('.\n')
            while True:
                line = readFile.readline().strip()
                if not line:
                    break
                writeFile.write(line)
                writeFile.write('\n')
            readFile.close()
            writeFile.close()

addTopic()
