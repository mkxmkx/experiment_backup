# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import codecs
import csv


class WikispiderPipeline(object):
    def process_item(self, item, spider):
        base_dir = "D:/Experiment/CrowdComp_spiderData/PublicKeyCryptography"
        topic = item['topic']
        id = item['id']
        filename = base_dir + '/' + str(id) + '.txt'

        f = codecs.open(filename,'w','utf-8')
        f.write(topic)
        f.write('.\n')
        for text in item['text']:
            f.write(text)

        topicfile = base_dir + "/" + "id_topic.csv"
        #topicfile = "D:/Experiment/spiderData_v4/Global_warming/id_topic.csv"
        with open(topicfile, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([id, topic])

        return item
