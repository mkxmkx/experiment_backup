# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import csv


class GettopicspiderPipeline(object):
    def process_item(self, item, spider):
        filename = "D:/Experiment/spiderData/prerequisite_topics_v2.csv"
        csvfile = open(filename,'a')
        writer = csv.writer(csvfile)
        id = item['id']
        topic = item['topic']
        url = item['url']
        temp = [id, topic, url]
        writer.writerow(temp)
        csvfile.close()
        return item
