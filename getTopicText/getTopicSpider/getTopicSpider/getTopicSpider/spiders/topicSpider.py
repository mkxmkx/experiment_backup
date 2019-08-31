import scrapy
from .getURLFromCSV import CSVURL
from getTopicSpider.items import GettopicspiderItem

class topicSpider(scrapy.Spider):
    name = 'get_topic_from_url'
    allowed_domains = ['wiki']
    excelurl = CSVURL()
    dict_list = excelurl.csv_read()
    start_urls = []
    for key in dict_list:
        id = key
        url = dict_list[key]
        if (url != "NULL"):
            url = url + "?id=" + str(id)
            start_urls.append(url)

    def parse(self, response):
        items = GettopicspiderItem()
        id = str(response.url).strip().split('?id=')[-1]
        items["id"] = id
        items["url"] = str(response.url).strip().split('?id=')[0]
        topic = response.xpath("//*[@id='firstHeading']/text()").extract()[0].strip()
        items['topic'] = topic
        return items
