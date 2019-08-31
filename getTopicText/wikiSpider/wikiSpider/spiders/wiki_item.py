# -*- coding: utf-8 -*-
import scrapy

from wikiSpider.items import WikispiderItem
from .getURLFromCSV import excelURL

class WikiItemSpider(scrapy.Spider):
    name = 'wiki_item'
    allowed_domains = ['wiki']
    excelurl = excelURL()
    dict_list = excelurl.csv_read()

    start_urls = []

    for id in dict_list:
        url = dict_list[id]
        if(url != 'NULL'):
            url = url + "?id=" + str(id)
            print("url 字典内容", url)
            start_urls.append(url)

    def parse(self, response):
        topic = response.xpath("//*[@id='firstHeading']/text()").extract()[0].strip()

        url_list = []
        #print("当前url:", response.request.url)
        url_list.append(response.request.url)
        id = str(response.request.url).strip().split("id=")[-1]

        '''
        将文本中链接对应的页面也爬下来
        textlist = response.xpath('//*[@id="mw-content-text"]/div[@class="mw-parser-output"]/*')
        flag = False
        for li in textlist:
            pTag = li.xpath("self::p")
            #print("ptag内容", pTag)
            if (pTag):
                #print("集合不为空，p标签: ", li)
                if (li.css(".mw-empty-elt")):
                    continue
                flag = True
                tempURL = li.xpath('./a/@href').extract()
                print("temp url 内容:" ,tempURL)
                for url in tempURL:
                    url = "https://en.wikipedia.org" + url
                    url_list.append(url)
            if ((not pTag) and flag == True):
                break

        print("url list size :",len(url_list))
        '''

        for item_url in url_list:
            yield scrapy.Request(url=item_url,meta={'topic':topic, 'id':id},callback = self.topic_page,dont_filter=True)

    def topic_page(self,response):

        print("in topic_page函数")
        topic = response.meta['topic']
        id = response.meta['id']

        items = WikispiderItem()
        items['topic'] = topic
        items['id'] = id

        text = []
        textlist = response.xpath('//*[@id="mw-content-text"]/div[@class="mw-parser-output"]/*')
        #print("textlist内容:", textlist.extract())
        flag = False
        for li in textlist:
            pTag = li.xpath("self::p")
            #print("ptag内容", pTag)
            if (pTag):
                #print("集合不为空，p标签: ", li)
                if (li.css(".mw-empty-elt")):
                    continue
                flag = True
                # temp = li.xpath('.//text()').extract()
                temp = li.xpath('string(.)').extract()
                final_text = ""
                for word in temp:
                    if word.strip():
                        final_text = final_text + " " + word.strip()
                #print("temp text内容: ", final_text)
                if (final_text.strip()):
                    text.append(final_text.strip())
                    text.append('\n')
            if ((not pTag) and flag == True):
                break
        #items['text'] = text
        #print("text内容：", text)
        items["text"] = text

        return items





