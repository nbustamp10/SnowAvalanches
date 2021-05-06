import scrapy
from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector
from craigslist_sample.item import CraigslistSampleItem

class MySpider(BaseSpider):
name ="craig"
allowed_domains=["https://climatologia.meteochile.gob.cl/application/informacion/listadoDeComponentesDeUnElemento/330050/60"]
start_urls=["https://climatologia.meteochile.gob.cl/application/informacion/inventarioComponentesPorEstacion/330050/60/125"]

def parse(self, response):
hxs=HtmlXPathSelector(response)
titles=hxs.select("//p")
items=[]

for titles in titles:
    item=CraigslistSampleItem()
item["title"]=titles.select("a/text()").extract()
item["link"] =titles.select("a/@href").extract()
item.apped(item)
return items
