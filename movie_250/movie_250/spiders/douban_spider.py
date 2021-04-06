from scrapy.spiders import Spider
import sys
# print(sys.path)
import os
sys.path.append(os.path.realpath('./movie_250'))
sys.path.append(os.path.realpath('./../movie_250'))
sys.path.append(os.path.realpath('./../../movie_250'))
from scrapy import Request
from movie_250.items import DoubanMovieItem


class DoubanMovieTop250Spider(Spider):
    name = 'movie_250'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36',
    }
    def start_requests(self):
        url = 'https://movie.douban.com/top250'
        yield Request(url, headers=self.headers)
    # start_urls = ['https://movie.douban.com/top250']
    
    def parse(self, response):
        item = DoubanMovieItem()
        movies = response.xpath('//ol[@class="grid_view"]/li')
        for movie in movies:
            item['ranking'] = movie.xpath(
                './/div[@class="pic"]/em/text()').extract()[0]
            item['movie_name'] = movie.xpath(
                './/div[@class="hd"]/a/span[1]/text()').extract()[0]
            item['score'] = movie.xpath(
                './/div[@class="star"]/span[@class="rating_num"]/text()'
            ).extract()[0]
            item['score_num'] = movie.xpath(
                './/div[@class="star"]/span/text()').re(u'(\d+)人评价')[0]
            yield item