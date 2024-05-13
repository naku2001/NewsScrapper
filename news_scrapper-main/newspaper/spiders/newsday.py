import scrapy
from scrapy_splash import SplashRequest
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError
from twisted.internet.error import TimeoutError, TCPTimedOutError

class NewsDaySpider(scrapy.Spider):
    name = 'newsday_spider'
    start_urls = [
        'https://www.newsday.co.zw/category/4/business',
        'https://www.newsday.co.zw/category/9/opinion-and-analysis',
        'https://www.newsday.co.zw/category/5/sport',
        'https://www.newsday.co.zw/category/8/lifestyle-and-arts'
        
    ]

    def parse(self, response):
        # Extract data from each article
            category = response.css('div.brand-title h2 a.links::text').get()
            title = response.css('a.text-dark div.sub-title.mt-3::text').extract()
            link = response.css('div.card-body.pad-o.mt-3 a::attr(href)').extract()
            content = response.css('div.card-body.pad-o.mt-3 div.mb-3.pt-2.top-article::text').extract()

            row_data = zip(title,link,content)
            
        #Making extracted data row wise
            for item in row_data:
                #create a dictionary to store the scraped info
                scraped_info = {
                    #key:value
                    'category':category,
                    'title' : item[0],
                    'link' : item[1],
                    'content' : item[2]
                }

                #yield or give the scraped info to scrapy
                yield scraped_info

                
class Fingaz(scrapy.Spider):
    name = 'fingaz'
    start_urls = [
        'https://fingaz.co.zw/category/c77-companies-a-markets',
        'https://fingaz.co.zw/category/lifestyle',
        'https://fingaz.co.zw/category/sport',
        'https://fingaz.co.zw/category/tech/'
    ]

    def parse(self, response):
        # Extract data from each article
        category = response.css('div.category-title h2::text').get()
        title_links = response.css('h3.entry-title a')
        content = response.css('div.post-excerpt p::text').extract()

        # Loop through each title-link pair and corresponding content
        for title_link, content in zip(title_links, content):
            # Extract title and link
            title = title_link.css('::text').get()
            link = title_link.css('::attr(href)').get()

            # Create a dictionary to store the scraped info
            scraped_info = {
                'category': category,
                'title': title,
                'link': link,
                'content': content.strip()  # Remove leading/trailing whitespaces
            }

            # Yield the scraped info to Scrapy
            yield scraped_info
            


class StandardSpider(scrapy.Spider):
    name = 'standard'
    start_urls = [
        'https://www.newsday.co.zw/thestandard/category/18/business',
        'https://www.newsday.co.zw/thestandard/category/23/opinion-and-analysis',
        'https://www.newsday.co.zw/thestandard/category/19/sports',
        'https://www.newsday.co.zw/thestandard/category/25/standard-people'
    ]

    def parse(self, response):
        # Extract category from the updated structure
        category = response.css('div.brand-title h2 a.links::text').get() 

        for article in response.css('div.card-body'):
            title = article.css('h3.mb-3 ::text').get()
            link = article.css('a.text-dark::attr(href)').get()
            content = article.css('div.mb-3.pt-2.top-article ::text').get() 

            scraped_info = {
                'category': category,
                'title': title,
                'link': link,
                'content': content
            }
            yield scraped_info
class ZimMail(scrapy.Spider):
    name = 'zimmail'
    start_urls = [
        'https://www.thezimbabwemail.com/category/sports/',
        'https://www.thezimbabwemail.com/category/business/',
        'https://www.thezimbabwemail.com/category/entertainment/',
        'https://www.thezimbabwemail.com/category/politics/'
    ]
    def parse(self, response):
        # Extract data from each article
            category = response.css('h1.page-title::text').get() 
            title = response.css('.entry-title a::text').extract()
            link = response.css('.entry-title a::attr(href)').extract()
            content = response.css('.mh-excerpt p::text').extract() 
            row_data = zip(title,link,content)
            
        #Making extracted data row wise
            for item in row_data:
                #create a dictionary to store the scraped info
                scraped_info = {
                    #key:value
                    'category':category,
                    'title' : item[0],
                    'link' : item[1],
                    'content' : item[2]
                }

                #yield or give the scraped info to scrapy
                yield scraped_info


    # def parse(self, response):
    #     # Extract category from the updated structure
    #     category = response.css('div.brand-title h2 a.links::text').get() 

    #     for article in response.css('.mh-loop-item'):
    #         title = article.css('.entry-title a::text').get()
    #         link = article.css('.entry-title a::attr(href)').get()
    #         content = article.css('.mh-excerpt p::text').get() 

    #         scraped_info = {
    #             'category': category,
    #             'title': title,
    #             'link': link,
    #             'content': content
    #         }
    #         yield scraped_info

