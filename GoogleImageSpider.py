'''DISCLAIMER: DUE TO COPYRIGHT ISSUES, IMAGES GATHERED SHOULD
   ONLY BE USED FOR RESEARCH AND EDUCATION PURPOSES ONLY'''
from bs4 import BeautifulSoup
import certifi, urllib3
from urllib3.util.url import parse_url
import imghdr
import re
import sys, os
import json

class GoogleImageSpider:
    debug = False
    actual_images=[]    # Contains the link for Large original images, type of  image
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
    header = {         # Headers emulation of a usual browser (My use case: Chrome in linux)
#        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9,es-ES;q=0.8,es;q=0.7,ca-ES;q=0.6,ca;q=0.5',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
        'x-chrome-uma-enabled': '1',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'cache-control': 'max-age=0',
        'authority': 'www.google.es',
        'cookie': 'CONSENT=WP.26915a; 1P_JAR=2018-01-21-15; NID=122=iyOn0m6Eogu0T1dNztHWOm2ckyjz6TqeB7aHEav-1YhcyClxqPWDIXn764n5y_asAmLWjS9mKPZkbt_kvRj_TarUFGaSLXdAUElpQo9FRoheTxlu7DFwLwLI3mjiaaGp',
        'x-client-data': 'CI22yQEIorbJAQitmMoBCPqcygEIqZ3KAQioo8oB'
    }
    
    def __init__(self, debug = False):
        self.actual_images=[]
        self.http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
        self.debug = debug

    def get_html(self, keyword):
        url = "https://www.google.es/search?q="+keyword+"&source=lnms&tbm=isch"
        print ("Looking for: ", keyword)
        request = self.http.request('GET', url, headers=self.header)
        page = request.data.decode('utf-8')
        request.release_conn()
        return BeautifulSoup(page, 'html.parser')

    def get_more_html(self, keyword, offset):
        url = 'https://www.google.es/search?ei=bqtkWtnIBIORUZPosNgF&yv=2&q='+keyword+'\
&tbm=isch&vet=10ahUKEwjZt8HGqenYAhWDSBQKHRM0DFsQuT0I6AEoAQ.bqtkWtnIBIORUZPosNgF.i&ved=0\
ahUKEwjZt8HGqenYAhWDSBQKHRM0DFsQuT0I6AEoAQ&ijn=2&start='+str(offset)+'&asearch=ichunk&async=_id:rg_s,_pms:s'
        print ("Looking for", keyword, "at offset", offset)
        request = self.http.request('GET', url, headers=self.header)
        page = request.data.decode('unicode-escape').replace("\\/", "/")
        request.release_conn()
        return BeautifulSoup(page, 'html.parser')

    def get_images_from_html(self, html, limit=1000):
        size = len(self.actual_images)
        for a in html.find_all("div",{"class":"rg_meta"}):
            link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
            self.actual_images.append((link,Type))
            limit -= 1
            if (limit == 0):
                break
        if self.debug:
            print ("Images added: ", len(self.actual_images)-size)
            
    def save_images(self, prefix, path):
        if not os.path.exists(path):
            os.mkdir(path)
        
        chunk_size = 1024
        for i in range(len(self.actual_images)):
            try:
                url_image = parse_url(self.actual_images[i][0])
                url_image = url_image.url
#                url_image = self.actual_images[i][0].split('?')[0]
#                print (i, url_image, self.actual_images[i][1])
                r = self.http.request('GET', url_image, preload_content=False, timeout=5.0)
                filename = "./"+path+"/"+str(prefix)+str(i)
                if self.actual_images[i][1] != "":
                    filename += '.'+self.actual_images[i][1]
                with open(filename, 'wb') as out:
                    while True:
                        data = r.read(chunk_size)
                        if not data:
                            break
                        out.write(data)
                r.release_conn()
                
                if self.actual_images[i][1] == "":
#                    print ("Detecting type:", imghdr.what(filename))
                    os.rename(filename, filename+"."+imghdr.what(filename))
            except:
                print ("Something goes wrong trying to save image: ", sys.exc_info()[0])
                
        print (len(self.actual_images), "images saved to: ", path)
            
    def get_images(self, keyword, how_many):
        parsed_keyword = keyword.split()
        parsed_keyword = '+'.join(parsed_keyword)
        
        html = self.get_html(parsed_keyword)
        self.get_images_from_html(html, limit=how_many)
        
        if how_many > 100:
            for i in range(0, how_many, 100):
                html = self.get_more_html(parsed_keyword, i)
                self.get_images_from_html(html, limit=how_many-i)
        
    def clear(self):
        self.actual_images=[]
