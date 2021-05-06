from bs4 import BeautifulSoup
import requests
import re




def solvehref(href):
    tmps = href.split('"')
    tmp = tmps[1]
    tmp = tmp.replace('http://','')
    tmp = tmp.replace('https://','')
    tmps = tmp.split('/')
    tmp = tmps[0]
    return tmp


def return_all_url(url):
    allatags = []
    try:
        r = requests.get(url)
        r.encoding = r.apparent_encoding
        # r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, 'html.parser')
        pattern = re.compile("href\s{0,3}=\s{0,3}\"[^\"]+\"")
        hrefs=re.findall(pattern,soup.prettify())
        for href in hrefs:
            tmpurl = solvehref(href)
            if tmpurl not in allatags:
                allatags.append(tmpurl)
    except Exception as e:
        print(e)
        pass
    return allatags


urls = return_all_url("http://sina.com")
print(urls)


                
