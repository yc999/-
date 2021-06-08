#-- coding: utf-8 --

# 自动访问网站

import  requests
from bs4 import  BeautifulSoup, Comment
import random
import re
import time
import traceback
from selenium.webdriver.firefox.options import Options
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import UnexpectedAlertPresentException



option = Options()
option.add_argument('--no-sandbox')
option.add_argument('log-level=3')
# option.add_argument('--disable-dev-shm-usage')
# option.add_argument('--headless') #静默运行
# option.add_argument('--disable-gpu')  # 禁用GPU加速,GPU加速可能会导致Chrome出现黑屏，且CPU占用率高达80%以上
time_limit = 60
browser = webdriver.Firefox(options=option)
# browser = webdriver.Chrome(options=option)
browser.implicitly_wait(time_limit)
browser.set_page_load_timeout(time_limit)


filepath = "D:\dnswork\sharevm\dnsdata\myclass/音乐网站.txt"
f = open(filepath,"r",encoding="utf-8")
urlList = f.readlines()

for url in urlList:
        print(url)
        try:
            url = "".join(url.split())
            url = url.split(",")[1]
        except:
            continue

        tmpurl = url.replace('www.','',1)
        httpurl =  'http://' + url
        try:
            browser.get(httpurl)
            WebDriverWait(browser, time_limit, 1).until_not(EC.title_is(""))
        except:
            pass
        get_inputstr = input()
        if get_inputstr == "0":
            if url.split(".")[0]!="www":
                httpurl = 'http://www.' + url
            else:
                httpurl = 'http://' + url.replace('www.','',1)
            try:
                browser.get(httpurl)
                WebDriverWait(browser, time_limit, 1).until_not(EC.title_is(""))
            except:
                print(url,"  wrong")
                pass
            get_inputstr = input()

