import csv
import pandas as ps
import requests
from bs4 import BeautifulSoup
import json
str_url = "https://guba.eastmoney.com/news,of015909,1515874124.html"
str_url = str(str_url).replace(".html","_1.html")
print(str_url)