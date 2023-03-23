import pandas as pd
import numpy as np
import openpyxl
import cloudscraper as cs

from time import time
from bs4 import BeautifulSoup as bs
from sklearn.metrics import roc_curve, auc
from Probability import probability

k = [5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3]
hta = [35, 40, 44, 50, 55, 60, 65]
mvma = np.arange(0.6086, 0.7386, 0.02)
mvmb = np.arange(0.7448, 0.8748, 0.02)
acf = np.arange(2.02, 2.09, 0.01)
acm = np.arange(0.0007, 0.0013, 0.0001)

URL = 'https://www.hockey-reference.com/leagues/NHL_2023_games.html'

args = {'browser': 'firefox', 'desktop': True, 'platform': 'windows'}

scraper = cs.create_scraper(browser=args)
response = scraper.get(URL)

soup = bs(response.text, 'html.parser')
tex = soup.find('table').find_all('tr')
wb_name = 'Main.xlsx'

wb = openpyxl.Workbook()
wb.save(wb_name)
wb.close()


def my_func():
    sheet_num = 1
    scores = []
    for a in k:
        for b in hta:
            for c in mvma:
                for d in mvmb:
                    for e in acf:
                        for f in acm:

                            probability(K=a, HTA=b, MVM_A=c, MVM_B=d, ACF=e, ACM=f, wb_name=wb_name, sheet_num=sheet_num,
                                        tex=tex)

                            data = pd.read_excel('Main.xlsx', sheet_name=f'sheet{sheet_num}', header=[2])

                            sheet_num += 1

                            fpr, tpr, thresholds = roc_curve(data['home_win'], data['home_prob'])
                            roc_auc = auc(fpr, tpr)
                            scores.append(roc_auc)

    print(max(scores), scores.index(max(scores)) + 1)
    return

my_func()
