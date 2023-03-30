import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import openpyxl
import cloudscraper as cs

from datetime import timedelta
from bs4 import BeautifulSoup as bs
from sklearn.metrics import roc_curve, auc
from Probability import probability

k = np.arange(5.5, 6.5, 0.01)
hta = np.round(np.arange(25., 75., 0.1), 2)
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

yest_games = []
for i in range(175):
    start = pd.to_datetime('2022-10-06', format='%Y-%m-%d') + timedelta(days=i)
    ##################
    today_date = start.strftime('%Y-%m-%d')
    for _, t in enumerate(tex):
        if str(t.find('th').text) == str(today_date):
            yest_games.append((t.find_all('td')[0].text, t.find_all('td')[1].text, t.find_all('td')[2].
                               text, t.find_all('td')[3].text))

a = 6
# b = 50
c = 0.6686
d = 0.8048
e = 2.05
f = 0.001

def my_func():
    sheet_num = 1
    scores = []

    for b in hta:
        probability(K=a, HTA=b, MVM_A=c, MVM_B=d, ACF=e, ACM=f, wb_name=wb_name, sheet_num=sheet_num,
                    yest_games=yest_games)

        data = pd.read_excel('Main.xlsx', sheet_name=f'sheet{sheet_num}', header=[2])

        sheet_num += 1

        fpr, tpr, thresholds = roc_curve(data['home_win'], data['home_prob'])
        roc_auc = auc(fpr, tpr)
        scores.append(roc_auc)
        print(f'{b} Done!')

    print(max(scores), scores.index(max(scores)) + 1)

    plt.plot(hta, scores, marker = 'o', color = 'red')
    plt.xlabel('HTA')
    plt.ylabel('AUC score')
    plt.title('HTA  -  AUC Scores')
    plt.savefig('HTA_AUC')
    plt.show()
    return

my_func()
