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
hta = np.round(np.arange(-500., 500., 10), 2)
mvma = np.round(np.arange(0.1, 1.1, 0.01), 2)
mvmb = np.round(np.arange(0.5, 1.1, 0.01), 2)
acf = np.arange(2.02, 2.09, 0.01)
acm = np.round(np.arange(0.0007, 0.0013, 0.0001), 4)

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
b = 50
c = 0.6686
d = 0.8048
e = 2.05
# f = 0.001


def my_func():
    sheet_num = 1
    scores = []

    data = pd.DataFrame(columns=['Team', 'Elo'])
    wb = openpyxl.load_workbook('Elo_test.xlsx')
    ws = wb.active

    for j in range(2, 34):
        data = data.append({'Team': ws[f'A{j}'].value, 'Elo': ws[f'B{j}'].value}, ignore_index=True)

    elo = data.set_index('Team', drop=True).to_dict()['Elo']

    wb.close()

    for f in acm:
        probability(K=a, HTA=b, MVM_A=c, MVM_B=d, ACF=e, ACM=f, wb_name=wb_name, sheet_num=sheet_num,
                    yest_games=yest_games, elo_sub=elo)

        data = pd.read_excel('Main.xlsx', sheet_name=f'sheet{sheet_num}', header=[2])

        sheet_num += 1

        fpr, tpr, thresholds = roc_curve(data['home_win'], data['home_prob'])
        roc_auc = auc(fpr, tpr)
        scores.append(roc_auc)
        print(f'{f} Done!')

    print(max(scores), scores.index(max(scores)) + 1)

    plt.plot(acm, scores, marker = 'o', color = 'red')
    plt.xlabel('ACM')
    plt.ylabel('AUC score')
    plt.title('ACM  -  AUC Scores')
    plt.savefig('ACM_AUC')
    plt.show()
    return

my_func()
