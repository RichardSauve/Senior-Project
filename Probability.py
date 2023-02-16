import numpy as np
import pandas as pd
import openpyxl
import warnings
from datetime import date, timedelta
import cloudscraper as cs
from bs4 import BeautifulSoup as bs

warnings.filterwarnings('ignore')

##################

today_date = date.today()
yest_date = today_date - timedelta(days=1)
print(yest_date)

URL = 'https://www.hockey-reference.com/leagues/NHL_2023_games.html'


args = {'browser': 'chrome', 'desktop': True, 'platform': 'windows'}

scraper = cs.create_scraper(browser=args)
response = scraper.get(URL)

soup = bs(response.text, 'html.parser')
tex = soup.find('table').find_all('tr')

yest_games = []
for i, t in enumerate(tex):
    if str(t.find('th').text) == str(yest_date):
        yest_games.append((t.find_all('td')[0].text, t.find_all('td')[1].text, t.find_all('td')[2].text,
                            t.find_all('td')[3].text))

print('Yesterday\'s Games done!')

######################

data = pd.DataFrame(columns=['Team', 'Elo'])
wb = openpyxl.load_workbook('Elo_test.xlsx')
ws = wb.active

for i in range(2, 34):
    data = data.append({'Team': ws[f'A{i}'].value, 'Elo': ws[f'B{i}'].value}, ignore_index=True)

data = data.set_index('Team', drop=True).to_dict()['Elo']

print('Elo Dict done!')

#######################
K = 6.
for game in yest_games:
    elo_diff = -(data[game[2]] + 50 - data[game[0]])/400
    prob_a = np.round(1. / (np.power(10, elo_diff) + 1), 4)
    prob_b = np.round(1. - prob_a, 4)
    MoV = int(game[3]) - int(game[1])
    MVM = 0.06686 * np.log(abs(MoV)) + 0.8048
    if MoV > 0:
        TWa = 1.
        TWb = 0.
    else:
        TWa = 0.
        TWb = 1.

    shifta = K * MVM * (TWa - prob_a)
    shiftb = K * MVM * (TWb - prob_b)
    A = float(data[game[2]])
    B = float(data[game[0]])
    data[game[2]] = A + shifta
    data[game[0]] = B + shiftb


print('Elo Shift Done!')

#######################


output = pd.DataFrame(data.items(), columns=['Team', 'Elo']).sort_values('Elo', ascending=False)
for i in range(2, 34):
    ws[f'A{i}'] = output.iloc[i-2, 0]
    ws[f'B{i}'] = output.iloc[i-2, 1]

print('Updated Excel File Done!')

#######################

wb.save('Elo_test.xlsx')
wb.close()

print('All Done!')
###################

