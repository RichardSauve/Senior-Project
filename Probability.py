import numpy as np
import pandas as pd
import openpyxl
import warnings
import datetime as dt
from datetime import date, timedelta
import cloudscraper as cs
from bs4 import BeautifulSoup as bs

warnings.filterwarnings('ignore')

################

URL = 'https://www.hockey-reference.com/leagues/NHL_2023_games.html'

args = {'browser': 'chrome', 'desktop': True, 'platform': 'windows'}

scraper = cs.create_scraper(browser=args)
response = scraper.get(URL)

soup = bs(response.text, 'html.parser')
tex = soup.find('table').find_all('tr')

##################
for i in range(1):
    start = pd.to_datetime('2022-11-03', format = '%Y-%m-%d') + timedelta(days=i)
##################

    today_date = start.strftime('%Y-%m-%d')
    print(today_date)
    yest_games = []
    for _, t in enumerate(tex):
        if str(t.find('th').text) == str(today_date):
            yest_games.append((t.find_all('td')[0].text, t.find_all('td')[1].text, t.find_all('td')[2].text,
                               t.find_all('td')[3].text))

    # print('Yesterday\'s Games done!')
    ######################

    data = pd.DataFrame(columns=['Team', 'Elo'])
    wb = openpyxl.load_workbook('Elo_test.xlsx')
    ws = wb.active

    for j in range(2, 34):
        data = data.append({'Team': ws[f'A{j}'].value, 'Elo': ws[f'B{j}'].value}, ignore_index=True)

    data = data.set_index('Team', drop=True).to_dict()['Elo']
    print(data['Carolina Hurricanes'])
    #print('Elo Dict done!')

    #######################
    K = 6.
    for game in yest_games:
        elo_diff = -(data[game[2]] + 50 - data[game[0]])/400
        AC = 2.05 / (abs(elo_diff) * 0.001 + 2.05)
        prob_a = 1. / (np.power(10, elo_diff) + 1)
        prob_b = 1. - prob_a
        MoV = int(game[3]) - int(game[1]) * AC
        MVM = 0.06686 * np.log(abs(MoV)) + 0.8048
        print(game[2], prob_a, ':', game[0], prob_b)
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

    #print('Elo Shift Done!')

    #######################


    output = pd.DataFrame(data.items(), columns=['Team', 'Elo']).sort_values('Elo', ascending=False)
    for z in range(2, 34):
        ws[f'A{z}'] = output.iloc[z-2, 0]
        ws[f'B{z}'] = output.iloc[z-2, 1]

    #print('Updated Excel File Done!')

    #######################

    wb.save('Elo_test.xlsx')
    wb.close()

    print('All Done!')
###################

