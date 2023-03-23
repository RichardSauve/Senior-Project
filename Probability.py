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
def probability(K, HTA, MVM_A, MVM_B, ACF, ACM, wb_name, sheet_num, tex):


    #################
    data = pd.DataFrame(columns=['Team', 'Elo'])
    wb = openpyxl.load_workbook('Elo_test.xlsx')
    ws = wb.active

    for j in range(2, 34):
        data = data.append({'Team': ws[f'A{j}'].value, 'Elo': ws[f'B{j}'].value}, ignore_index=True)

    elo = data.set_index('Team', drop=True).to_dict()['Elo']

    #################

    wb1 = openpyxl.load_workbook(f'{wb_name}')
    ws1 = wb1.create_sheet(f'sheet{sheet_num}')

    ws1.append(['K', 'HTA', 'MVM_A', 'MVM_B', 'ACF', 'ACM'])
    ws1.append([K, HTA, MVM_A, MVM_B, ACF, ACM])

    header = ['Date', 'Home_team', 'Away_team', 'home_prob', 'home_win']
    ws1.append(header)

    #################

    ##################
    for i in range(14):
        start = pd.to_datetime('2022-10-06', format='%Y-%m-%d') + timedelta(days=i)
    ##################

        today_date = start.strftime('%Y-%m-%d')
        yest_games = []
        for _, t in enumerate(tex):
            if str(t.find('th').text) == str(today_date):
                yest_games.append((t.find_all('td')[0].text, t.find_all('td')[1].text, t.find_all('td')[2].
                                                text, t.find_all('td')[3].text))


    #######################
        for game in yest_games:
            elo_diff = -(elo[game[2]] + HTA - elo[game[0]])/400
            AC = ACF / ((abs(elo_diff) * ACM) + ACF)
            prob_a = 1. / (np.power(10, elo_diff) + 1)
            prob_b = 1. - prob_a
            MoV = int(game[3]) - int(game[1]) * AC
            MVM = MVM_A * np.log(abs(MoV)) + MVM_B
            # print(game[2], prob_a, ':', game[0], prob_b)
            if MoV > 0:
                TWa = 1.
                TWb = 0.
            else:
                TWa = 0.
                TWb = 1.

            ws1.append([today_date, game[2], game[0], prob_a, TWa])

            shifta = K * MVM * (TWa - prob_a)
            shiftb = K * MVM * (TWb - prob_b)
            A = float(elo[game[2]])
            B = float(elo[game[0]])
            elo[game[2]] = A + shifta
            elo[game[0]] = B + shiftb


        #######################

    wb1.save('Main.xlsx')
    wb1.close()
    if sheet_num % 100 == 0:
        print(f'Done {sheet_num}')
     ###################

