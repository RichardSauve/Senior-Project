import numpy as np
import pandas as pd
import openpyxl
import warnings
import multiprocessing
import datetime as dt
from datetime import date, timedelta
import cloudscraper as cs
from bs4 import BeautifulSoup as bs

warnings.filterwarnings('ignore')

################
def probability(K, HTA, MVM_A, MVM_B, ACF, ACM, wb_name, sheet_num, yest_games, elo_sub):
    elo = elo_sub.copy()
    #################

    wb1 = openpyxl.load_workbook(f'{wb_name}')
    ws1 = wb1.create_sheet(f'sheet{sheet_num}')

    ws1.append(['K', 'HTA', 'MVM_A', 'MVM_B', 'ACF', 'ACM'])
    ws1.append([K, HTA, MVM_A, MVM_B, ACF, ACM])

    header = ['Home_team', 'Away_team', 'home_prob', 'home_win']
    ws1.append(header)

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

        ws1.append([game[2], game[0], prob_a, TWa])

        shifta = K * MVM * (TWa - prob_a)
        shiftb = K * MVM * (TWb - prob_b)
        A = float(elo[game[2]])
        B = float(elo[game[0]])
        elo[game[2]] = A + shifta
        elo[game[0]] = B + shiftb


        #######################

    wb1.save('Main.xlsx')
    wb1.close()

     ###################

