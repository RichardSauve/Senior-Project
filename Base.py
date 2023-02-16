import pandas as pd
import requests
import numpy as np
from bs4 import BeautifulSoup as bs
from datetime import date
import cloudscraper as cs
import openpyxl
from Basic_Web_Scrap import web_scrap
from Games import todays_games
import time

import warnings
warnings.filterwarnings('ignore')


cols = ["AvAge", 'GP', "W", "L", "OL", "PTS", "PTS%", "GF", "GA", "SRS", "SOS", "GF/G", "GA/G", "PP", "PPO", "PP%",
        "PPA", "PPOA", "PK%", "SH", "SHA", "S", "S%", "SA", "SV%", "PDO", "SO"]

cols_to_drop = ["AvAge", 'GP', "W", "L", "OL", "PTS", "GF", "GA", "PP", "PPO",
                "PPA", "PPOA", "SH", "SHA", "S", "SA", "PDO", "SO"]

cols_to_keep = ['PTS%', 'SRS', 'SOS', 'GF/G', 'GA/G', 'PP%', 'PK%', 'SV%', 'S%']


dict_of_teams = {'Anaheim Ducks': "ANA", 'Arizona Coyotes': "ARI", 'Boston Bruins': 'BOS', 'Buffalo Sabres': "BUF",
                 'Carolina Hurricanes': "CAR",
                 'Columbus Blue Jackets': "CBJ", 'Calgary Flames': 'CGY', 'Chicago Blackhawks': "CHI",
                 "Colorado Avalanche": "COL", 'Dallas Stars': "DAL",
                 'Detroit Red Wings': "DET", 'Edmonton Oilers': "EDM", 'Florida Panthers': "FLA",
                 'Los Angeles Kings': "LAK",
                 'Minnesota Wild': "MIN", 'Montreal Canadiens': "MTL", 'New Jersey Devils': "NJD",
                 'Nashville Predators': "NSH", 'New York Islanders': "NYI",
                 'New York Rangers': "NYR", 'Ottawa Senators': "OTT", 'Philadelphia Flyers': "PHI",
                 'Pittsburgh Penguins': "PIT", 'Seattle Kraken': "SEA", 'San Jose Sharks': "SJS",
                 'St. Louis Blues': "STL", 'Tampa Bay Lightning': "TBL", 'Toronto Maple Leafs': "TOR",
                 'Vancouver Canucks': "VAN",
                 'Vegas Golden Knights': "VEG", 'Winnipeg Jets': "WPG", 'Washington Capitals': "WSH"}


current_date, games = todays_games()
df = web_scrap()
df = df.set_index('Team', drop=True)


wb = openpyxl.load_workbook('games_data.xlsx')
ws = wb.active

for game in games:
    away, home = game
    A = df.loc[dict_of_teams[away]][cols_to_keep]
    H = df.loc[dict_of_teams[home]][cols_to_keep]
    ser = H - A
    final = [current_date, away, home]
    for v in ser.values:
        final.append(np.round(v, 4))
    print(final)
    ws.append(final)

wb.save('games_data.xlsx')
wb.close()
