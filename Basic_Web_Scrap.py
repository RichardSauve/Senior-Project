import pandas as pd
from bs4 import BeautifulSoup as bs
import cloudscraper as cs
import warnings
import time

warnings.filterwarnings('ignore')

def web_scrap():
    """
    Description:
        Webscrapes hockey reference for the aggregate stats for each team in the NHL

    Args:
        N/A

    Returns:
        final(pd.Dataframe): A dataframe containing the desire remaining aggregate stats of all the NHL teams

    """
    cols = ["AvAge", 'GP', "W", "L", "OL", "PTS", "PTS%", "GF", "GA", "SRS", "SOS", "GF/G", "GA/G", "PP", "PPO", "PP%",
            "PPA", "PPOA", "PK%", "SH", "SHA", "S", "S%", "SA", "SV%", "PDO", "SO"]

    final: pd.DataFrame = pd.DataFrame(columns=['Team'] + cols)
    teams = ["ANA", "ARI", 'BOS', "BUF", "CAR", "CBJ", 'CGY', "CHI", "COL", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL",
             "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "VAN", "VEG", "WPG", "WSH"]

    args = {'browser': 'chrome', 'desktop': True, 'platform': 'windows'}

    print(len(teams))
    for team in teams:
        url = f'https://www.hockey-reference.com/teams/{team}/2023.html'

        scraper = cs.create_scraper(browser=args)
        response = scraper.get(url)

        soup = bs(response.text, 'html.parser')
        text: bs = soup.find('table', id='team_stats')

        dat = {'Team': team}
        for i, c in enumerate(cols):
            dat[c] = float(text.find_all('td')[i].text)

        final = final.append(dat, ignore_index=True)
        print(f'{team} done!')
        time.sleep(10)

    return final
