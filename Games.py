from bs4 import BeautifulSoup as bs
from datetime import date
import cloudscraper as cs
import warnings
warnings.filterwarnings('ignore')

def todays_games():

    """
    Description:
        Webscrapes hockey reference for the current games

    Args:
        N/A

    Returns:
        today(str): The current date
            'YYYY-mm-dd'

        today_games(list): A list of tuples containing the away team and the home team
            (Away Team, Home Team)
    """

    url = 'https://www.hockey-reference.com/leagues/NHL_2023_games.html'

    today = str(date.today())
    print("Today date is: ", today)

    args = {'browser': 'firefox', 'desktop': True, 'platform': 'windows'}

    scraper = cs.create_scraper(browser=args)
    response = scraper.get(url)

    soup = bs(response.text, 'html.parser')
    tex = soup.find('table').find_all('tr')

    today_games = []
    for i, t in enumerate(tex):
        if t.find('th').text == today:
            a = t.find_all('a')
            today_games.append((a[0].text, a[1].text))

    return today, today_games
