import pandas as pd
import requests
from bs4 import BeautifulSoup
from curr_score import *

URL = "http://berkeley-cs170.github.io/project-leaderboard-sp21/?team=cashmoneyaintnothingfunny"



def get_ranking_dict(url=URL):

    file = open("leaderboard.html")
    soup = BeautifulSoup(file, 'html.parser')
    rows = soup.find_all('tr')[1:]

    data = get_best_sols_data("best_sols.json")


    for r in rows:
        name = r.contents[1].text
        place = r.contents[3].text
        data[name]["ranking"] = int(place)
        print(data[name])

    write_best_sols_data(data, "best_sols.json")

get_ranking_dict()
