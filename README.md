# CS 170 Project Spring 2021


## How to reproduce

- If submission already exists, copy the html table of our teams performance from the website into `leaderboard.html`
- Then run `python3 web_scraper.py` which uses both `curr_score.py` and `utils.py` to parse the `leaderboard.html` data and store the solution, score, and rank for each graph and write it to `best_sols.json`



- Then run `python3 solver.py all` which will run our solver that is based on looking at the k-shortest paths and being quasi-greedy
- If `best_sols.json` has been created then this function skips any file that has already achieved rank 1. This massively improves performance as we do not waste time recomputing good solutions
- Finally run `python3 random_solver.py all` which tries to solve the problem using a fixed iteration count and good old fashioned randomness - this was primarily used for hammering out small and medium inputs

The above steps run all the code we tried at once when in reality we ran it chunk by chunk, varying only one hyperparameter at a time and trying to score as many rank 1 paths so that computation could be saved.


## Resources Used
- Google Cloud Compute - got a free account and used an 8cpu machine for 15 hours total. Google billed ~$10.00
- multiprocessing library - allowed us to maximize google cloud compute VM
- defaultdict - using this allows us to not worry about checking for existence which is always annoying
- BeautifulSoup - allowed us to parse our leaderboard table and maintain rankings which helped us save time
