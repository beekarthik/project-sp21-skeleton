import sys
import os
import json
import parse
import io

try:
    to_unicode = unicode
except NameError:
    to_unicode = str


def calculate_best_scores():
    best_sols = dict()
    error = []
    for folder in os.listdir("inputs"):
        for file in os.listdir(f'inputs/{folder}'):
            graph_name = file.split('.')[0]
            #print(graph_name)
            output_file = f'outputs/{folder}/{graph_name}.out'
            try:
                G = parse.read_input_file(f'inputs/{folder}/{graph_name}.in')
                score, cities, edges = parse.read_output_file2(G, output_file)
                best_sols[graph_name] = {"score": score, "c": cities, "e": edges}
            except AssertionError:
                error.append(graph_name)

    print(error)

    return best_sols



def get_best_sols_data(filename="best_sols.json"):
    with open(filename) as data_file:
        data = json.load(data_file)

    return data

def write_best_sols_data(dict, filename="best_sols.json"):
    with io.open(filename, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(dict,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))
