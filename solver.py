import numpy as np
import networkx as nx

from utils import *
from parse import *
from best_score import *

import random
from tqdm import tqdm, trange
from multiprocessing import Pool, cpu_count


def get_budget(n):
    if n <= 30:
        return 1, 15
    elif n <= 50:
        return 3, 50
    else:
        return 5, 100


def longest_simple_paths(graph, source, target):
    longest_paths = []
    longest_path_length = 0
    for path in nx.all_simple_paths(graph, source=source, target=target):
        if len(path) > longest_path_length:
            longest_path_length = len(path)
            longest_paths.clear()
            longest_paths.append(path)
        elif len(path) == longest_path_length:
            longest_paths.append(path)
    return longest_paths


def complete_search_helper(args):
    G, removable_nodes, iterations = args
    node_budget, edge_budget = get_budget(G.number_of_nodes())
    source, target = 0, G.number_of_nodes() - 1
    max_score, max_c, max_k = 0, [], []

    for iteration in range(iterations):
        # longest_path = longest_simple_paths(G, source, target)
        _G = G.copy()
        c, k = [], []

        # randomized algorithm
        removed_nodes = random.sample(removable_nodes, node_budget)
        for node in removed_nodes:
            _G.remove_node(node)
            c.append(node)

        removed_edges = random.sample(
            list(_G.edges), min(_G.number_of_edges(), edge_budget))
        for edge in removed_edges:
            _G.remove_edge(edge[0], edge[1])
            k.append(edge)

            if not nx.is_connected(_G):
                _G.add_edge(edge[0], edge[1], weight=edge[-1])
                k.remove(edge)

        if is_valid_solution(G, c, k):
            score = calculate_score(G, c, k)
            if score > max_score:
                max_score, max_c, max_k = score, c, k

    return max_score, max_c, max_k


def complete_search_mt(G, iterations, BLOCK_SIZE=256):
    articulartion_points = nx.articulation_points(G)
    removable_nodes = list(filter(lambda x: x not in articulartion_points, list(
        np.arange(G.number_of_nodes())[1:-1])))

    p = Pool(cpu_count())
    args = [[G, removable_nodes, BLOCK_SIZE]] * (iterations // BLOCK_SIZE)
    results = p.imap(complete_search_helper, args)

    max_score, max_c, max_k = 0, [], []
    n_updates = 0

    for result in tqdm(results):
        n_updates += BLOCK_SIZE

        if result[0] > max_score:
            max_score, max_c, max_k = result

    return max_score, max_c, max_k


def complete_search(G, iterations):
    node_budget, edge_budget = get_budget(G.number_of_nodes())
    source, target = 0, G.number_of_nodes() - 1
    max_score, max_c, max_k = 0, [], []

    articulartion_points = nx.articulation_points(G)
    removable_nodes = list(filter(lambda x: x not in articulartion_points, list(
        np.arange(G.number_of_nodes())[1:-1])))

    status = trange(iterations, desc="Max Score", leave=True)
    n_updates = 0
    for iteration in status:
        # longest_path = longest_simple_paths(G, source, target)
        _G = G.copy()
        c, k = [], []

        # randomized algorithm
        removed_nodes = random.sample(removable_nodes, node_budget)
        for node in removed_nodes:
            _G.remove_node(node)
            c.append(node)

        removed_edges = random.sample(
            list(_G.edges), min(_G.number_of_edges(), edge_budget))
        for edge in removed_edges:
            _G.remove_edge(edge[0], edge[1])
            k.append(edge)

            if not nx.is_connected(_G):
                _G.add_edge(edge[0], edge[1], weight=edge[-1])
                k.remove(edge)

        if is_valid_solution(G, c, k):
            score = calculate_score(G, c, k)
            if score > max_score:
                max_score, max_c, max_k = score, c, k
                n_updates += 1
                status.set_description(f"Max Score {max_score} out of {n_updates}")


    return max_c, max_k


if __name__ == '__main__':
    best_sols = get_best_sols_data()

    for file in os.listdir(f'inputs/small'):
        graph_name = file.split('.')[0]
        best_score = best_sols[graph_name]['score']
        if best_score == 1:
            pass
        # print(graph_name)
        input_file = f'inputs/small/{graph_name}.in'
        output_file = f'outputs/small/{graph_name}.out'
        g = read_input_file(input_file)
        score, c, k = complete_search_mt(g, 20000)

        if score > best_score:
            print("monte carlo gave better score for graph " + str(graph_name))
            write_output_file(g, c, k, output_file)
    for file in os.listdir(f'inputs/medium'):
        graph_name = file.split('.')[0]
        if best_sols[graph_name]['ranking'] < 75:
            continue
        #print(graph_name)
        input_file = f'inputs/small/{graph_name}.in'
        output_file = f'outputs/small/{graph_name}.out'

        g = read_input_file(input_file)
        score, c, k = complete_search_mt(g, 10000)

        if score > best_score:
            print("monte carlo gave better score for graph " + str(graph_name))
            write_output_file(g, c, k, output_file)

