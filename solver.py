import networkx as nx
from parse import read_input_file, write_output_file, write_input_file
from utils import *
import sys
from os.path import basename, normpath
import glob
import random
import collections
import heapq
import operator
import numpy as np
from collections import defaultdict
import time

from tqdm import tqdm, trange
from multiprocessing import Pool, cpu_count


def path_and_weight(G, path):
    return path, path_weight(G, path)


def path_weight(G, path):
    total_weight = 0
    for n in range(0, len(path) - 1):
        total_weight += G[path[n]][path[n + 1]]['weight']

    return total_weight


def get_budget(n):
    if n <= 30:
        return 1, 15
    elif n <= 50:
        return 3, 50
    else:
        return 5, 100


def invert_graph(G):
    _G = G.copy()

    for edge in _G.edges.data():
        edge[-1]['weight'] *= -1

    return _G


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

        removed_edges = random.sample(list(_G.edges), min(_G.number_of_edges(), edge_budget))
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
    removable_nodes = list(filter(lambda x: x not in articulartion_points, list(np.arange(G.number_of_nodes())[1:-1])))

    p = Pool(cpu_count())
    args = [[G, removable_nodes, BLOCK_SIZE]] * iterations
    results = p.imap(complete_search_helper, args)

    max_score, max_c, mac_k = 0, [], []
    n_updates = 0

    for result in tqdm(results):
        n_updates += BLOCK_SIZE

        if result[0] > max_score:
            max_score, max_c, max_k = result
            print(max_score)

def complete_search(G, iterations):
    node_budget, edge_budget = get_budget(G.number_of_nodes())
    source, target = 0, G.number_of_nodes() - 1
    max_score, max_c, max_k = 0, [], []

    articulartion_points = nx.articulation_points(G)
    removable_nodes = list(filter(lambda x: x not in articulartion_points, list(np.arange(G.number_of_nodes())[1:-1])))

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

        removed_edges = random.sample(list(_G.edges), min(_G.number_of_edges(), edge_budget))
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

def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """
    s = 0
    t = G.number_of_nodes() - 1
    c = []
    k = []

    node_budget, edge_budget = get_budget(G.number_of_nodes())

    for i in range(node_budget):
        # with probability p, skip this path
        shortest_path = nx.shortest_path(G, s, t, weight="weight", method='dijkstra')
        shortest_path_as_nodes = shortest_path[1:-1]
        heuristic = k_short_path_heuristic(G, s, t, k=10, edge=False, show_data=False)

        # maybe turn into set for potential speed increase
        artic_points = list(nx.articulation_points(G))

        node_removed = False
        while not node_removed and shortest_path_as_nodes:
            if not shortest_path_as_nodes:
                break
            target = max(shortest_path_as_nodes, key=lambda x: heuristic[x])
            if target not in artic_points:
                node_removed = True
                G.remove_node(target)
                #print(str(target) + " was removed")
                c.append(target)
                # REMOVE THIS LINE eventually
                assert nx.is_connected(G), 'should still be connected'
                assert target != s, 'cannot remove source'
                assert target != t, 'cannot remove sink'
            else:
                shortest_path_as_nodes.remove(target)

    for i in range(edge_budget):
        # with probability p, skip this path
        shortest_path = nx.shortest_path( G, s, t, weight="weight", method='dijkstra')
        shortest_path_as_edges = [(shortest_path[i], shortest_path[i + 1])
                                  for i in range(0, len(shortest_path) - 1)]
        heuristic = k_short_path_heuristic(G, s, t, k=10, edge=True)

        edge_removed = False
        while not edge_removed and shortest_path_as_edges:
            target = max(shortest_path_as_edges, key=lambda x: heuristic[x])
            shortest_path_as_edges.remove(target)
            weight = G[target[0]][target[1]]['weight']
            G.remove_edge(target[0], target[1])
            if nx.is_connected(G):
                edge_removed = True
                #print(str(target) + " was removed")
                k.append(target)
            else:
                G.add_edge(target[0], target[1], weight=weight)

    #print(path_and_weight(G, nx.shortest_path(G, s, t, weight="weight", method='dijkstra')))
    return c, k


def k_short_path_heuristic(G, s, t, k=10, edge=True, show_data=False):
    """
    Returns (node/edge, dictionary_of_data)
    """
    if edge:
        common_edges = defaultdict(lambda: 0)

        if show_data:
            print(path_and_weight(G, nx.shortest_path(G, s, t, weight="weight", method='dijkstra')))

        short_path_generator = nx.shortest_simple_paths(G, s, t, weight="weight")
        for i in range(k):
            try:
                path = next(short_path_generator)
            except StopIteration:
                break

            path, weight = path_and_weight(G, path)
            if show_data:
                print((path, weight))

            for n in range(0, len(path) - 1):
                common_edges[(path[n], path[n + 1])] += 100 / weight

        # remove = max(common_edges.items(), key=operator.itemgetter(1))[0]
        if show_data:
            for a, b, in common_edges:
                print(str((a, b)) + ": " + str(common_edges[(a, b)]))
            # print(remove)

        return common_edges  # remove, common_edges

    else:
        common_nodes = defaultdict(lambda: 0)
        if show_data:
            print(path_and_weight(G, nx.shortest_path(
                G, s, t, weight="weight", method='dijkstra')))
        short_path_generator = nx.shortest_simple_paths(
            G, s, t, weight="weight")

        for i in range(k):
            try:
                path = next(short_path_generator)
            except StopIteration:
                break
            if show_data:
                print(path_and_weight(G, path))
            # print(path)
            path, weight = path_and_weight(G, path)
            for node in path[1:-1]:
                common_nodes[node] += 100 / weight

        #removal_node = max(common_nodes.items(), key=operator.itemgetter(1))[0]

        return common_nodes  # removal_node, common_nodes


def generate_rand_graph(n, path):
    G = nx.dense_gnm_random_graph(n, random.randint(0, n * (n - 1) / 2))
    for (u, v, w) in G.edges(data=True):
        w['weight'] = int(random.random() * 30)
    write_input_file(G, path)
    return G


def make_valid_graph(n, path):
    g = None
    while not is_valid_graph(g):
        g = generate_rand_graph(n, path)
    return g


if __name__ == '__main__':
    # assert len(sys.argv) == 2
    # path = sys.argv[1]
    # G = read_input_file(path)
    #a = make_valid_graph(8, "a.in")
    # print(list(a.edges(data=True)))
    a = read_input_file("inputs/medium/medium-221.in")
    c, k = complete_search_mt(a.copy(), 1000)

    assert is_valid_solution(a, c, k)
    print("Shortest Path Difference: {}".format(calculate_score(a, c, k)))


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
# if __name__ == '__main__':
#     inputs = glob.glob('inputs/large/*')
#     counter = 1
#     t0 = time.time()
#     for input_path in inputs:
#         print("reading graph " + str(counter) + "/300")
#         output_path = 'outputs/large/' + basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G.copy())
#         assert is_valid_solution(G, c, k)
#         distance = calculate_score(G, c, k)
#         write_output_file(G, c, k, output_path)
#         counter += 1
#         if counter % 50 == 0:
#             print(str(time.time() - t0) + " seconds elapsed.")
