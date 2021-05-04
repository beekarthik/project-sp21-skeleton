import networkx as nx
from parse import read_input_file, write_output_file, write_input_file
from utils import *
import sys, glob
from os.path import basename, normpath
import random, heapq
import operator, math, pprint
from collections import defaultdict
import time
from curr_score import *
from multiprocessing import *

best_sols = get_best_sols_data()


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


def solve(G, n, e, b, w):
    """
    Args:
        G: networkx.Graph
        n: heuristic num paths for nodes
        e: heuristic num paths for edges
        b: skip first b shortest paths for nodes
        w: skip first w shortest paths for edges
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """
    s = 0
    t = G.number_of_nodes() - 1
    c = []
    k = []

    node_budget, edge_budget = get_budget(G.number_of_nodes())
    short_path_generator = nx.shortest_simple_paths(G, s, t, weight="weight")
    try:
        for i in range(b):
            next(short_path_generator)
    except StopIteration:
        pass

    for i in range(node_budget):
        try:
            shortest_path = next(short_path_generator)
        except StopIteration:
            break

        shortest_path_as_nodes = shortest_path[1:-1]
        heuristic = k_short_path_heuristic(G, s, t, k=n, edge=False, show_data=False)
        artic_points = list(nx.articulation_points(G))  # maybe turn into set for potential speed increase

        node_removed = False
        while not node_removed and shortest_path_as_nodes:
            if not shortest_path_as_nodes:
                break
            target = max(shortest_path_as_nodes, key=lambda x: heuristic[x])
            if target not in artic_points:      # do not remove node
                node_removed = True
                G.remove_node(target)
                c.append(target)
                # assert nx.is_connected(G), 'should still be connected' # REMOVE THIS LINE eventually
                assert target != s, 'cannot remove source'
                assert target != t, 'cannot remove sink'
            else:                               # remove node
                shortest_path_as_nodes.remove(target)
        short_path_generator = nx.shortest_simple_paths(G, s, t, weight="weight")       # regenerate generator

    try:
        for i in range(w):
            next(short_path_generator)
    except StopIteration:
        pass

    for i in range(edge_budget):
        try:
            shortest_path = next(short_path_generator)
        except StopIteration:
            break
        shortest_path_as_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(0, len(shortest_path) - 1)]
        heuristic = k_short_path_heuristic(G, s, t, k=e, edge=True)

        edge_removed = False
        while not edge_removed and shortest_path_as_edges:
            target = max(shortest_path_as_edges, key=lambda x: heuristic[x])
            shortest_path_as_edges.remove(target)
            weight = G[target[0]][target[1]]['weight']
            G.remove_edge(target[0], target[1])
            if nx.is_connected(G):
                edge_removed = True
                k.append(target)
            else:
                G.add_edge(target[0], target[1], weight=weight)
        short_path_generator = nx.shortest_simple_paths(G, s, t, weight="weight")       # regenerate generator

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

        if show_data:
            for a, b, in common_edges:
                print(str((a, b)) + ": " + str(common_edges[(a, b)]))

        return common_edges

    else:
        common_nodes = defaultdict(lambda: 0)
        if show_data:
            print(path_and_weight(G, nx.shortest_path(G, s, t, weight="weight", method='dijkstra')))
        short_path_generator = nx.shortest_simple_paths(G, s, t, weight="weight");

        for i in range(k):
            try:
                path = next(short_path_generator)
            except StopIteration:
                break
            if show_data:
                print(path_and_weight(G, path))
            path, weight = path_and_weight(G, path)
            for node in path[1:-1]:
                common_nodes[node] += 100 / weight

        return common_nodes

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


def meta_heuristic_bash(folder, file):
        graph_name = file.split('.')[0]
        print(graph_name)
        input_file = f'inputs/{folder}/{graph_name}.in'
        output_file = f'outputs/{folder}/{graph_name}.out'

        best_score = best_sols[graph_name]['score']
        best_c = []
        best_k = []
        score_change = False
        G = read_input_file(input_file)

        """skipping first 1 thru 7 paths"""

        for i in range(0, 8):
            for j in range(0, 4):
                c, k = solve(G.copy(), 3, 3, i, j)
                new_score = calculate_score(G, c, k)

                if new_score > best_score:
                    # print("i: " + str(i) + ", j: " + str(j) + " gave improvement " + str(new_score - best_score))
                    best_score = new_score
                    best_c = c
                    best_k = k
                    score_change = True

                c, k = solve(G.copy(), 12, 8, i, j)
                new_score = calculate_score(G, c, k)

                if new_score > best_score:
                    # print("i: " + str(i) + ", j: " + str(j) + " gave improvement " + str(new_score - best_score))
                    best_score = new_score
                    best_c = c
                    best_k = k
                    score_change = True
        if score_change:
            write_output_file(G, best_c, best_k, output_file)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    if path == "all":

        pool = Pool()
        for folder in os.listdir("inputs"):
            # if folder != "small": # only run on small inputs for now
            #     continue
            for file in os.listdir(f'inputs/{folder}'):
                try:
                    pool.apply_async(meta_heuristic_bash, [folder, file])
                except Exception:
                    continue

        pool.close()
        pool.join()

    else:
        G = read_input_file(path)
        c, k = solve(G.copy())
        score = calculate_score(G, c, k)
        print(score)

    print("writing answers")
    write_best_sols_data(calculate_best_scores())
