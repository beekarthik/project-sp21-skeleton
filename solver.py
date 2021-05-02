import networkx as nx
from parse import read_input_file, write_output_file, write_input_file
from utils import *
import sys
from os.path import basename, normpath
import glob
import random
import collections
import heapq
import operator, math, pprint
from collections import defaultdict


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

    node_budget, edge_budget = get_budget(G.number_of_nodes())

    for i in range(edge_budget):
        shortest_path = nx.shortest_path(G, s, t, weight="weight", method='dijkstra')
        shortest_path_as_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(0, len(shortest_path) - 1)]
        heuristic = k_short_path_heuristic(G, s, t, k=10, edge=True)

        edge_removed = False
        while not edge_removed:
            target = max(shortest_path_as_edges, key=lambda x: heuristic[x])
            shortest_path_as_edges.remove(target)
            print(target)
            weight = G[target[0]][target[1]]['weight']
            G.remove_edge(target[0], target[1])
            if nx.is_connected(G):
                edge_removed = True
                print(str(target) + " was removed")
            else:
                G.add_edge(target[0], target[1], weight=weight)

    print(path_and_weight(G, nx.shortest_path(G, s, t, weight="weight", method='dijkstra')))


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
            path = next(short_path_generator)
            if not path:
                break
            path, weight = path_and_weight(G, path)
            if show_data:
                print((path, weight))
            if path is StopIteration:
                break
            for n in range(0, len(path) - 1):
                common_edges[(path[n], path[n + 1])] += 100 / weight

        # remove = max(common_edges.items(), key=operator.itemgetter(1))[0]
        if show_data:
            for a, b, in common_edges:
                print(str((a, b)) + ": " + str(common_edges[(a, b)]))
            #print(remove)

        return common_edges #remove, common_edges

    else:
        common_nodes = defaultdict(lambda: 0)
        if show_data:
            print(path_and_weight(G, nx.shortest_path(G, s, t, weight="weight", method='dijkstra')))
        short_path_generator = nx.shortest_simple_paths(G, s, t, weight="weight");

        for i in range(k):
            path = next(short_path_generator)
            if show_data:
                print(path_and_weight(G, path))
            if path is StopIteration:
                break
            path = path[1:-1]
            path, weight = path_and_weight(G, path)
            for node in path:
                common_nodes[node] += 100 / math.log(math.log(weight))

        #removal_node = max(common_nodes.items(), key=operator.itemgetter(1))[0]

        return common_nodes #removal_node, common_nodes


def generate_rand_graph(n, path):
    G = nx.dense_gnm_random_graph(n, random.randint(0, n * (n - 1) / 2))
    for (u, v, w) in G.edges(data=True):
        w['weight'] = round(random.random(), 3)
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
    a = make_valid_graph(8, "a.in")
    print(list(a.edges))

    # a.remove_node(1)
    # print(list(a.edges))

    solve(a)
    #c, k = solve(a)
    #assert is_valid_solution(a, c, k)
    #print("Shortest Path Difference: {}".format(calculate_score(a, c, k)))

    # write_output_file(G, c, k, 'outputs/small-1.out')


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
# if __name__ == '__main__':
#     inputs = glob.glob('inputs/*')
#     for input_path in inputs:
#         output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G)
#         assert is_valid_solution(G, c, k)
#         distance = calculate_score(G, c, k)
#         write_output_file(G, c, k, output_path)
