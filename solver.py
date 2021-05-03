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
        if b == 0:  # skip 1
            if i == 0:
                continue
        elif b == 1:  # skip 1 and 2
            if i == 0 or i == 1:
                continue
        elif b == 2:  # skip 1 and 3
            if i == 0 or i == 2:
                continue
        elif b == 3:  # skip 1, 2, and 3
            if i == 0 or i == 1 or i == 2:
                continue

        shortest_path = nx.shortest_path(G, s, t, weight="weight",
                                         method='dijkstra')  # with probability p, skip this path
        shortest_path_as_nodes = shortest_path[1:-1]
        heuristic = k_short_path_heuristic(G, s, t, k=n, edge=False, show_data=False)
        artic_points = list(nx.articulation_points(G))  # maybe turn into set for potential speed increase

        node_removed = False
        while not node_removed and shortest_path_as_nodes:
            if not shortest_path_as_nodes:
                break
            target = max(shortest_path_as_nodes, key=lambda x: heuristic[x])
            if target not in artic_points:
                node_removed = True
                G.remove_node(target)
                c.append(target)
                # assert nx.is_connected(G), 'should still be connected' # REMOVE THIS LINE eventually
                assert target != s, 'cannot remove source'
                assert target != t, 'cannot remove sink'
            else:
                shortest_path_as_nodes.remove(target)

    for i in range(edge_budget):
        if w == 1:  # skip 1
            if i == 0 or i == 1:
                continue
        elif w == 2:  # skip 1 and 2
            if i == 0 or i == 2:
                continue
        shortest_path = nx.shortest_path(G, s, t, weight="weight",
                                         method='dijkstra')  # with probability p, skip this path
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

        if best_sols[graph_name]['ranking'] == 1:
            return

        best_score = best_sols[graph_name]['score']
        best_c = []
        best_k = []
        score_change = False
        G = read_input_file(input_file)


        for i in range(0, 4):
            for b in range(4):
                for w in range(3):
                    # 000, 001, 010, 011, 100, 101, 110, 111
                    j = 1
                    if i == 0:
                        j = 1
                    else:
                        j = 5 * i
                    c, k = solve(G.copy(), j, j, b, w)
                    new_score = calculate_score(G, c, k)


                    if new_score > best_score:
                        print("b: " + str(b) + ", w: " + str(w) + " gave improvement")
                        best_score = new_score
                        best_c = c
                        best_k = k
                        score_change = True

        # for i in range(10):
        #     for j in range(10):
        #         print((i, j))
        #         c, k = solve(G.copy(), i, j)
        #         new_score = calculate_score(G, c, k)
        #         if new_score > best_score:
        #             best_score = new_score
        #             best_c = c
        #             best_k = k
        #             score_change = True

        if score_change:
            write_output_file(G, best_c, best_k, output_file)




if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    if path == "all":

        pool = Pool()
        for folder in os.listdir("inputs"):
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
#     #a = make_valid_graph(8, "a.in")
#     #print(list(a.edges(data=True)))
#     a = read_input_file("inputs/medium/medium-221.in")
#     plot_graph(a)
#
#     c, k = solve(a.copy())
#     assert is_valid_solution(a, c, k)
#     print("Shortest Path Difference: {}".format(calculate_score(a, c, k)))

# write_output_file(G, c, k, 'outputs/small-1.out')


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
