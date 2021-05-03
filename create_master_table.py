from solver import *
from parse import *
import os

if __name__ == '__main__':
    for small_graph in os.listdir('inputs/small/'):
        print(f"Simulating: {small_graph}")

        graph = read_input_file(f'inputs/small/{small_graph}')
        score, c, k = complete_search_mt(graph.copy(), 10000)
        print(f"Max Score on {small_graph}: {score}")

        small_graph_out_path = 'outputs_mt/small/{small_graph[:-3]}.out'
        write_output_file(graph, c, k, small_graph_out_path)

