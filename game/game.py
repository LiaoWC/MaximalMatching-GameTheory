import numpy as np
import graphviz
import random
from copy import deepcopy
from typing import Callable, List, Tuple, Optional
import matplotlib.pyplot as plt
import ast
import json as std_json


class Graph:
    def __init__(self, strategies: Optional[List], edges: Optional[np.ndarray], weights: Optional[List]):
        """
        :param strategies: List with length n. (n is the number of vertices)
        :param edges: nxn ndarray.
        """
        self.strategies = strategies
        self.edges = edges
        self.weights = weights

    def __repr__(self):
        representation = "\nVertexNo\tEdges\tStrategies\tWeights\n"
        for i in range(len(self.edges) + 1):
            if i > 0:
                for j in range(len(self.edges) + 1):
                    if j == 0:
                        representation += f"{i:03} -> [ "
                    else:
                        representation += str(self.edges[i - 1][j - 1]) + " "
                representation += f"]\ts{self.strategies[i - 1]}\tw{self.weights[i - 1]}\n"
        return representation

    def n_vertices(self):
        return len(self.edges)

    def plot_graph(self, filename='plotted_graph', show=False, picked_vertex: Optional[int] = None,
                   have_better_choice: Optional[List[int]] = None):
        d = graphviz.Digraph(filename=filename)
        d.format = 'png'
        d.attr(rankdir='LR', concentrate='True')
        # Vertex index from 1
        # Vertices
        d.attr('node', shape='circle')
        for vertex in range(self.n_vertices()):
            color = 'lightblue2' if self.strategies[vertex] != 0. else 'black'
            style = 'filled' if self.strategies[vertex] != 0. else ''
            shape = 'octagon' if (have_better_choice is not None and vertex in have_better_choice) else 'circle'
            shape = 'tripleoctagon' if (picked_vertex is not None and vertex == picked_vertex) else shape
            v_name = f'{vertex}\n(w{self.weights[vertex]})'
            # if self.strategies[vertex] != 0.:  # If action is not 0, fill color.
            #     d.node(f'{vertex}\n(w{self.weights[vertex]})', color='lightblue2', style='filled')
            # else:
            #     d.node(f'{vertex}\n(w{self.weights[vertex]})')
            d.node(v_name, color=color, style=style, shape=shape)
        for i in range(self.n_vertices()):
            for j in range(self.n_vertices()):
                if self.edges[i][j] != 0.:
                    d.edge(f'{i}\n(w{self.weights[i]})', f'{j}\n(w{self.weights[j]})')
        d.render(view=False)
        if show is True:
            plt.imshow(plt.imread(f'{filename}.png'))
            plt.axis('off')
            plt.show()

    def strategy_1_weight_sum(self) -> float:
        s_w_sum = 0.0
        for v, s in enumerate(self.strategies):
            if s != 0.0:
                s_w_sum += self.weights[v]
        return s_w_sum

    def deserialize(self, codes: str):
        contents = ast.literal_eval(codes)
        # self.edges = self.deserialize_edges(contents['edges'])
        # self.strategies = self.deserialize_strategies(contents['strategies'])
        # self.weights = self.deserialize_weights(contents['weights'])
        self.edges = np.array(contents['edges'])
        self.strategies = contents['strategies']
        self.weights = contents['weights']

    @staticmethod
    def deserialize_edges(s):
        return np.array([line.split(' ') for line in s.split('\n')])

    @staticmethod
    def serialize_edges(edges: np.ndarray):
        s = ''
        for row in edges:
            for element in row:
                s += element + ' '
        return s.strip()

    @staticmethod
    def deserialize_strategies(s):
        return list(s.split(' '))

    @staticmethod
    def serialize_strategies(strategies):
        return ' '.join(map(str, strategies))

    @staticmethod
    def deserialize_weights(s):
        return list(s.split(' '))

    @staticmethod
    def serialize_weights(weights):
        return ' '.join(map(str, weights))

    def serialize(self) -> str:
        return std_json.dumps(
            {'edges': self.edges.tolist(), 'strategies': list(self.strategies), 'weights': list(self.weights)})

    @staticmethod
    def create_graph_with_edges(edges: np.ndarray):
        return Graph(strategies=Strategy.zeros(len(edges)), edges=edges, weights=VertexWeight.same(len(edges)))


class Strategy:
    @staticmethod
    def zeros(n_vertices):
        return np.zeros((n_vertices,)).tolist()

    @staticmethod
    def ones(n_vertices):
        return np.ones((n_vertices,)).tolist()

    @staticmethod
    def partial_ones(n_vertices, prob_one: float):  # prob_one: probability of one to appear
        strategies = np.zeros((n_vertices,))  # 0 is default action.
        options = [0.0, 1.0]
        distribution = [1 - prob_one, prob_one]
        for i in range(n_vertices):
            strategies[i] = random.choices(options, distribution)[0]
        return strategies.tolist()


class VertexWeight:
    @staticmethod
    def same(n_vertices):
        return np.ones((n_vertices,)).tolist()

    @staticmethod
    def random(n_vertices):
        # return (np.floor(np.random.random((n_vertices,)) * 5) % 5 + 1) / 50 + 0.97
        # return (np.floor(np.random.random((n_vertices,)) * 5) % 5 + 1)
        # return np.random.normal(10, 3, (n_vertices,))
        # return np.random.randint(1, n_vertices + 1, size=(n_vertices,))
        return np.random.randint(0, n_vertices, size=(n_vertices,))


class Edge:
    @staticmethod
    def vertex_index(n_vertices, idx):
        if idx < 0:
            return n_vertices + idx
        if idx >= n_vertices:
            return idx - n_vertices
        return idx

    @staticmethod
    def ws_model(n_vertices, k_nearest, rewire_prob) -> np.ndarray:
        # Each vertex connects to its 4 nearest neighbors (based on number)
        # Vertex numbers are from 0 to (n_vertices - 1)
        # Edge Matrix:
        #     - Indices are vertex no.
        #     - 0 means unconnected; 1 means connected.
        #     - Init state is bi-directional.
        # Wiki: https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model
        edges = np.zeros((n_vertices, n_vertices))

        # Connect k nearest neighbors
        if k_nearest % 2 != 0:
            raise NotImplementedError('K-nearest must be even.')
        for vertex in range(n_vertices):
            for i in range(k_nearest):
                if i < k_nearest // 2:
                    target = vertex + (i - k_nearest // 2)
                else:
                    target = vertex + (i - k_nearest // 2 + 1)
                # Handle targets not in 0~n_vertices
                target = Edge.vertex_index(n_vertices, target)
                #
                edges[vertex][target] = 1.0
        # Rewire with a probability
        if k_nearest % 2 != 0:
            raise NotImplementedError('K-nearest must be even.')
        random_options = [True, False]
        random_distribution = [rewire_prob, 1. - rewire_prob]
        for vertex in range(n_vertices):
            for i in range(k_nearest // 2):
                do_rewire = random.choices(random_options, random_distribution)
                if do_rewire[0]:
                    old_target = Edge.vertex_index(n_vertices, vertex + i + 1)
                    # Erase original edge
                    edges[vertex][old_target] = 0.
                    edges[old_target][vertex] = 0.
                    # Rewire
                    available_vertices = []
                    for possible_target in range(n_vertices):
                        if possible_target != vertex and possible_target != old_target and \
                                edges[vertex][possible_target] == 0.:
                            available_vertices.append(possible_target)
                    new_target = available_vertices[np.random.randint(len(available_vertices))]
                    edges[vertex][new_target] = 1.
                    edges[new_target][vertex] = 1.
        return edges


class Utility:
    """Each utility function here must have at least 2 arguments: graph & vertex."""

    @staticmethod
    def maximal_independent_set(graph: Graph, vertex: int, alpha=2.0):
        if graph.strategies[vertex] != 0.0:
            utility = 1.0
            for neighbor in range(graph.n_vertices()):
                if graph.edges[vertex][neighbor] != 0. and vertex != neighbor:
                    utility += -1 * alpha * graph.strategies[vertex] * graph.strategies[neighbor]
        else:
            utility = 0.
        return utility

    @staticmethod
    def weighted_maximal_independent_set_1(graph: Graph, vertex: int, alpha=2.0):
        # Calculate priority
        #     w(v)/((degree(v)+1)
        priority = []
        for v in range(len(graph.edges)):
            degree = sum(graph.edges[v])
            priority.append(graph.weights[v] / (degree + 1))
        #
        if graph.strategies[vertex] != 0.0:
            utility = 1.0
            for neighbor in range(graph.n_vertices()):
                if graph.edges[vertex][neighbor] != 0. and vertex != neighbor and \
                        priority[vertex] <= priority[neighbor]:  # using priority
                    utility += -1 * alpha * graph.strategies[vertex] * graph.strategies[neighbor]
        else:
            utility = 0.
        return utility

    @staticmethod
    def weighted_maximal_independent_set_2(graph: Graph, vertex: int, alpha=2.0):
        # Calculate priority
        #     w(v)/sum of all v's closed neighbors' w
        priority = []
        for v in range(len(graph.edges)):
            tmp_p = graph.weights[v]
            for other_v in range(len(graph.edges)):
                if graph.edges[v][other_v] != 0.0:
                    tmp_p += graph.weights[other_v]
            priority.append(graph.weights[v] / tmp_p)
        #
        if graph.strategies[vertex] != 0.0:
            utility = 1.0
            for neighbor in range(graph.n_vertices()):
                if graph.edges[vertex][neighbor] != 0. and vertex != neighbor and \
                        priority[vertex] <= priority[neighbor]:  # using priority
                    utility += -1 * alpha * graph.strategies[vertex] * graph.strategies[neighbor]
        else:
            utility = 0.
        return utility


class Game:
    """
    In this game, players only have 2 actions:
            1. Turn off (0)
            2. Turn on (1)
    """

    def __init__(self, graph: Graph, utility_func: Callable):
        self.graph = graph
        self.utility_func = utility_func

    @staticmethod
    def check_nash_equilibrium(graph: Graph, utility_func: Callable, alpha=None) -> List[int]:
        better_choice_vertices = []
        for vertex in range(graph.n_vertices()):
            temp = graph.strategies[vertex]
            if alpha is None:
                cur_utility = utility_func(graph=graph, vertex=vertex)
                graph.strategies[vertex] = 1. if temp == 0. else 0.
                new_utility = utility_func(graph=graph, vertex=vertex)
            else:
                cur_utility = utility_func(graph=graph, vertex=vertex, alpha=alpha)
                graph.strategies[vertex] = 1. if temp == 0. else 0.
                new_utility = utility_func(graph=graph, vertex=vertex, alpha=alpha)
            #
            graph.strategies[vertex] = temp
            #
            if new_utility > cur_utility:
                better_choice_vertices.append(vertex)
        return better_choice_vertices

    @staticmethod
    def run_a_time(graph, utility_func, alpha=None) -> Tuple[bool, List, Graph, Optional[int]]:  # True means terminated
        better_choice_vertices = Game.check_nash_equilibrium(graph=graph, utility_func=utility_func, alpha=alpha)
        if len(better_choice_vertices) == 0:
            return True, better_choice_vertices, graph, None  # Reach NE
        # Randomly pick one to change its action to improve its utility
        selected_index = np.random.randint(len(better_choice_vertices))
        selected_vertex = better_choice_vertices[selected_index]
        # Switch from 0/1 to 1/0
        graph.strategies[selected_vertex] = 0. if graph.strategies[selected_vertex] == 1. else 1.
        return False, better_choice_vertices, graph, selected_vertex

    def run(self, show_each_move=False, alpha=None) -> Tuple[Graph, int]:
        graph = deepcopy(self.graph)
        move_cnt = 0
        while True:
            if show_each_move:
                graph.plot_graph(show=True)
            if self.run_a_time(graph=graph, utility_func=self.utility_func, alpha=alpha)[0]:
                break
            move_cnt += 1
        return graph, move_cnt

# times = 100
# avg_moves = []
# avg_cardinality = []
# all_p = np.arange(10.) / 10.
# for rewire_p in all_p:
#     total_moves = 0
#     total_cardinality = 0
#     for _ in range(times):
#         init_edges = Edge.ws_model(n_vertices=30, k_nearest=4, rewire_prob=rewire_p)
#         init_strategies = Strategy.partial_ones(n_vertices=len(init_edges), prob_one=0.5)
#         init_weights = VertexWeight.random(n_vertices=len(init_edges))
#         g = Graph(strategies=init_strategies, edges=init_edges, weights=init_weights)
#         m = Game(graph=g, utility_func=Utility.maximal_independent_set)
#         result_graph, moves = m.run(show_each_move=False)
#         total_moves += moves
#         total_cardinality += sum(result_graph.strategies)
#         g.plot_graph(show=True)
#         exit(1)
#     avg_moves.append(total_moves / times)
#     avg_cardinality.append(total_cardinality / times)
# plt.plot(all_p, avg_moves, label='avg_moves')
# plt.plot(all_p, avg_cardinality, label='avg_cardinality')
# plt.legend()
# plt.show()

# TODO: verify that the game state is a valid solution
