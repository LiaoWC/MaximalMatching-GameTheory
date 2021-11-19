import numpy as np
import graphviz
import random
from copy import deepcopy
from typing import Callable, List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import ast
import json as std_json

'''
==== Formal definition ====
P.S. n is number of total vertices. 
     v is vertex.
      
------------------------------------------------------
  Name           Type              Shape      Notes
-----------------------------------------------
  strategies     List[float]       n(1D)      each v's strategy
  edges          ndarray(float)    nxn(2D)    1=connect, 0=disconnect; MUST be bi-directed if it's undirected. 
  weights        List[float]       n(1D)      each v's special value. You can ignore this. You can use it as weight,
                                              or even use it to represent other meaning.
'''


class Graph:
    def __init__(self, strategies: Optional[List[float]], edges: Optional[np.ndarray], weights: Optional[List[float]]):
        """
        :param strategies: List with length n. (n is the number of vertices)
        :param edges: n by n ndarray.
        """
        self.strategies: Optional[List[float]] = strategies
        self.edges: Optional[np.ndarray] = edges
        self.weights: Optional[List[float]] = weights

    def __repr__(self):
        """Show the graph as a string."""
        representation = "\nVertexNo\tEdges\tStrategies\tWeights\n"
        for i in range(len(self.edges) + 1):
            if i > 0:
                for j in range(len(self.edges) + 1):
                    if j == 0:
                        representation += f"{i - 1:03} -> [ "
                    else:
                        representation += str(self.edges[i - 1][j - 1]) + " "
                representation += f"]\ts{self.strategies[i - 1]}\tw{self.weights[i - 1]}\n"
        return representation

    def n_vertices(self) -> int:
        """Total number of vertices."""
        return int(len(self.edges))

    def plot_graph(self, filename='plotted_graph',
                   show=False,
                   picked_vertex: Optional[int] = None,
                   have_better_choice: Optional[List[int]] = None,
                   maximal_matching: bool = False,
                   fill_when_s_not_zero: bool = True):
        """
        :param filename:
        :param show: whether plt show the graph
        :param have_better_choice: highlight those vertices which has other strategy to improve its payoff in this round
                                   (shown as a octagon)
        :param picked_vertex: highlight  the vertex picked
                              (may indicate its the one change its strategy to improve its payoff)
                              (shown as a tripleoctagon)
        """
        d = graphviz.Digraph(filename=filename)
        d.format = 'png'
        d.attr(rankdir='LR', concentrate='True')
        # Vertex index from 1        # Vertices
        d.attr('node', shape='circle')
        for vertex in range(self.n_vertices()):
            color = 'lightblue2' if self.strategies[vertex] != 0. else 'black'
            style = 'filled' if self.strategies[vertex] != 0. else ''
            shape = 'octagon' if (have_better_choice is not None and vertex in have_better_choice) else 'circle'
            shape = 'tripleoctagon' if (picked_vertex is not None and vertex == picked_vertex) else shape
            v_name = f'{vertex}\n(w{self.weights[vertex]})'
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

    def plot_graph_maximal_matching(self, filename='plotted_graph',
                                    show=False,
                                    picked_vertex: Optional[int] = None,
                                    have_better_choice: Optional[List[int]] = None):
        """
        :param filename:
        :param show: whether plt show the graph
        :param have_better_choice: highlight those vertices which has other strategy to improve its payoff in this round
                                   (shown as a octagon)
        :param picked_vertex: highlight  the vertex picked
                              (may indicate its the one change its strategy to improve its payoff)
                              (shown as a tripleoctagon)
        :param fill_when_s_not_zero:
        """
        d = graphviz.Digraph(filename=filename)
        d.format = 'png'
        d.attr(rankdir='LR', concentrate='False')
        # Vertex index from 1        # Vertices
        d.attr('node', shape='circle')

        def v_name(v):
            return f'{v}\n(s{self.strategies[v]})'

        for vertex in range(self.n_vertices()):
            i = vertex
            j = int(self.strategies[i])
            k = int(self.strategies[j])
            if k == i and i != j:
                style = 'filled'
                color = 'lightblue2'
            else:
                style = ''
                color = 'black'
            shape = 'octagon' if (have_better_choice is not None and vertex in have_better_choice) else 'circle'
            shape = 'tripleoctagon' if (picked_vertex is not None and vertex == picked_vertex) else shape

            d.node(v_name(vertex), color=color, style=style, shape=shape)
        for i in range(self.n_vertices()):
            for j in range(self.n_vertices()):
                if self.edges[i][j] != 0. and i != j:
                    c_i = int(self.strategies[i])
                    c_j = int(self.strategies[j])
                    if c_i == j and c_j == i:
                        d.edge(v_name(i), v_name(j), color='red',
                               style='tapered', penwidth='7', arrowhead='none', arrowtail='none', dir='none')
                    elif c_i == j:
                        d.edge(v_name(i), v_name(j), color='orange',
                               style='tapered', penwidth='7', arrowhead='normal', arrowtail='none', dir='both')
                    else:
                        d.edge(v_name(i), v_name(j))
        d.render(view=False, )
        if show is True:
            plt.imshow(plt.imread(f'{filename}.png'))
            plt.axis('off')
            plt.show()

    def strategy_1_weight_sum(self) -> float:
        """Sum of the weight of all vertices which pick strategy 1"""
        s_w_sum = 0.0
        for v, s in enumerate(self.strategies):
            if s != 0.0:
                s_w_sum += self.weights[v]
        return s_w_sum

    def deserialize(self, codes: str) -> None:
        """Fetch strategies, edges, and weights from a string."""
        contents = ast.literal_eval(codes)
        self.strategies = contents['strategies']
        self.edges = np.array(contents['edges'])
        self.weights = contents['weights']

    @staticmethod
    def deserialize_edges(s: str) -> np.ndarray:
        return np.array([line.split(' ') for line in s.split('\n')])

    @staticmethod
    def serialize_edges(edges: np.ndarray) -> str:
        s = ''
        for row in edges:
            for element in row:
                s += element + ' '
        return s.strip()

    @staticmethod
    def deserialize_strategies(s: str) -> List[float]:
        return list(map(float, s.split(' ')))

    @staticmethod
    def serialize_strategies(strategies: List[float]) -> str:
        return ' '.join(map(str, strategies))

    @staticmethod
    def deserialize_weights(s: str) -> List[float]:
        return list(map(float, s.split(' ')))

    @staticmethod
    def serialize_weights(weights: List[float]) -> str:
        return ' '.join(map(str, weights))

    def serialize(self) -> str:
        return std_json.dumps(
            {'edges': self.edges.tolist(), 'strategies': list(self.strategies),
             'weights': list(map(float, self.weights))})

    @staticmethod
    def create_graph_with_edges(edges: np.ndarray) -> "Graph":
        """Create a default graph only given edges. Strategies -> all zero. Weights -> all 1."""
        return Graph(strategies=Strategy.zeros(len(edges)), edges=edges, weights=VertexWeight.same(len(edges)))

    def open_neighbors(self, v: float) -> List[int]:
        return self.edges[v].nonzero()[0].tolist()


class Strategy:
    @staticmethod
    def arange(n_vertices) -> List[float]:
        return np.arange(float(n_vertices)).tolist()

    @staticmethod
    def minus_ones(n_vertices) -> List[float]:
        return (-1 * np.ones((n_vertices,))).tolist()

    @staticmethod
    def zeros(n_vertices) -> List[float]:
        return np.zeros((n_vertices,)).tolist()

    @staticmethod
    def ones(n_vertices) -> List[float]:
        return np.ones((n_vertices,)).tolist()

    @staticmethod
    def partial_ones(n_vertices, prob_one: float) -> List[float]:  # prob_one: probability of one to appear
        """Each vertex has a chance to select strategy one, otherwise zero."""
        strategies = np.zeros((n_vertices,))  # 0 is default action.
        options = [0.0, 1.0]
        distribution = [1 - prob_one, prob_one]
        for i in range(n_vertices):
            strategies[i] = random.choices(options, distribution)[0]
        return strategies.tolist()


class VertexWeight:

    @staticmethod
    def same(n_vertices) -> List[float]:
        return np.ones((n_vertices,)).tolist()

    @staticmethod
    def ones(n_vertices) -> List[float]:
        return VertexWeight.same(n_vertices)

    @staticmethod
    def random(n_vertices) -> List[float]:
        """Permutation all weights and distribute them."""
        # Other type you may try:
        # return (np.floor(np.random.random((n_vertices,)) * 5) % 5 + 1) / 50 + 0.97
        # return (np.floor(np.random.random((n_vertices,)) * 5) % 5 + 1)
        # return np.random.normal(10, 3, (n_vertices,))
        # return np.random.randint(1, n_vertices + 1, size=(n_vertices,))
        # return np.random.randint(0, n_vertices, size=(n_vertices,))
        pool = np.arange(n_vertices)
        np.random.shuffle(pool)
        return pool.tolist()


class Edge:
    @staticmethod
    def vertex_index(n_vertices: int, idx: int) -> int:
        """Adjust idx if it is not in the bound."""
        if idx < 0:
            return n_vertices + idx
        if idx >= n_vertices:
            return idx - n_vertices
        return idx

    @staticmethod
    def ws_model(n_vertices, k_nearest, rewire_prob) -> np.ndarray:
        """
        Each vertex connects to its k_nearest neighbors.
        K here only support even numbers.
        Half of k will try rewiring to other vertices.
        Vertex numbers are from 0 to (n_vertices - 1)
        Edge Matrix:
            - Indices are vertex no.
            - 0 means unconnected; 1 means connected.
            - Init state is bi-directional.
        Wiki: https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model
        """
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
    MIN = -99999999  # effective smallest number (may be chosen)
    MAX = 99999999  # effective largest number (may be chosen)
    ZERO = 0
    GOOD = 999
    """
    Each utility function here must have 3 arguments: graph(Graph) & vertex(int) & strategy(float) & opt(dict).
    Must check if the strategy is valid; if invalid, return NEG_INF.
    """

    @staticmethod
    def maximal_matching(graph: Graph, vertex: int, strategy: float, opt: dict) -> float:
        """Unweighted MIS game.
        If strategy == vertex, it means this v chooses not to connect anyone.
        :param graph:
        :param vertex: the vertex to calculate utility
        :param strategy:
        :param opt:

        Let c() indicate a vertex's strategy
        Let i = vertex to discuss
        Let j = i's strategy = c(i)
        Let k = j's strategy = c(j)
        if i == j: return 0
        elif k == i: return +999999 (i forms a matching)
        elif k == j: return -999999 (i connects a v which already has a matching)
        else: return 999 (encourage v to find a matching instead of be silent)
        """
        strategy = int(strategy)
        i = int(vertex)
        j = int(strategy)
        k = int(graph.strategies[j])
        c_k = int(graph.strategies[k])
        if i == j:
            return Utility.ZERO
        if graph.edges[vertex][strategy] == 0.:  # Not itself or open neighbor
            return Utility.MIN
        elif k == i:
            return Utility.MAX
        elif c_k == j and j != k:
            return Utility.MIN
        else:
            return Utility.GOOD


class PossibleStrategies:
    @staticmethod
    def all_vertices(graph: Graph) -> List[float]:
        return np.arange(graph.n_vertices()).tolist()


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
    def check_nash_equilibrium(graph: Graph, utility_func: Callable,
                               strategy_list: List, opt: dict) -> Dict[int, Tuple[int, float]]:
        """Check what vertices can improve its utility function by changing its original strategy.
        :param graph:
        :param utility_func:
        :param strategy_list: List of possible strategy float numbers
        :param opt: settings for utility function
        return the idx of those vertices which can improve its utility THE MOST and > original utility.
        """
        better_choices = {}  # dict key: v's idx, value: tuple of best other strategies and utility to v
        for vertex in range(graph.n_vertices()):  # iter all vertices
            original_s = graph.strategies[vertex]
            original_u = utility_func(graph=graph, vertex=vertex, strategy=original_s, opt=opt)
            max_u_s = original_s
            max_u = original_u
            for s in strategy_list:
                new_u = utility_func(graph=graph, vertex=vertex, strategy=s, opt=opt)
                if new_u > max_u:
                    max_u_s, max_u = s, new_u
            print('kkk', original_s, original_u, max_u_s, max_u)
            if max_u_s != original_s:
                better_choices[vertex] = (max_u_s, max_u)
        return better_choices

    @staticmethod
    def run_a_time(graph: Graph,
                   utility_func: Callable,
                   strategy_list: List[float],
                   opt: dict) -> Tuple[bool, List[int], Graph, Optional[int], Dict[int, Tuple[int, float]]]:
        """
        Returning True means terminated.
        """
        better_choices = Game.check_nash_equilibrium(graph=graph, utility_func=utility_func,
                                                     strategy_list=strategy_list, opt=opt)

        have_better_choice = [key for key in better_choices]
        if len(better_choices) == 0:
            return True, have_better_choice, graph, None, better_choices  # Reach NE
        # Randomly pick one to change its action to improve its utility
        candidates = [key for key in better_choices]
        selected_vertex: int = candidates[np.random.randint(len(candidates))]
        new_s, _ = better_choices[selected_vertex]
        # Switch strategy
        graph.strategies[selected_vertex] = new_s
        return False, have_better_choice, graph, selected_vertex, better_choices

    def run(self, show_each_move=False, alpha=None) -> Tuple[Graph, int]:
        # TODO: need to revise
        graph = deepcopy(self.graph)
        move_cnt = 0
        while True:
            if show_each_move:
                graph.plot_graph(show=True)
            if self.run_a_time(graph=graph, utility_func=self.utility_func, alpha=alpha)[0]:
                break
            move_cnt += 1
        return graph, move_cnt

    @staticmethod
    def check_real_independent_set(graph: Graph):
        # TODO: make a test function to check maximal matching
        # Method: test each vertex
        #         Turn a vertex with strategy 0 to 1, and see if it doesn't connect any vertices with strategy 1.
        #         If so, the new graph is an independent set so the original graph is not a "maximal" independent set.
        n_vertices = len(graph.edges)
        for v in range(n_vertices):
            if graph.strategies[v] != 0.:
                continue
            no_neighbor_with_s1 = True
            # See if a strategy-0 vertex's all neighbors are with strategy-0.
            for other_v in range(n_vertices):
                if other_v == v:
                    continue
                if graph.edges[v][other_v] == 1. and graph.strategies[other_v] == 1.:
                    no_neighbor_with_s1 = False
                    break
            if no_neighbor_with_s1 is True:
                return False  # TODO: check again
        return True


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

init_edges = Edge.ws_model(n_vertices=10, k_nearest=2, rewire_prob=0.4)
init_strategies = Strategy.arange(n_vertices=len(init_edges))
init_weights = VertexWeight.same(n_vertices=len(init_edges))
g = Graph(strategies=init_strategies, edges=init_edges, weights=init_weights)
m = Game(graph=g, utility_func=Utility.maximal_matching)

is_terminated, have_better_choices, g, selected_vx, better_choicess = Game.run_a_time(graph=g,
                                                                                      utility_func=Utility.maximal_matching,
                                                                                      strategy_list=PossibleStrategies.all_vertices(
                                                                                          g),
                                                                                      opt={})
print(is_terminated, have_better_choices, selected_vx, better_choicess)
g.plot_graph_maximal_matching(show=True, picked_vertex=selected_vx, have_better_choice=have_better_choices)

# TODO: verify that the game state is a valid solution
