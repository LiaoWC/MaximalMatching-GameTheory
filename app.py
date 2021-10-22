from flask import Flask, render_template, request, json
from game.game import Graph, Strategy, VertexWeight, Edge, Game, Utility
import numpy as np
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.use('Agg')

app = Flask(__name__)


# Home page
@app.route('/', methods=["GET"])
def home():
    return render_template("home.html")


def get_data_and_codes() -> [dict, str]:
    received = json.loads(request.data)
    return received['data'], received['codes'].strip()


@app.route('/submit_ws_graph', methods=['POST'])
def submit_ws_graph():
    data, codes = get_data_and_codes()
    print('data', data)
    # Make edge matrix by data
    try:
        n = int(data['n'])
        k = int(data['k'])
        p = float(data['p'])
        edges = Edge.ws_model(n_vertices=n, k_nearest=k, rewire_prob=p)
        strategies = Strategy.partial_ones(n_vertices=n, prob_one=0.5) if \
            data['if_rand_strategies'] is True else Strategy.zeros(n_vertices=n)
        weights = VertexWeight.random(n_vertices=n) if data['if_rand_weights'] else \
            VertexWeight.same(n_vertices=n)
    except ValueError:
        return {'state': 'failed'}
    #
    try:
        shutil.rmtree('static/img/tmp')
    except FileNotFoundError:
        pass
    graph = Graph(edges=edges, strategies=strategies, weights=weights)
    img_rand_code = np.random.randint(100000, 999999)
    graph.plot_graph(filename=f'static/img/tmp/{img_rand_code}', show=False)
    #
    return {'state': 'successful', 'codes': graph.serialize(), 'img': img_rand_code,
            'weight_sum': graph.strategy_1_weight_sum()}


@app.route('/submit_custom_graph', methods=['POST'])
def submit_custom_graph():
    data, codes = get_data_and_codes()
    print('data', data)
    # Make edge matrix by data
    try:
        edges_input: str = data['edges']
        strategies_input: str = data['strategies']
        weights_input: str = data['weights']
        #
        n_vertices = len(edges_input.strip().split('\n'))
        edges = [line.strip().split(' ') for line in edges_input.strip().split('\n')]
        new_edges = []
        for row in edges:
            if len(row) != n_vertices:  # Check if all row has same size
                raise ValueError()
            new_row = [float(element) for element in row]
            new_edges.append(new_row)
        edges = np.array(new_edges)
        strategies = list(map(float, strategies_input.strip().split(' ')))
        weights = list(map(float, weights_input.strip().split(' ')))
        if len(strategies) != len(weights) or len(strategies) != len(edges):
            raise ValueError()
    except ValueError:
        return {'state': 'failed'}
    #
    try:
        shutil.rmtree('static/img/tmp')
    except FileNotFoundError:
        pass
    graph = Graph(edges=edges, strategies=strategies, weights=weights)
    img_rand_code = np.random.randint(100000, 999999)
    graph.plot_graph(filename=f'static/img/tmp/{img_rand_code}', show=False)
    #
    return {'state': 'successful', 'codes': graph.serialize(), 'img': img_rand_code,
            'weight_sum': graph.strategy_1_weight_sum()}


@app.route('/run_a_move', methods=['POST'])
def run_a_move():
    data, codes = get_data_and_codes()
    try:
        alpha = float(data['alpha'])
        selected_u = int(data['selected_u'])
        print(alpha, selected_u)
        if selected_u == 0:
            u_func = Utility.maximal_independent_set
        elif selected_u == 1:
            u_func = Utility.weighted_maximal_independent_set_1
        elif selected_u == 2:
            u_func = Utility.weighted_maximal_independent_set_2
        else:
            raise ValueError
    except ValueError:
        return {'state': 'failed'}
    graph = Graph(strategies=None, edges=None, weights=None)
    graph.deserialize(codes=codes)
    # Run a move
    if_reach_ne, better_choice_vertices, graph, selected = \
        Game.run_a_time(graph=graph, utility_func=u_func, alpha=alpha)
    # Output
    try:
        shutil.rmtree('static/img/tmp')
    except FileNotFoundError:
        pass
    img_rand_code = np.random.randint(100000, 999999)
    graph.plot_graph(filename=f'static/img/tmp/{img_rand_code}', show=False,
                     picked_vertex=selected, have_better_choice=better_choice_vertices)
    #
    return {'state': 'successful', 'codes': graph.serialize(), 'img': img_rand_code,
            'weight_sum': graph.strategy_1_weight_sum()}


@app.route('/run_games', methods=['POST'])
def run_games():
    data, codes = get_data_and_codes()
    # Clear previous output
    try:
        shutil.rmtree('static/img/tmp')
    except FileNotFoundError:
        pass
    # Run games
    try:
        n = int(data['n'])
        k = int(data['k'])
        n_games = int(data['n_games'])
        img_rand_code = np.random.randint(100000, 999999)

        all_avg_moves = []
        all_avg_cardinality = []
        all_avg_s1_weight_sum = []
        for u in [Utility.maximal_independent_set, Utility.weighted_maximal_independent_set_1,
                  Utility.weighted_maximal_independent_set_2]:
            avg_moves = []
            avg_cardinality = []
            avg_s1_weight_sum = []
            all_p = np.arange(6.) / 5.
            for rewire_p in all_p:
                total_moves = 0
                total_cardinality = 0
                total_s1_weight_sum = 0.
                for i in range(n_games):
                    init_edges = Edge.ws_model(n_vertices=n, k_nearest=k, rewire_prob=rewire_p)
                    init_strategies = Strategy.partial_ones(n_vertices=n, prob_one=0.5)
                    init_v_weights = VertexWeight.random(n_vertices=n)
                    g = Graph(strategies=init_strategies, edges=init_edges, weights=init_v_weights)
                    m = Game(graph=g, utility_func=u)
                    result_graph, moves = m.run(show_each_move=False, alpha=2.0)
                    total_moves += moves
                    total_cardinality += sum(result_graph.strategies)
                    total_s1_weight_sum += result_graph.strategy_1_weight_sum()
                avg_moves.append(total_moves / n_games)
                avg_cardinality.append(total_cardinality / n_games)
                avg_s1_weight_sum.append(total_s1_weight_sum / n_games)
                print(f'--- u:{u.__name__} rewire_p:{rewire_p}')
            all_avg_moves.append(avg_moves)
            all_avg_cardinality.append(avg_cardinality)
            all_avg_s1_weight_sum.append(avg_s1_weight_sum)
        print('all_avg_move:', all_avg_moves)
        print('all_avg_cardinality:', all_avg_cardinality)
        print('all_avg_s1_weight_sum:', all_avg_s1_weight_sum)

        save_filename = f'static/img/tmp/{img_rand_code}.png'
        Path(save_filename).parent.mkdir(parents=True, exist_ok=True)
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(21, 5))
        x = np.arange(6.) / 5.
        # Plot avg_cardinality
        ax0.plot(x, all_avg_cardinality[0], label='Non-weighted')
        ax0.plot(x, all_avg_cardinality[1], label='Weighted (u1)')
        ax0.plot(x, all_avg_cardinality[2], label='Weighted (u2)')
        ax0.set_title(f'Avg. Size (n={n},k={k}),#games={n_games}')
        ax0.set_xlabel('Rewiring probability')
        ax0.set_ylabel('Avg. size of picking strategy 1')
        ax0.legend()
        # Plot avg_s1_weight_sum
        ax1.plot(x, all_avg_s1_weight_sum[0], label='Non-weighted')
        ax1.plot(x, all_avg_s1_weight_sum[1], label='Weighted (u1)')
        ax1.plot(x, all_avg_s1_weight_sum[2], label='Weighted (u2)')
        ax1.set_title(f'Avg. Strategy 1 Weight Sum (n={n},k={k},#games={n_games})')
        ax1.set_xlabel('Rewiring probability')
        ax1.set_ylabel('Avg. strategy 1 weight sum')
        ax1.legend()
        # Plot avg_moves
        ax2.plot(x, all_avg_moves[0], label='Non-weighted')
        ax2.plot(x, all_avg_moves[1], label='Weighted (u1)')
        ax2.plot(x, all_avg_moves[2], label='Weighted (u2)')
        ax2.set_title(f'Avg. Moves (n={n},k={k},#games={n_games})')
        ax2.set_xlabel('Rewiring probability')
        ax2.set_ylabel('Avg. moves')
        ax2.legend()
        fig.savefig(save_filename)
        plt.close()

    except ValueError:
        return {'state': 'failed'}
    #
    graph = Graph(strategies=None, edges=None, weights=None)
    graph.deserialize(codes=codes)
    return {'state': 'successful', 'codes': graph.serialize(), 'img': img_rand_code,
            'weight_sum': graph.strategy_1_weight_sum()}


if __name__ == '__main__':
    app.run(debug=True)

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
#         init_v_weights = VertexWeight.random(n_vertices=len(init_edges))
#         g = Graph(strategies=init_strategies, edges=init_edges, v_weights=init_v_weights)
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
