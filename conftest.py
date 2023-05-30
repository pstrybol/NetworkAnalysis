from NetworkAnalysis.Graph import Graph
from NetworkAnalysis.UndirectedInteractionNetwork import UndirectedInteractionNetwork
from NetworkAnalysis.MultiGraph import MultiGraph
import networkx as nx
import pandas as pd
import numpy as np
import pytest
import pickle
import json


@pytest.fixture(scope='module')
def mut_data():
    return pd.DataFrame(np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]]), columns=['G' + str(i) for i in range(3)])


@pytest.fixture(scope='module')
def cna_data():
    return pd.DataFrame(np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0]]),
                            columns=['G' + str(i) for i in range(4)])


@pytest.fixture(scope='module')
def network_chain_4(N_nodes=4):
    edges = [('G' + str(i), 'G' + str(i + 1)) for i in range(N_nodes - 1)]
    network = pd.DataFrame(np.array(edges), columns=['GA', 'GB'])
    return UndirectedInteractionNetwork(network)


# @pytest.fixture(scope='module')
# def network_chain_4_directed(N_nodes=4):
#     edges = [('G' + str(i), 'G' + str(i + 1)) for i in range(N_nodes - 1)]
#     network = pd.DataFrame(np.array(edges), columns=['GA', 'GB'])
#     return DirectedInteractionNetwork(network)


@pytest.fixture(scope='module')
def network_chain_10(N_nodes=10):
    edges = [('G' + str(i), 'G' + str(i + 1)) for i in range(N_nodes - 1)]
    network = pd.DataFrame(np.array(edges), columns=['GA', 'GB'])
    return UndirectedInteractionNetwork(network)


@pytest.fixture(scope='module')
def network_chain_10_Graph(N_nodes=10):
    edges = [('G' + str(i), 'G' + str(i + 1)) for i in range(N_nodes - 1)]
    network = pd.DataFrame(np.array(edges), columns=['GA', 'GB'])
    return Graph(network)


@pytest.fixture(scope='module')
def network_star_4(N_nodes=4):
    edges = [('G' + str(0), 'G' + str(i + 1)) for i in range(N_nodes - 1)]
    network = pd.DataFrame(np.array(edges), columns=['GA', 'GB'])
    return UndirectedInteractionNetwork(network)


# @pytest.fixture(scope='module')
# def network_star_4_directed(N_nodes=4):
#     edges = [('G' + str(0), 'G' + str(i + 1)) for i in range(N_nodes - 1)]
#     network = pd.DataFrame(np.array(edges), columns=['GA', 'GB'])
#     return DirectedInteractionNetwork(network)


@pytest.fixture(scope='module')
def network_circle(N_nodes=4):
    edges = [('G' + str(i), 'G' + str(i + 1)) for i in range(N_nodes - 1)] +\
            [('G' + str(0), 'G' + str(N_nodes - 1))]
    network = pd.DataFrame(np.array(edges), columns=['GA', 'GB'])
    return UndirectedInteractionNetwork(network)


@pytest.fixture(scope='module')
def df_star(N_nodes=4):
    edges = [('G' + str(0), 'G' + str(i + 1)) for i in range(N_nodes - 1)]
    return pd.DataFrame(np.array(edges), columns=['GA', 'GB'])


@pytest.fixture(scope='module')
def adj_dict_star(N_nodes=4):
    adj_dict = {'G' + str(i + 1): np.array(['G0']) for i in range(N_nodes - 1)}
    adj_dict['G' + str(0)] = np.array(['G' + str(i + 1) for i in range(N_nodes - 1)])
    return adj_dict


@pytest.fixture(scope='module')
def karate_club():
    G = nx.karate_club_graph()
    G_obj = UndirectedInteractionNetwork(pd.DataFrame(G.edges, columns=['GeneA', 'GeneB']))
    return G_obj


@pytest.fixture(scope='module')
def karate_club_df():
    G = nx.karate_club_graph()
    return pd.DataFrame(G.edges, columns=['GeneA', 'GeneB'])


@pytest.fixture(scope='module')
def karate_club_Graph():
    G = nx.karate_club_graph()
    return Graph(pd.DataFrame(G.edges, columns=['GeneA', 'GeneB']))


# @pytest.fixture(scope='module')
# def karate_club_dir():
#     G = nx.karate_club_graph()
#     return DirectedInteractionNetwork(pd.DataFrame(G.edges, columns=['GeneA', 'GeneB']))


@pytest.fixture(scope='module')
def karate_net_training_test():
    G = nx.karate_club_graph()
    G_obj = UndirectedInteractionNetwork(pd.DataFrame(G.edges, columns=['GeneA', 'GeneB']))

    X_train, X_test, Y_train, Y_test, _ = G_obj.getTrainTestData(neg_pos_ratio=5,
                                                                 balanced=False,
                                                                 train_ratio=0.8)

    return X_train, X_test, Y_train, Y_test


@pytest.fixture(scope='module')
def disconnected_network(N_comps=3):

    fc_net = UndirectedInteractionNetwork.createFullyConnectedNetwork(['G' + str(i) for i in range(3)])
    j = 3
    for comp in range(N_comps - 1):
        fc_net_ = UndirectedInteractionNetwork.createFullyConnectedNetwork(['G' + str(i) for i in range(j, j+3)])
        j += 3
        fc_net = fc_net.mergeNetworks(fc_net_)

    return fc_net


# @pytest.fixture(scope='module')
# def disconnected_network_directed(N_comps=3):

#     fc_net = DirectedInteractionNetwork.createFullyConnectedNetwork(['G' + str(i) for i in range(3)])
#     j = 3
#     for comp in range(N_comps - 1):
#         fc_net_ = DirectedInteractionNetwork.createFullyConnectedNetwork(['G' + str(i) for i in range(j, j+3)])
#         j += 3
#         fc_net = fc_net.mergeNetworks(fc_net_)

#     return fc_net


@pytest.fixture(scope='module')
def probability_matrix(N_nodes=5):
    np.random.seed(42)
    return np.random.random((N_nodes, N_nodes))


@pytest.fixture(scope='module')
def combined_undirobjects_brain():
    with open("STRING_Brain_dependencies_drugsensitivity", 'rb') as handle:
        combined_undir_objects = pickle.load(handle)
    return combined_undir_objects


@pytest.fixture(scope='module')
def small_multigraph(network_chain_4):
    edges = [('G0', 'G1'), ('G1', 'G3'), ('G2', 'G3'), ('G5', 'G4'), ('G3', 'G4')]
    network = pd.DataFrame(np.array(edges), columns=['GA', 'GB'])
    undir = UndirectedInteractionNetwork(network)

    return MultiGraph(graph_dict={"chain": network_chain_4, "random": undir})


@pytest.fixture(scope='module')
def small_heterogeneous_multigraph(network_chain_4):
    # Dependency
    edges_dep = [('C0', 'G1'),  ('C0', 'G3'), ('C0', 'G5'), ('C1', 'G4'), ('C1', 'G2'), ('C0', 'G2')]
    network_dep = pd.DataFrame(np.array(edges_dep), columns=['GA', 'GB'])

    # Co expression
    edges_coexp = [('G0', 'G1'), ('G0', 'G3'), ('G1', 'G3'), ('G3', 'G4'), ('G4', 'G6')]
    network_coexp = pd.DataFrame(np.array(edges_coexp), columns=['GA', 'GB'])

    node_types = {"G" + str(i): "gene" for i in range(7)}
    node_types["C0"] = "cell line"
    node_types["C1"] = "cell line"

    return MultiGraph(graph_dict={"Interaction": network_chain_4, "dependency": network_dep,
                                  "co_expression": network_coexp},
                      node_types=node_types)


@pytest.fixture(scope='module')
def dis_df_brain():
    return pd.read_csv("STRING_Brain_Cancer.csv", header=0, index_col=0)


@pytest.fixture(scope='module')
def config_brain():
    with open("STRING_Brain_dependencies_drugsensitivity_config.json", 'r') as fp:
        config_file = json.load(fp)
    return config_file


@pytest.fixture(scope='module')
def prob_matrix_karate():
    G = nx.karate_club_graph()
    return pd.DataFrame(np.random.rand(len(G.nodes), len(G.nodes)), 
                        index=[str(i) for i in G.nodes], columns=[str(i) for i in G.nodes])

