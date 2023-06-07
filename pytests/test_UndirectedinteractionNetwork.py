from NetworkAnalysis.Graph import perform_single_LP_swap, filter_graph_by_LP_swaps, _perform_single_LP_swap_fastest, \
    filter_graph_by_LP_swaps_fastest, adj_dict_to_df
from NetworkAnalysis.UndirectedInteractionNetwork import UndirectedInteractionNetwork
import pandas as pd
import numpy as np
import time


def test_adj_dict_to_df(adj_dict_star):
    df = adj_dict_to_df(adj_dict_star)

    outcome = set(zip(df['Gene_A'].values, df['Gene_B'].values))
    expected = set([(k, v) for k, vals in adj_dict_star.items() for v in vals])

    assert expected == outcome


def test_degreePreservingPermutation(network_chain_10, karate_club):

    start = time.time()
    perms = karate_club.deepcopy().degreePreservingPermutation(N_swaps=50)
    stop = time.time()

    print('Elapsed time: %f' % (stop - start))
    overlap = perms.getOverlap(karate_club)
    print(overlap)

    outcome = perms.getDegreeDF().set_index('Gene')['Count']
    expected = karate_club.getDegreeDF().set_index('Gene')['Count']
    genes = expected.index.values

    assert karate_club.interactions.shape[0] == perms.interactions.shape[0]
    assert np.all(expected.loc[genes].values == outcome.loc[genes].values)

    perms = network_chain_10.degreePreservingPermutation(N_swaps=1)
    overlap = network_chain_10.getOverlap(perms)

    outcome = perms.getDegreeDF().set_index('Gene')['Count']
    expected = network_chain_10.getDegreeDF().set_index('Gene')['Count']
    genes = expected.index.values

    assert np.all(expected.loc[genes].values == outcome.loc[genes].values)
    assert True


def test_replaceNodesWithInteractions(network_chain_4, network_star_4):
    tbr_nodes = np.array(['G1'])
    net = network_star_4.replaceNodesWithInteractions(tbr_nodes)
    outcome = set([tuple(l) for l in net.getInteractionNamed().values.tolist()])
    expected = {('G0', 'G3'), ('G0', 'G2')}
    assert expected == outcome

    net = network_chain_4.replaceNodesWithInteractions(tbr_nodes)
    outcome = set([tuple(l) for l in net.getInteractionNamed().values.tolist()])
    expected = {('G0', 'G2'), ('G2', 'G3')}
    assert expected == outcome

    tbr_nodes = np.array(['G0'])
    net = network_star_4.replaceNodesWithInteractions(tbr_nodes)
    outcome = set([tuple(sorted(l)) for l in net.getInteractionNamed().values.tolist()])
    expected = {('G1', 'G3'), ('G2', 'G3'), ('G1', 'G2')}

    assert expected == outcome


def test_getInteractionNamed(network_chain_4):

    df = network_chain_4.getInteractionNamed()
    df2 = network_chain_4.getInteractionNamed(return_both_directions=True)

    assert 2 * df.shape[0] == df2.shape[0]
    assert 2 * len(set(zip(df.Gene_A, df.Gene_B))) == len(set(zip(df2.Gene_A, df2.Gene_B)))


def test_createFullyConnectedNetwork():

    node_names = ['G1', 'G2', 'G3']
    fc_net = UndirectedInteractionNetwork.createFullyConnectedNetwork(node_names)

    assert fc_net.interactions.shape[0] == 3
    assert set(list(fc_net.node_names)) == set(node_names)


def test_mergeNetworks():
    #TODO improve asserts
    node_names = ['G1', 'G2', 'G3']
    fc_net = UndirectedInteractionNetwork.createFullyConnectedNetwork(node_names)

    node_names2 = ['G1', 'G4', 'G5']

    fc_net2 = UndirectedInteractionNetwork.createFullyConnectedNetwork(node_names2)

    total_net = fc_net.mergeNetworks(fc_net2)

    print(total_net)

    assert set(total_net.node_names) == set(node_names + node_names2)
    assert total_net.interactions.shape[0] == 6


def test_isConnected(disconnected_network):
    node_names = ['G1', 'G2', 'G3']
    fc_net = UndirectedInteractionNetwork.createFullyConnectedNetwork(node_names)

    assert fc_net.isConnected

    print(disconnected_network)
    print(disconnected_network.isConnected)
    assert not disconnected_network.isConnected


def test_getComponents(disconnected_network):
    comps = disconnected_network.getComponents(return_subgraphs=True)

    assert len(comps) == 3
    assert max(comps, key=len).interactions.shape[0] == 3

    comps_df = disconnected_network.getComponents(return_subgraphs=False)

    assert len(np.unique(comps_df.Component.values)) == 3
    assert comps_df.shape[0] == 9

    fc_net = UndirectedInteractionNetwork.createFullyConnectedNetwork(['G1', 'G2', 'G3'])

    comps = fc_net.getComponents(return_subgraphs=True)

    assert len(comps) == 1
    assert max(comps, key=len).interactions.shape[0] == 3


def test_keepLargestComponent(disconnected_network):

    disconnected_network_df = disconnected_network.getInteractionNamed()
    disconnected_network_largest_cc = UndirectedInteractionNetwork(disconnected_network_df, keeplargestcomponent=True)

    largest_comp = disconnected_network.keepLargestComponent()

    assert largest_comp.N_nodes == 3
    assert largest_comp.interactions.shape[0] == 3

    disconnected_network.keepLargestComponent(inplace=True)

    assert disconnected_network.N_nodes == 3
    assert disconnected_network.interactions.shape[0] == 3


def test_getAdjDict(network_chain_10, network_chain_4):
    print(network_chain_10.getAdjDict(return_names=True))

    print(network_chain_4.getAdjDict(return_names=True))


def test_getAdjMatrix():

    #TODO: insert other test scenarios
    fc_net = UndirectedInteractionNetwork.createFullyConnectedNetwork(['G1', 'G2', 'G3'])

    A, node_names = fc_net.getAdjMatrix(as_df=False)
    print(A)
    print(A[np.triu_indices(3, k=1)])
    assert np.all(A[np.triu_indices(3, k=1)] == 1)
    assert set(list(node_names)) == {'G1', 'G2', 'G3'}


def test_perform_single_LP_swap(probability_matrix):
    print(probability_matrix)

    curr_rows = np.array([1, 1, 2, 3, 3, 4, 5, 5, 5]) - 1
    curr_cols = np.array([3, 4, 5, 1, 5, 1, 5, 2, 3]) - 1

    swapped, (new_rows, new_cols), _ = perform_single_LP_swap(probability_matrix, curr_rows, curr_cols,
                                                             r_i=0, c_i=2)

    if not swapped:
        print('No swap performed')
        assert np.all(curr_rows == new_rows)
        assert np.all(curr_cols == new_cols)

    else:
        row_mask, col_mask = new_rows != curr_rows, new_cols != curr_cols
        new_edge = (new_rows[row_mask], new_cols[col_mask])
        old_edge = (curr_rows[row_mask], curr_cols[col_mask])

        print('old edge: %s' % (old_edge, ))
        print('new edge: %s' % (new_edge, ))

        diff = probability_matrix[(new_rows, new_cols)] - probability_matrix[(curr_rows, curr_cols)]
        assert np.sum(diff) > 0
        assert len(curr_rows) == len(new_rows)
        assert len(curr_cols) == len(new_cols)

    swapped, new_ids, _ = perform_single_LP_swap(probability_matrix, curr_rows, curr_cols,
                                              r_i=0, c_i=3)
    new_rows, new_cols = new_ids

    print(new_ids)

    if not swapped:
        print('No swap performed')
        assert np.all(curr_rows == new_rows)
        assert np.all(curr_cols == new_cols)

    else:
        row_mask, col_mask = new_rows != curr_rows, new_cols != curr_cols
        new_edge = (new_rows[row_mask], new_cols[col_mask])
        old_edge = (curr_rows[row_mask], curr_cols[col_mask])

        print('old edge: %s' % (old_edge, ))
        print('new edge: %s' % (new_edge, ))

        diff = probability_matrix[(new_rows, new_cols)] - probability_matrix[(curr_rows, curr_cols)]
        assert np.sum(diff) > 0
        assert len(curr_rows) == len(new_rows)
        assert len(curr_cols) == len(new_cols)
        print('Observed difference: %f' % np.sum(diff))

    curr_rows = np.array([2, 1, 2, 3, 3, 4, 5, 5, 5]) - 1
    curr_cols = np.array([3, 4, 5, 2, 5, 1, 5, 2, 3]) - 1

    swapped, new_ids, _ = perform_single_LP_swap(probability_matrix, curr_rows, curr_cols,
                                              r_i=2, c_i=1)
    new_rows, new_cols = new_ids

    assert not swapped
    print('No swap performed')
    assert np.all(curr_rows == new_rows)
    assert np.all(curr_cols == new_cols)


def test_perform_single_LP_swap_fastest(probability_matrix):
    print(probability_matrix)

    curr_rows = np.array([1, 1, 2, 3, 3, 4, 5, 5, 5]) - 1
    curr_cols = np.array([3, 4, 5, 1, 5, 1, 5, 2, 3]) - 1

    A = np.zeros(probability_matrix.shape)
    A[curr_rows, curr_cols] = 1

    swapped, old_ints, curr_ints, edge_id = _perform_single_LP_swap_fastest(probability_matrix,
                                                                                       A,
                                                                                       (curr_rows, curr_cols),
                                                                                       r_i=0,
                                                                                       c_i=2)

    if not swapped:
        print('No swap performed')
        assert old_ints is None
        assert edge_id is None

    else:

        diff = probability_matrix[curr_ints] - probability_matrix[old_ints]
        assert np.sum(diff) > 0
        assert len(curr_ints[0]) == len(old_ints[0])
        assert len(curr_ints[1]) == len(old_ints[1])

    swapped, old_ints, new_ids, edge_id = _perform_single_LP_swap_fastest(probability_matrix,
                                                                          A,
                                                                          (curr_rows, curr_cols),
                                                                          r_i=0,
                                                                          c_i=2)

    if not swapped:
        print('No swap performed')
        assert old_ints is None
        assert edge_id is None

    else:

        diff = probability_matrix[curr_ints] - probability_matrix[old_ints]
        assert np.sum(diff) > 0
        assert len(curr_ints[0]) == len(old_ints[0])
        assert len(curr_ints[1]) == len(old_ints[1])

    curr_rows = np.array([2, 1, 2, 3, 3, 4, 5, 5, 5]) - 1
    curr_cols = np.array([3, 4, 5, 2, 5, 1, 5, 2, 3]) - 1

    A = np.zeros(probability_matrix.shape)
    A[curr_rows, curr_cols] = 1
    bool = False

    try:
        swapped, old_ints, new_ids, edge_id = _perform_single_LP_swap_fastest(probability_matrix,
                                                                              A,
                                                                              (curr_rows, curr_cols),
                                                                              r_i=0,
                                                                              c_i=2)

    except AssertionError:
        bool = True

    assert bool

    node_names = ['North', 'East', 'South', 'West', 'Center']
    net_df = pd.DataFrame({'Gene_A': ['North', 'East', 'South', 'West', 'Center', 'Center', 'Center', 'Center'],
                           'Gene_B': ['South', 'West', 'East', 'North', 'South', 'West', 'East', 'North']})

    net = UndirectedInteractionNetwork(net_df)

    prob_mat = np.array([[1, 0.9, 0.2, 0.9, 0.8],
                         [0.9, 1, 0.9, 0.2, 0.8],
                         [0.2, 0.9, 1, 0.9, 0.8],
                         [0.9, 0.2, 0.9, 1, 0.8],
                         [0.8, 0.8, 0.8, 0.8, 1]])

    A = net.getAdjMatrix(as_df=True)
    A = A[node_names].loc[node_names].values
    curr_ints = np.where(A)

    mask = curr_ints[0] < curr_ints[1]

    curr_ints = (curr_ints[0][mask], curr_ints[1][mask])

    print('Starting on the N-S network.')
    swapped, old_ints, new_ids, edge_id = _perform_single_LP_swap_fastest(prob_mat,
                                                                          A,
                                                                          curr_ints,
                                                                          r_i=0,
                                                                          c_i=2)

    diff = np.abs(np.sum(prob_mat[old_ints] - prob_mat[new_ids]))

    print(diff)
    assert (diff - 1.4) < 1e-5


def test_interactions_as_set(karate_club):
    int_set = karate_club.interactions_as_set()

    assert len(int_set) == karate_club.interactions.shape[0]


def test_eq(karate_club):
    karate_club2 = karate_club.deepcopy()

    assert karate_club == karate_club2


def test_filter_graph_by_LP_swaps(probability_matrix):
    visualize = False

    curr_cols = np.array([1, 1, 2, 3, 3, 4, 5, 5]) - 1
    curr_rows = np.array([3, 4, 5, 5, 1, 1, 2, 3]) - 1
    new_rows, new_cols = filter_graph_by_LP_swaps(probability_matrix,
                                                  (curr_rows, curr_cols),
                                                  n_swaps=2,
                                                  max_attempts=10)

    diff = probability_matrix[(new_rows, new_cols)] - probability_matrix[(curr_rows, curr_cols)]
    assert np.sum(diff) > 0
    #errs because of duplicate edges

    if visualize:
        df1 = pd.DataFrame(np.transpose(np.array((curr_rows, curr_cols))),
                           columns=['Gene_A', 'Gene_B'])
        print(df1)
        UndirectedInteractionNetwork(df1).visualize(show_labels=True)

        df2 = pd.DataFrame(np.transpose(np.array((new_rows, new_cols))),
                           columns=['Gene_A', 'Gene_B'])
        print(df2)
        UndirectedInteractionNetwork(df2).visualize(show_labels=True)


def test_getTrainTestData(karate_club):

    # Unbalanced
    # Without validation set
    X_train, X_test, Y_train, Y_test = karate_club.getTrainTestData(train_ratio=0.7,
                                                                     neg_pos_ratio=5, 
                                                                     return_summary=False,
                                                                     random_state=23,
                                                                     balanced=False)
    
    assert not set(karate_club.nodes) - set(np.unique(X_train[np.where(Y_train == 1)].ravel()))
    assert not set([tuple(i) for i in X_train]) & set([tuple(i) for i in X_test])
    del X_train, X_test, Y_train, Y_test

    # With validation set
    X_train, X_val, X_test, Y_train, Y_val, Y_test = karate_club.getTrainTestData(train_ratio=0.7,
                                                                                   train_validation_ratio=0.7,
                                                                                   neg_pos_ratio=5, 
                                                                                   return_summary=False,
                                                                                   random_state=23,
                                                                                   balanced=False)
    assert not set([tuple(i) for i in X_train]) & set([tuple(i) for i in X_test])
    assert not set([tuple(i) for i in X_train]) & set([tuple(i) for i in X_val])
    assert not set([tuple(i) for i in X_test]) & set([tuple(i) for i in X_val])

    # Balanced
    # Without validation set
    X_train, X_test, Y_train, Y_test = karate_club.getTrainTestData(train_ratio=0.7,
                                                                     neg_pos_ratio=5, 
                                                                     return_summary=False,
                                                                     random_state=23,
                                                                     balanced=True)
    
    assert not set(karate_club.nodes) - set(np.unique(X_train[np.where(Y_train == 1)].ravel()))
    assert not set([tuple(i) for i in X_train]) & set([tuple(i) for i in X_test])
    del X_train, X_test, Y_train, Y_test

    # With validation set
    X_train, X_val, X_test, Y_train, Y_val, Y_test = karate_club.getTrainTestData(train_ratio=0.7,
                                                                                   train_validation_ratio=0.7,
                                                                                   neg_pos_ratio=5, 
                                                                                   return_summary=False,
                                                                                   random_state=23,
                                                                                   balanced=True)
    assert not set([tuple(i) for i in X_train]) & set([tuple(i) for i in X_test])
    assert not set([tuple(i) for i in X_train]) & set([tuple(i) for i in X_val])
    assert not set([tuple(i) for i in X_test]) & set([tuple(i) for i in X_val])


# def test_filter_graph_by_LP(karate_club, prob_matrix_karate):
#     visualize = False
    
#     filtered_net = karate_club.filter_graph_by_LP(prob_matrix_df=prob_matrix_karate,
#                                                   n_swaps=5,
#                                                   max_attempts=50)

#     if visualize:
#         karate_club.visualize(show_labels=True)

#         filtered_net.visualize(show_labels=True)

#     assert np.sum(prob_matrix_karate.values[karate_club.getInteractionInts_as_tuple()] -
#                   prob_matrix_karate.values[filtered_net.getInteractionInts_as_tuple()]) > 0

#     perms = np.random.permutation(np.arange(karate_club.N_nodes))

#     prob_matrix_karate2 = prob_matrix_karate.iloc[perms].iloc[:, perms]
#     filtered_net2 = karate_club.filter_graph_by_LP(prob_matrix_df=prob_matrix_karate2,
#                                                    n_swaps=5,
#                                                    max_attempts=50)

#     assert np.any(prob_matrix_karate.values != prob_matrix_karate2.values)

#     assert filtered_net == filtered_net2


# def test_filter_graph_fast_by_LP(karate_club, dl_net_karate, karate_net_training_test):
#     visualize = False
#     np.random.seed(42)
#     X_train, X_test, Y_train, Y_test = karate_net_training_test

#     dl_net_karate.fit(X_train, Y_train,
#                       validation_data=(X_test, Y_test),
#                       allow_nans=False,
#                       verbose=2)

#     prob_matrix = dl_net_karate.predictProbMatrix()
#     print(prob_matrix)
#     filtered_net = karate_club.filter_graph_by_LP(prob_matrix_df=prob_matrix,
#                                                   n_swaps=5,
#                                                   max_attempts=50)

#     if visualize:
#         karate_club.visualize(show_labels=True)

#         filtered_net.visualize(show_labels=True)

#     assert np.sum(prob_matrix.values[karate_club.getInteractionInts_as_tuple()] -
#                   prob_matrix.values[filtered_net.getInteractionInts_as_tuple()]) > 0

#     perms = np.random.permutation(np.arange(karate_club.N_nodes))

#     prob_matrix2 = prob_matrix.iloc[perms].iloc[:, perms]
#     filtered_net2 = karate_club.filter_graph_by_LP(prob_matrix_df=prob_matrix2,
#                                                    n_swaps=5,
#                                                    max_attempts=50)

#     assert np.any(prob_matrix.values != prob_matrix2.values)

#     assert filtered_net == filtered_net2


def test_getInteractionInts_as_tuple(network_chain_4):

    tup = network_chain_4.getInteractionInts_as_tuple()
    print(network_chain_4.interactions)
    print(network_chain_4.getInteractionInts_as_tuple())

    assert np.all(network_chain_4.interactions['Gene_A'].values == tup[0])
    assert np.all(network_chain_4.interactions['Gene_B'].values == tup[1])

    A = network_chain_4.getAdjMatrix(as_df=True)

    A2 = np.zeros(A.shape)
    tup = network_chain_4.getInteractionInts_as_tuple(both_directions=True)

    assert len(tup[1]) == 2 * network_chain_4.N_interactions

    A2[tup] = 1

    assert np.all(A2 == A)


def test_filter_graph_by_LP_subset(probability_matrix):
    visualize = False
    node_names = ['North', 'East', 'South', 'West']
    net_df = pd.DataFrame({'Gene_A': ['North', 'East', 'South', 'West'],
                           'Gene_B': ['South', 'West', 'East', 'North']})

    net = UndirectedInteractionNetwork(net_df)

    prob_mat = np.array([[1, 0.9, 0.2, 0.9],
                         [0.9, 1, 0.9, 0.2],
                         [0.2, 0.9, 1, 0.9],
                         [0.9, 0.2, 0.9, 1]])

    prob_df = pd.DataFrame(prob_mat, columns=node_names, index=node_names)
    prob_df = prob_df[['East', 'North', 'South', 'West']].loc[['East', 'North', 'South', 'West']]
    new_graph = net.filter_graph_by_LP(prob_df,
                                       n_swaps=2,
                                       max_attempts=10)

    diff = np.sum(prob_df.values[net.getInteractionInts_as_tuple()] -
                  prob_df.values[new_graph.getInteractionInts_as_tuple()])

    assert np.abs(1.4 - np.abs(diff)) < 1e-4

    if visualize:

        net.visualize(show_labels=True)
        new_graph.visualize(show_labels=True)

    node_names = ['North', 'East', 'South', 'West', 'Center']
    net_df = pd.DataFrame({'Gene_A': ['North', 'East', 'South', 'West', 'Center', 'Center', 'Center', 'Center'],
                           'Gene_B': ['South', 'West', 'East', 'North', 'South', 'West', 'East', 'North']})

    net = UndirectedInteractionNetwork(net_df)

    prob_mat = np.array([[1, 0.9, 0.2, 0.9, 0.8],
                         [0.9, 1, 0.9, 0.2, 0.8],
                         [0.2, 0.9, 1, 0.9, 0.8],
                         [0.9, 0.2, 0.9, 1, 0.8],
                         [0.8, 0.8, 0.8, 0.8, 1]])

    prob_df = pd.DataFrame(prob_mat, columns=node_names, index=node_names)
    prob_df = prob_df[['Center', 'East', 'North', 'South', 'West']].loc[['Center', 'East', 'North', 'South', 'West']]

    new_graph = net.filter_graph_by_LP(prob_df,
                                       n_swaps=2,
                                       max_attempts=10)

    diff = np.sum(prob_df.values[net.getInteractionInts_as_tuple()] -
                  prob_df.values[new_graph.getInteractionInts_as_tuple()])

    print(diff)
    assert np.abs(1.4 - np.abs(diff)) < 1e-4
    assert new_graph.N_interactions == 8
    assert new_graph.N_nodes == 5

    if visualize:

        net.visualize(show_labels=True)
        new_graph.visualize(show_labels=True)

    prob_df_sub = prob_df[['East', 'North', 'South', 'West']].loc[['East', 'North', 'South', 'West']]

    new_graph = net.filter_graph_by_LP(prob_df_sub,
                                       n_swaps=2,
                                       max_attempts=10)

    diff = np.sum(prob_df.values[net.getInteractionInts_as_tuple()] -
                  prob_df.values[new_graph.getInteractionInts_as_tuple()])

    assert np.abs(1.4 - np.abs(diff)) < 1e-4
    assert new_graph.N_interactions == 8
    assert new_graph.N_nodes == 5


#TODO: add other tests for LP filtering fast
def test_filter_graph_by_LP_fastest_subset(probability_matrix):
    visualize = False
    node_names = ['North', 'East', 'South', 'West']
    net_df = pd.DataFrame({'Gene_A': ['North', 'East', 'South', 'West'],
                           'Gene_B': ['South', 'West', 'East', 'North']})

    net = UndirectedInteractionNetwork(net_df)

    prob_mat = np.array([[1, 0.9, 0.2, 0.9],
                         [0.9, 1, 0.9, 0.2],
                         [0.2, 0.9, 1, 0.9],
                         [0.9, 0.2, 0.9, 1]])

    prob_df = pd.DataFrame(prob_mat, columns=node_names, index=node_names)
    prob_df = prob_df[['East', 'North', 'South', 'West']].loc[['East', 'North', 'South', 'West']]

    new_graph = net.filter_graph_by_LP_fastest(prob_df,
                                               n_swaps=2,
                                               max_attempts=10)
    ints = new_graph.getInteractionNamed().applymap(lambda x: net.gene2int[x])
    ints = (ints.Gene_A.values, ints.Gene_B.values)

    diff = np.sum(prob_df.values[net.getInteractionInts_as_tuple()] -
                  prob_df.values[ints])

    assert np.abs(1.4 - np.abs(diff)) < 1e-5

    if visualize:

        net.visualize(show_labels=True)
        new_graph.visualize(show_labels=True)

    node_names = ['North', 'East', 'South', 'West', 'Center']
    net_df = pd.DataFrame({'Gene_A': ['North', 'East', 'South', 'West', 'Center', 'Center', 'Center', 'Center'],
                           'Gene_B': ['South', 'West', 'East', 'North', 'South', 'West', 'East', 'North']})

    net = UndirectedInteractionNetwork(net_df)

    prob_mat = np.array([[1, 0.9, 0.2, 0.9, 0.8],
                         [0.9, 1, 0.9, 0.2, 0.8],
                         [0.2, 0.9, 1, 0.9, 0.8],
                         [0.9, 0.2, 0.9, 1, 0.8],
                         [0.8, 0.8, 0.8, 0.8, 1]])

    prob_df = pd.DataFrame(prob_mat, columns=node_names, index=node_names)
    prob_df = prob_df[['Center', 'East', 'North', 'South', 'West']] \
                     .loc[['Center','East', 'North', 'South', 'West']]

    new_graph2 = net.filter_graph_by_LP_fastest(prob_df,
                                               n_swaps=2,
                                               max_attempts=10)

    ints = new_graph2.getInteractionNamed().applymap(lambda x: net.gene2int[x])
    ints = (ints.Gene_A.values, ints.Gene_B.values)

    diff = np.sum(prob_df.values[net.getInteractionInts_as_tuple()] - prob_df.values[ints])

    if visualize:

        net.visualize(show_labels=True)
        new_graph2.visualize(show_labels=True)

    assert np.abs(1.4 - np.abs(diff)) < 1e-5
    assert new_graph2.N_interactions == 8
    assert new_graph2.N_nodes == 5

    node_names = ['South', 'West', 'North', 'East']  # random order to test if this matters
    prob_df_sub = prob_df[node_names].loc[node_names]
    prob_df_sub = prob_df_sub[['East', 'North', 'South', 'West']].loc[['East', 'North', 'South', 'West']]

    new_graph3 = net.filter_graph_by_LP_fastest(prob_df_sub,
                                               n_swaps=2,
                                               max_attempts=10)

    diff = np.sum(prob_df.values[net.getInteractionInts_as_tuple()] -
                  prob_df.values[new_graph3.getInteractionInts_as_tuple()])

    assert np.abs(1.4 - np.abs(diff)) < 1e-5
    assert new_graph3.N_interactions == 8
    assert new_graph3.N_nodes == 5


# def test_get_KNN_network_(network_circle, karate_club):
#     knn_graph = network_circle.get_KNN_network_(kn=2)
#     expected_ints = network_circle.getInteractionInts_as_tuple(both_directions=True)
#     expected_ints = set(zip(list(network_circle.node_names[expected_ints[0]]),
#                             list(network_circle.node_names[expected_ints[1]])))

#     assert expected_ints == knn_graph.interactions_as_set()

#     new_net = karate_club.get_KNN_network_(kn=None)
#     new_net2 = karate_club.get_KNN_network(kn=None)

#     fast_ints = new_net.getInteractionInts_as_tuple()
#     slow_ints = new_net.getInteractionInts_as_tuple()

#     print(fast_ints)

#     diff = new_net.interactions_as_set(return_names=True).difference(new_net2.interactions_as_set(return_names=True))

#     print(diff)

#     assert new_net2 == new_net


# def test_get_KNN_network(network_circle, karate_club):
#     knn_graph = network_circle.get_KNN_network(kn=2)
#     expected_ints = network_circle.getInteractionInts_as_tuple(both_directions=True)
#     expected_ints = set(zip(list(network_circle.node_names[expected_ints[0]]),
#                             list(network_circle.node_names[expected_ints[1]])))

#     assert expected_ints == knn_graph.interactions_as_set()

#     new_net = karate_club.get_KNN_network(kn=None)

#     _, out_degrees = np.unique(new_net.interactions.values[:, 0], return_counts=True)

#     assert new_net.isConnected
#     assert np.all(out_degrees[0] == out_degrees)


def test_setEqual(network_chain_4, network_circle):

    network_chain_4.setEqual(network_circle)

    assert network_circle == network_chain_4


def test_getGeodesicDistance(karate_club):
    print(karate_club.int2gene)
    print(karate_club.N_nodes)

    pathlen_df1 = karate_club.getGeodesicDistance(['10', '13'], ['0', '16'])

    print(pathlen_df1)

    pathlen_df2 = karate_club.getGeodesicDistance([10, 13], [0, 16])

    print(pathlen_df2)
    print(pathlen_df2.columns.values)

    diff = pathlen_df1[['0', '16']].loc[['10', '13']].values - pathlen_df2.loc[[10, 13]][[0, 16]].values
    assert np.all(np.abs(diff) < 1e-4)

    nx_Graph = karate_club.getnxGraph(return_names=True)
    diff1 = karate_club.getGeodesicDistance(['10', '13'], ['0', '16'], nx_Graph=nx_Graph)

    nx_Graph = karate_club.getnxGraph(return_names=False)
    diff = karate_club.getGeodesicDistance([10, 13], [0, 16], nx_Graph=nx_Graph)


def test_calculate_proximity_significance(karate_club):
    nodes_from = ['16', '10', '5']
    nodes_to = ['4', '6']

    d, z, (m, s), pval = karate_club.calculate_proximity_significance(nodes_from, nodes_to, min_bin_size=10)

    print(z)
    print(pval)


def test_minimum_spanning_tree(network_chain_10):

    edge_list = network_chain_10.getMinimmumSpanningTree()
    print(edge_list)

    MST_undir = network_chain_10.getMinimmumSpanningTree(as_edge_list=False)
    print(MST_undir)

    assert MST_undir == network_chain_10


def test_sample_positives_and_negatives(karate_club):
    all_pos, all_neg = karate_club.sample_positives_and_negatives(neg_pos_ratio=3)

    assert not set(list(map(tuple, all_pos))) & set(list(map(tuple, all_neg))), \
        "getTrainTestPairs_MStree: overlap negatives train - test"

    assert len(set(list(map(tuple, all_neg)))) == len(all_neg)

    net2 = karate_club.deepcopy()
    net2.set_node_types({g: i % 2 for i, g in enumerate(net2.node_names)})

    all_pos, all_neg = net2.sample_positives_and_negatives(neg_pos_ratio=3)

    assert not set(list(map(tuple, all_pos))) & set(list(map(tuple, all_neg))), \
        "getTrainTestPairs_MStree: overlap negatives train - test"

    assert len(set(list(map(tuple, all_neg)))) == len(all_neg)


def test_ndex2_graph_init():
    ndex_example = UndirectedInteractionNetwork.from_ndex(ndex_id='4fde0a71-c571-11eb-9a85-0ac135e8bacf',
                                   keeplargestcomponent=False)
    
    ndex_example_lcc = UndirectedInteractionNetwork.from_ndex(ndex_id='4fde0a71-c571-11eb-9a85-0ac135e8bacf',
                                   keeplargestcomponent=True)
    
    ndex_example_large = UndirectedInteractionNetwork.from_ndex(ndex_id='c3554b4e-8c81-11ed-a157-005056ae23aa',
                                   keeplargestcomponent=False, 
                                   attributes_for_names='v', node_type=int)
    
