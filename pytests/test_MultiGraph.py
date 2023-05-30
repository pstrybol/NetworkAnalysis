from NetworkAnalysis.MultiGraph import MultiGraph, merge_labels_MLT
import pandas as pd
import numpy as np
from NetworkAnalysis.InteractionNetwork import UndirectedInteractionNetwork
from NetworkAnalysis.MultiGraph import get_random_rows, all_asserts_X_v2_undir, all_asserts_X_undir, all_asserts_X_v2_dir


def test_initializer(small_multigraph, small_heterogeneous_multigraph):
    assert 6 == small_multigraph.N_nodes

    df = small_multigraph.getInteractionNamed()
    mg2 = MultiGraph(df)

    assert small_multigraph == mg2

    print(small_heterogeneous_multigraph)
    print(np.unique(small_heterogeneous_multigraph.edge_types_named))
    assert np.all(np.array(["Interaction", "co_expression", "dependency"]) == np.unique(small_heterogeneous_multigraph.edge_types_named))


def test_representation(network_chain_4):

    edges = [('G0', 'G1'), ('G1', 'G3'), ('G2', 'G3'), ('G5', 'G4'), ('G3', 'G4')]
    network = pd.DataFrame(np.array(edges), columns=['GA', 'GB'])
    undir = UndirectedInteractionNetwork(network)

    print({"chain": network_chain_4, "random": undir})

    mg = MultiGraph(graph_dict={"chain": network_chain_4, "random": undir})

    print(mg)


def test_edge_to_edge_type(small_multigraph):
    mapper = small_multigraph.edge_to_edge_type()
    print(mapper)


def test_get_labels_from_pairs_and_mapper(small_multigraph):
    pair_list = [('G0', 'G1'), ('G1', 'G3'), ('G2', 'G3'), ('G5', 'G4'), ('G3', 'G4')]
    pair_list = small_multigraph.interactions_as_set(return_names=False)
    pair_list = [(t[0], t[1]) for t in pair_list]

    type_df = small_multigraph.get_labels_from_pairs_and_mapper(pair_list, as_dicts=True)

    print(type_df)

    assert len(pair_list) - 1 == sum(type_df["random"])
    assert len(pair_list) - 3 == sum(type_df["chain"])


def test_getAdjMatrix(small_multigraph):
    # edges = [('G0', 'G1'), ('G1', 'G3'), ('G2', 'G3'), ('G5', 'G4'), ('G3', 'G4')]
    # network = pd.DataFrame(np.array(edges), columns=['GA', 'GB'])
    # undir = UndirectedInteractionNetwork(network)
    #
    # print({"chain": network_chain_4, "random": undir})
    #
    # mg = MultiGraph(graph_dict={"chain": network_chain_4, "random": undir})
    A, node_names = small_multigraph.getAdjMatrix()
    expected_A = np.array([[0, 2, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 2, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0]])

    assert np.all(expected_A == A)
    assert np.all(node_names == np.array(["G" + str(i) for i in range(6)]))


def test_edges_to_type(small_multigraph):
    print(small_multigraph.edges_to_type)


def test_edge_list(small_multigraph):
    print(small_multigraph.edge_list)


def test_edge_types_named(small_multigraph):
    print(small_multigraph)
    print(small_multigraph.edge_types_named)

    assert np.all(np.array(["chain"] * 3 + ["random"] * 5) == np.array(small_multigraph.edge_types_named))


def test_interactions_as_set(small_multigraph):
    interactions_as_set = small_multigraph.interactions_as_set()
    expected = {('G0', 'G1', 'chain'), ('G1', 'G3', 'random'), ('G3', 'G4', 'random'), ('G2', 'G3', 'chain'),
                ('G1', 'G2', 'chain'), ('G2', 'G3', 'random'), ('G0', 'G1', 'random'), ('G4', 'G5', 'random')}

    assert expected == interactions_as_set

    interactions_as_set_unnamed = small_multigraph.interactions_as_set(return_names=False)
    expected = {(0, 1, 1), (1, 2, 0), (1, 3, 1), (3, 4, 1), (2, 3, 0), (4, 5, 1), (0, 1, 0), (2, 3, 1)}

    assert expected == interactions_as_set_unnamed


def test_eq(small_multigraph):

    assert small_multigraph == small_multigraph
    small_multigraph2 = small_multigraph.deepcopy()
    small_multigraph.change_edge_type_name({"random": "structured"})

    assert not (small_multigraph2 == small_multigraph)


def test_change_edge_type_name(small_multigraph):
    print(small_multigraph)
    small_multigraph.change_edge_type_name({"random": "structured"})
    print(small_multigraph)

    assert "structured" in small_multigraph.getInteractionNamed().type.values


def test_getEdgeType_subset(small_heterogeneous_multigraph):
    undir = small_heterogeneous_multigraph.getEdgeType_subset("dependency")
    print(undir.interactions_as_set())
    expected = {('C0', 'G1'), ('C0', 'G2'), ('C1', 'G2'), ('C0', 'G5'), ('C0', 'G3'), ('C1', 'G4')}
    assert expected == undir.interactions_as_set()

    undir2 = small_heterogeneous_multigraph.getEdgeType_subset("Interaction")

    assert ("C0" not in undir2) and ("C1" not in undir2)
    print(undir2.node_names)


def test_getMinimmumSpanningTree(small_multigraph):
    print(small_multigraph)
    edge_list = small_multigraph.getMinimmumSpanningTree(as_edge_list=True)
    sm = small_multigraph.getMinimmumSpanningTree(as_edge_list=False)
    print(sm)

    assert set(edge_list) == set(sm.edge_list())
    assert 6 == sm.N_nodes


def test_is_bipartite(small_multigraph, small_heterogeneous_multigraph):
    bipartite, l, r = small_multigraph.is_bipartite

    assert not bipartite
    assert l is None
    assert r is None

    bipartite, l, r = small_heterogeneous_multigraph.is_bipartite

    assert not bipartite
    assert l is None
    assert r is None

    bipartite, l, r = small_heterogeneous_multigraph.getEdgeType_subset("dependency").is_bipartite

    assert bipartite
    assert (set(l) == {"C0", "C1"}) or (set(l) == {'G1', 'G3', 'G4', 'G5', 'G2'})
    assert (set(r) == {"C0", "C1"}) or (set(r) == {'G1', 'G3', 'G4', 'G5', 'G2'})


def test_get_interactions_per_node_type(small_heterogeneous_multigraph):

    print(small_heterogeneous_multigraph.get_interactions_per_node_type())


def test_get_node_type_edge_counts(small_heterogeneous_multigraph):
    odf = small_heterogeneous_multigraph.get_node_type_edge_counts()
    print(odf)


def test_get_edges_by_node_types(small_heterogeneous_multigraph):
    print(small_heterogeneous_multigraph.get_edges_by_node_types(node_type1="cell line", node_type2="gene"))


def test_sample_positives_and_negatives(small_heterogeneous_multigraph):
    print(small_heterogeneous_multigraph.node_types)

    all_pos, all_neg = small_heterogeneous_multigraph.sample_positives_and_negatives()
    print(all_pos)
    print(all_neg)

    named_negs = [(small_heterogeneous_multigraph.int2gene[t[0]], small_heterogeneous_multigraph.int2gene[t[1]])
                    for t in all_neg]

    named_pos = [(small_heterogeneous_multigraph.int2gene[t[0]], small_heterogeneous_multigraph.int2gene[t[1]])
                    for t in all_pos]

    print(named_pos)
    print(named_negs)

    all_pos_named, all_neg_named = small_heterogeneous_multigraph.sample_positives_and_negatives(return_names=True)
    print(all_pos_named)
    print(all_neg_named)


def test_merge_labels_MLT():

    X1 = np.array([['apples', 'kiwis'], ['apples', 'spinach'], ['apples', 'bananas']])
    X2 = np.array([['bananas', 'lemons'], ['bananas', 'spinach'], ['lemons', 'apples']])

    Y1 = np.array([1, 1, 0])
    Y2 = np.array([1, 0, 0])

    X_dict = {'Green': X1, 'Yellow': X2}
    Y_dict = {'Green': Y1, 'Yellow': Y2}

    X, Y = merge_labels_MLT(X_dict, Y_dict)

    assert X.shape[0] == 6
    assert Y['Yellow'].shape[0] == 6
    assert Y['Green'].sum() == 2
    assert Y['Yellow'].sum() == 1
    assert np.issubdtype(Y['Yellow'].dtype, np.integer)

    X, Y = merge_labels_MLT(X_dict, Y_dict, fill_with_NaNs=True)

    assert X.shape[0] == 6
    assert Y['Yellow'].shape[0] == 6
    assert not np.issubdtype(Y['Yellow'].dtype, np.integer)

    X3 = np.array([['oranges', 'kakis'], ['oranges', 'grapefruit'], ['grapefruit', 'mandarin']])
    Y3 = np.array([1, 1, 1])

    X_dict['Orange'], Y_dict['Orange'] = X3, Y3

    X, Y = merge_labels_MLT(X_dict, Y_dict)

    assert X.shape[0] == 9
    assert Y['Yellow'].shape[0] == 9
    assert Y['Green'].sum() == 2
    assert Y['Yellow'].sum() == 1
    assert Y['Orange'].sum() == 3
    assert np.issubdtype(Y['Yellow'].dtype, np.integer)


def test_sample_positives_negatives_in_train_test_validation(small_heterogeneous_multigraph):

    pos_train, neg_train, pos_validation, neg_validation, \
                    pos_test, neg_test = small_heterogeneous_multigraph.sample_positives_negatives_in_train_test_validation()

    all_asserts_X_v2_undir(pos_train, pos_validation, pos_test, neg_train, neg_validation, neg_test,
                            small_heterogeneous_multigraph)



def test_getTrainTestData(small_multigraph):

    X_train, X_val, X_test, Y_train, Y_val, Y_test = small_multigraph.getTrainTestData()
    all_asserts_X_undir(X_train, X_val, X_test, Y_train, Y_val, Y_test, small_multigraph)


def test_gene2int_init(small_heterogeneous_multigraph):
    gene2int = small_heterogeneous_multigraph.gene2int
    df = small_heterogeneous_multigraph.getInteractionNamed()

    gene2int["G1"] = 100000

    mg = MultiGraph(df, gene2int=gene2int)

    assert mg.gene2int["G1"] == 100000

    gene2int["G2"] = 2

    flag = False

    try:
        mg = MultiGraph(df, gene2int=gene2int)
    except AssertionError:
        flag = True

    assert flag

    gene2int["G2"] = 4

    gene2int["C7"] = 80
    mg = MultiGraph(df, gene2int=gene2int)

    gene2int.pop("C1")

    try:
        mg = MultiGraph(df, gene2int=gene2int)
    except IOError:
        flag = True

    assert flag


def test_get_random_rows():

    arr = np.arange(10)
    train, val, test = get_random_rows(arr, fraction1=0.1, fraction2=0.2)

    assert 7 == len(train)
    assert 2 == len(val)
    assert 1 == len(test)

    arr2 = np.transpose(np.vstack([arr, arr]))
    train, val, test = get_random_rows(arr2, fraction1=0.1, fraction2=None, as_list_of_tuples=True)

    assert 9 == len(train)
    assert 0 == len(val)
    assert 1 == len(test)
