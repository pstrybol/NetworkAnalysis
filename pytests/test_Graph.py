import pandas as pd
import numpy as np
from NetworkAnalysis.Graph import Graph, select_random_sets
import time


def test_get_degree_binning(karate_club_Graph):
    bins = karate_club_Graph.get_degree_binning(bin_size=10)
    print(bins)

    assert np.all(len(v) >= 10 for g1, g2, v in bins)


def test_get_degree_equivalent(karate_club_Graph):

    seeds = [11, 10, 23] # same example as test_get_degree_binning, values taken from there
    eq_dict = karate_club_Graph.get_degree_equivalents(seeds, bin_size=10, return_names=False)

    print(eq_dict)

    assert 26 in eq_dict[seeds[0]]
    assert 28 in eq_dict[seeds[1]]
    assert 8 in eq_dict[seeds[2]]


def test_select_random_sets(karate_club_Graph):

    seeds = [11, 10, 23]  # same example as test_get_degree_binning, values taken from there
    eq_dict = karate_club_Graph.get_degree_equivalents(seeds, bin_size=10, return_names=False)

    rand_sets = select_random_sets(eq_dict, seeds, nsets=1000)
    print(eq_dict)
    print(rand_sets)

    assert rand_sets.shape[0] == 1000
    assert rand_sets.shape[1] == len(seeds)
    assert all(len(np.unique(gs)) == len(seeds) for gs in rand_sets)


def test_init(karate_club_Graph, karate_club_df):
    gene2int = karate_club_Graph.gene2int

    assert np.all(np.array(list(gene2int.keys())).astype(int) == np.array(list(gene2int.values())))
    karate_club_Graph = Graph(karate_club_df.astype(str))

    gene2int = karate_club_Graph.gene2int
    print(gene2int)