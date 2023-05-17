# NetworkAnalysis: A Python library for basic graph handling, edge sampling and representation learning.
![Coverage](./pytests/Coverage/coverage.svg)

# About NetworkAnalysis


- Graph initialization: Similar to the standard graph handling framework NetworkX, NetworkAnalysis offers additional functionality of removing disconnected components, adding custom node-to-integer mappings, removing self-loops and duplicate interactions, and specifying custom node types. 

- Edge Sampling: This is the main functionality of the package and offers a high level of granularity for edge sampling. The user can choose between balanced or unbalanced sampling, add specific sets of interactions that are to be included or excluded from the negative set. Also, the sampling algorithm is extendable to heterogeneous graphs where it will sample each edge type separately with, if desired, a separate negative to positive ratio and separate training/validation/testing ratio.

- NRL evaluation: Basic Network Representation Learning (NRL) methods are avaiable through OpenNE and include but are not limited to DeepWalk, Node2Vec and LINE.

- Embedding visualization: To evaluate the quality of the learned embeddings, be it from within the NetworkAnalysis framework or from any other embedding method, NetworkAnalysis offers the functionality to manipulate and visualize embddings as the user sees fit. For example, the embeddings can be clustered and checked against original labels using the Adjusted Mutual Information (AMI)  metric and subsequently visualized.

# Installation

NetworkAnalysis is tested on Python 3.10

**Option 1:** Through pip install
` pip install NetworkAnalysis`

**Option 2:** Cloning this repository

# Usage


## Command Line Interface

## NetworkAnalysis as API

# Contributing

# License

