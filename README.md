# NetworkAnalysis: A Python framework for graph handling
<!-- ![Coverage](./pytests/Coverage/coverage.svg) -->

# About 

NetworkAnalysis provides a python-based graph handling framework with a **focus biomedical graphs** capable of:
- <u>Graph initialization and handling</u>: removing disconnected components, adding custom node-to-integer mappings, removing self-loops and duplicate interactions, custom node types, obtain N-order neighbours, find communities using Louvain algorithm, clustering, and much more.

- <u>Edge Sampling</u>e: NetworkAnalysis offer a high level of granularity when it comes to edge sampling for Link Prediction. The user can choose between balanced, unbalanced or graph distance-based sampling strategies. Additionally, specific sets of interactions that are to be included or excluded in the train/test sets can be added. 

- <u>Network Representaiton Learning (NRL) evaluation</u>: Basic NRL methods are avaiable through OpenNE and include but are not limited to DeepWalk, Node2Vec and LINE. After the representations are learned these can be plotted, clustered or written to csv files. 

<!-- - <u>Embedding visualization </u>: To evaluate the quality of the learned embeddings, be it from within the NetworkAnalysis framework or from any other embedding method, NetworkAnalysis offers the functionality to manipulate and visualize embddings as the user sees fit. For example, the embeddings can be clustered and checked against original labels using the Adjusted Mutual Information (AMI)  metric and subsequently visualized. -->

Other functionalities such as handling of directed and heterogeneous graphs will be added in future updates. 

NetworkAnalysis functionalities have been tested through the appropriate pytests.

# Installation

NetworkAnalysis is tested on Python 3.10

**Option 1:** Cloning this repository
```
git clone git@github.com:pstrybol/NetworkAnalysis.git
python setup.py install
```

**Option 2:** Through pip install -> TBA

# Usage

The `examples/` folder contains various jupter notebooks to assist in the usage of NetworkAnalysis functionalities. For now, NetworkAnalysis is offered solely as an API yet Command Line Interface (CLI) will be added in a future update.

# Contributing

Any suggestions or contributions to improving NetworkAnalysis are greatly appreciated. Feel free open issues tagged with the appropriate label ("feature request", "bug", etc.). Alternatively you can email me directly with feedback/suggestions at: pieterpaul.strybol@ugent.be

