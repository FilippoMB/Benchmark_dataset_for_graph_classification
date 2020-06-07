# Benchmark dataset for graph classification
This repository contains datasets to quickly test graph classification algorithms, such as Graph Kernels and Graph Neural Networks.

The purpose of this dataset is to make the features on the nodes and the adjacency matrix to be completely uninformative if considered alone.
Therefore, an algorithm that relies only on the node features or on the graph structure will fail to achieve good classification results.

## Dataset details

The dataset consists of graphs belonging to 3 different classes. The number of nodes in each graph is variable and the feature vector on each node is a one-hot vector of size 5, which encodes the color of the node. The class is determined by the relative position of the colors on the graph.

![](https://github.com/FilippoMB/Benchmark_dataset_for_graph_classification/blob/master/img/sample_graph.png) 
![](https://github.com/FilippoMB/Benchmark_dataset_for_graph_classification/blob/master/img/sample_graph2.png) 

There are 4 versions of the dataset

- **small_easy:** 100 graphs per class, number of nodes varying in 40 and 80. Highly connected graphs.
- **easy:** 600 graphs per class, number of nodes varying in 100 and 200. Highly connected graphs.
- **small_hard:** 100 graphs per class, number of nodes varying in 40 and 80. Sparse graphs.
- **hard:** 600 graphs per class, number of nodes varying in 100 and 200. Sparse graphs.

In the hard dataset, it is necessary to consider higher order neighborhoods to understand the correct class and the graphs might be disconnected.

| Dataset    | # classes | # graphs | TR size | VAL size | TEST size | avg nodes | avg edges | Node Attr. (Dim.) |
|------------|-----------|----------|---------|----------|-----------|-----------|-----------|-------------------|
| easy_small | 3         | 300      | 239     | 30       | 31        | 58.25     | 358.8     | 5                 |
| hard_small | 3         | 300      | 245     | 29       | 26        | 58.64     | 224.94    | 5                 |
| easy       | 3         | 1800     | 1475    | 162      | 163       | 147.82    | 922.66    | 5                 |
| hard       | 3         | 1800     | 1451    | 159      | 190       | 148.32    | 572.32    | 5                 |

#### Format

The dataset is already split in training, validation and classification sets.
Each set contains:
- the list of adjacency matrices in csr_matrix format,
- the list of node features as numpy arrays,
- the class labels contained in a numpy array,

The following code snippet shows how to load the data

````python
import numpy as np

loaded = np.load('datasets/hard.npz', allow_pickle=True)

X_train = loaded['tr_feat'] # node features
A_train = list(loaded['tr_adj']) # list of adjacency matrices
y_train = loaded['tr_class'] # class labels

X_val = loaded['val_feat'] # node features
A_val = list(loaded['val_adj']) # list of adjacency matrices
y_val = loaded['val_class'] # class labels

X_test = loaded['te_feat'] # node features
A_test = list(loaded['te_adj']) # list of adjacency matrices
y_test = loaded['te_class'] # class labels

# OPTIONAL - Convert to networkx format
import networkx as nx

G_train = []
for a, x in zip(A_train, X_train):
    G = nx.from_scipy_sparse_matrix(a)
    x_tuple = tuple(map(tuple, x))
    nx.set_node_attributes(G, dict(enumerate(x_tuple)), 'features')
    G_train.append(G)
    
G_val = []
for a, x in zip(A_val, X_val):
    G = nx.from_scipy_sparse_matrix(a)
    x_tuple = tuple(map(tuple, x))
    nx.set_node_attributes(G, dict(enumerate(x_tuple)), 'features')
    G_val.append(G)
    
G_test = []
for a, x in zip(A_test, X_test):
    G = nx.from_scipy_sparse_matrix(a)
    x_tuple = tuple(map(tuple, x))
    nx.set_node_attributes(G, dict(enumerate(x_tuple)), 'features')
    G_test.append(G)
````

## Results
Classification results obtained by using Graph Kernels and other techniques are reported below.

Feel free to send a pull request if you have results you'd like to share!

#### Graph Kernels
The Graph Kernels are computed with the [GraKeL](https://ysig.github.io/GraKeL/dev/index.html) library. After each kernel is computed, an SVM that uses as precomputed kernel the Graph Kernel is trained and then evaluated on the test data.
As SVM implementation, the [sklearn.svm](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm) module was used.
The code used to generate the results can be found in the [notebook](https://github.com/FilippoMB/Benchmark_dataset_for_graph_classification/blob/master/Example.ipynb) of this repository.

Dependecies to run the notebook:
- scikitlearn ````pip install sklearn````
- networkx ````pip install networkx````
- grakel ````pip install grakel-dev````

| Dataset            | easy_small       |                | hard_small       |                |
|--------------------|------------------|----------------|------------------|----------------|
| Shortest Path      | Accuracy: 100    | Time: 20.67 s  | Accuracy: 69.23  | Time: 7.85 s   |
| Graphlet Sampling  | Accuracy: 41.94  | Time: 281.35 s | Accuracy:  38.46 | Time:  37.84 s |
| Pyramid Match      | Accuracy: 51.61  | Time: 2.91 s   | Accuracy:  23.08 | Time:  2.86 s  |
| SVM Theta          | Accuracy: 32.26  | Time: 3.34 s   | Accuracy:  23.08 | Time:  2.91 s  |
| Neighborhood Hash  | Accuracy: 90.32  | Time: 2.73 s   | Accuracy: 69.23  | Time: 2.71 s   |
| Subtree WL         | Accuracy: 29.03  | Time: 0.01 s   | Accuracy: 15.38  | Time: 0.03 s   |
| ODD STH            | Accuracy: 77.42  | Time: 58.75 s  | Accuracy: 42.31  | Time: 24.48 s  |
| Propagation        | Accuracy: 87.1   | Time: 3.35 s   | Accuracy:  53.85 | Time:  2.61 s  |
| Vertex Histogram   | Accuracy: 29.03  | Time: 0.02 s   | Accuracy: 15.38  | Time: 0.01 s   |
| Weisfeiler Lehman  | Accuracy: 100    | Time: 151.81 s | Accuracy:  73.08 | Time: 58.92 s  |
| Core Framework     | Accuracy: 100    | Time: 62.18 s  | Accuracy:  69.23 | Time:  18.62 s |

#### Embedding Simplicial Complexes (ESC)
Technique proposed in \[1\].

| Dataset            | easy_small       |                | hard_small       |                |
|--------------------|------------------|----------------|------------------|----------------|
| ESC +  RBF-SVM | Accuracy: 74.19 ± 6.84  | Time: 0.68 s| Accuracy: 48.46 ± 8.43| Time: 0.48 s|
| ESC +  L1-SVM  | Accuracy: 94.19 ± 2.70  | Time: 0.68 s| Accuracy: 70.77 ± 5.83| Time: 0.48 s|
| ESC +  L2-SVM  | Accuracy: 92.26 ± 2.89  | Time: 0.68 s| Accuracy: 69.23 ± 5.44| Time: 0.48 s|

| Dataset            | easy             |                | hard             |                |
|--------------------|------------------|----------------|------------------|----------------|
| ESC +  RBF-SVM | Accuracy: 80.37 ± 7.04 | Time: 10.94 s| Accuracy: 62.53 ± 4.58| Time: 16.65 s|
| ESC +  L1-SVM  | Accuracy: 96.07 ± 0.93 | Time: 10.94 s| Accuracy: 72.21 ± 1.01| Time: 16.65 s|
| ESC +  L2-SVM  | Accuracy: 93.37 ± 1.96 | Time: 10.94 s| Accuracy: 69.26 ± 1.85| Time: 16.65 s|

#### Hypergraph kernels
Technique proposed in \[2\].

| Dataset            | easy_small       |                | hard_small       |                |
|--------------------|------------------|----------------|------------------|----------------|
| Hist Kernel      | Accuracy: 0.94 ± 0.02 | Time: 0.72 s| Accuracy: 0.77 ± 0.02 | Time: 0.46 s|
| Jaccard Kernel   | Accuracy: 0.94 ± 0.0  | Time: 0.86 s| Accuracy: 0.68 ± 0.02 | Time: 0.54 s|
| Edit Kernel      | Accuracy: 0.91 ± 0.01 | Time: 9.97 s| Accuracy: 0.6 ± 0.02  | Time: 7.70 s|
| Stratedit Kernel | Accuracy: 0.94 ± 0.0  | Time: 5.14 s| Accuracy: 0.58 ± 0.02 | Time: 4.79 s|

| Dataset            | easy                  |                | hard                  |                |
|--------------------|-----------------------|----------------|-----------------------|----------------|
| Hist Kernel        | Accuracy: 0.94 ± 0.01 | Time: 10.39  s  | Accuracy: 0.72 ± 0.01 | Time: 6.93    s |
| Jaccard Kernel     | Accuracy: 0.94 ± 0.01 | Time: 14.15  s  | Accuracy: 0.63 ± 0.0  | Time: 8.11    s |
| Edit Kernel        | Accuracy: 0.93 ± 0.0  | Time: 2784.47 s | Accuracy: 0.6 ± 0.0   | Time: 2183.41 s |
| Stratedit Kernel   | Accuracy: 0.93 ± 0.0  | Time: 932.96  s | Accuracy: 0.6 ± 0.01  | Time: 954.87  s |


## References
\[1\] Martino A, Giuliani A, Rizzi A., (Hyper) Graph Embedding and Classification via Simplicial Complexes. Algorithms. 2019 Nov; 12(11):223.

\[2\] Martino A. and Rizzi A., (Hyper)graph kernels over simplicial complexes. 2020. Pattern Recognition.

## License
The dataset and the code are released under the MIT License. See the attached LICENSE file.
