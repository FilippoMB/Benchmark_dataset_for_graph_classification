# Benchmark_dataset_for_graph_classification
This repository contains a synthetically generated dataset for quickly testing graph classification algorithms, such as Graph Kernels and Graph Neural Networks.

The purpose of this dataset is to make the features on the nodes and the adjacency matrix to be completely uninformative if considered alone.
Therefore, an alogorithm which relies solely on the node features or on the graph structure will fail to achieve good classification results.


## Results

We report some classification results obtained by using Graph Kernels and Graph Neural Networks.

#### Graph Kernels
The Graph Kernels are computed with the [GraKer](https://ysig.github.io/GraKeL/dev/index.html) library. After each kernel is computed, an SVM that uses as precomputed kernel the Graph Kernel is trained and then evaluated on the test data.
As SVM implementation, the [sklearn.svm](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm) module was used.
The code used to generate the results can be found in the [notebook]() of this repository.

Dependecies to run the notebook:
- scikitlearn
- networkx
- graker

| Dataset            | easy_small       |                | hard_small       |                |
|--------------------|------------------|----------------|------------------|----------------|
| Shortest Path      | Accuracy: 97.37   | Time: 23.57 s  | Accuracy: 57.14  | Time: 22.68 s  |
| Graphlet Sampling  | Accuracy: 42.11  | Time: 293.98 s | Accuracy: 35.71  | Time: 69.99 s  |
| Pyramid Match      | Accuracy: 44.74  | Time: 3.15 s   | Accuracy: 46.43  | Time: 5.68 s   |
| SVM Theta          | Accuracy: 39.47  | Time: 3.65 s   | Accuracy: 28.57  | Time: 7.04 s   |
| Neighborhood Hash  | Accuracy: 97.37  | Time: 3.13 s   | Accuracy: 53.57  | Time: 5.13 s   |
| Subtree WL         | Accuracy: 39.47  | Time: 0.02 s   | Accuracy: 28.57  | Time: 0.04 s   |
| ODD STH            | Accuracy: 65.79  | Time: 64.88 s  | Accuracy: 46.43  | Time: 51.12 s  |
| Propagation        | Accuracy: 84.21  | Time: 3.97 s   | Accuracy: 46.43  | Time: 5.52 s   |
| Pyramid Match      | Accuracy: 44.74  | Took: 3.67 s   | Accuracy: 46.43  | Time: 5.79 s   |
| Vertex Histogram   | Accuracy: 39.47  | Time: 0.02 s   | Accuracy:  28.57  | Time: 0.03 s   |
| Weisfeiler Lehman  | Accuracy: 97.37  | Time: 53.47 s | Accuracy: 57.14  | Time: 50.92 s |
| Hadamard Code      | Accuracy:  68.42  | Time: 44.5 s   |  Accuracy:  64.29   |  Time:  33.63          |
| Core Framework     | Accuracy: 97.37  | Time: 19.52 s  | Accuracy: 46.43  | Time: 19.8 s   |
