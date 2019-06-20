# Benchmark_dataset_for_graph_classification
This repository contains 4 different synthetically generated datasets for testing graph classification algorithms, such as Graph Kernels and Graph Neural Networks.


## Results

Graph Kernels + SVM

| Dataset            | easy_small       |                | hard_small       |                |
|--------------------|------------------|----------------|------------------|----------------|
| Shortest Path      | Accuracy: 96.3   | Time: 23.57 s  | Accuracy: 72.22  | Time: 22.68 s  |
| Graphlet Sampling  | Accuracy: 33.33  | Time: 293.98 s | Accuracy: 44.44  | Time: 69.99 s  |
| Pyramid Match      | Accuracy: 40.74  | Time: 3.15 s   | Accuracy: 36.11  | Time: 5.68 s   |
| SVM Theta          | Accuracy: 18.52  | Time: 3.65 s   | Accuracy: 33.33  | Time: 7.04 s   |
| Neighborhood Hash  | Accuracy: 92.59  | Time: 3.13 s   | Accuracy: 80.56  | Time: 5.13 s   |
| Subtree WL         | Accuracy: 18.52  | Time: 0.02 s   | Accuracy: 30.56  | Time: 0.04 s   |
| ODD STH            | Accuracy: 81.48  | Time: 64.88 s  | Accuracy: 44.44  | Time: 51.12 s  |
| Propagation        | Accuracy: 85.19  | Time: 3.97 s   | Accuracy: 36.11  | Time: 5.52 s   |
| Pyramid Match      | Accuracy: 40.74  | Took: 3.67 s   | Accuracy: 36.11  | Time: 5.79 s   |
| Vertex Histogram   | Accuracy: 18.52  | Time: 0.02 s   | Accuracy: 30.56  | Time: 0.03 s   |
| Weisfeiler Lehman  | Accuracy: 66.67  | Time: 116.47 s | Accuracy: 27.78  | Time: 104.29 s |
| Hadamard Code      | Accuracy: 70.37  | Time: 44.5 s   |                  |                |
| Core Framework     | Accuracy: 85.19  | Time: 19.52 s  | Accuracy: 36.11  | Time: 19.8 s   |
