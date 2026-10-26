
Apparatus
=========

Apparatus is a collection of low-level machine learning algorithms for node.js.

Note that within Apparatus the interface to the algorithms in
primarily arrays of numbers and vectors. If you're  looking for feature
extraction from text or natural language check out the "natural"
[https://github.com/NaturalNode/natural](https://github.com/NaturalNode/natural) node package. Natural uses
many of these algorithms but adds a layer of natural language/text
feature extraction.

# Apparatus Machine Learning Algorithms Documentation

Documentation of the low-level machine learning algorithms implemented in Apparatus.

## Table of Contents

1. [Classifiers](#classifiers)
   - [Naïve Bayes](#naïve-bayes-classifier)
   - [Logistic Regression](#logistic-regression-classifier)
   - [Random Forest](#random-forest-classifier)
2. [Clusterers](#clusterers)
   - [K-Means](#k-means-clustering)

---

## Classifiers

Classification algorithms predict discrete categories/labels for input data.

### Naïve Bayes Classifier

**Type:** Probabilistic classifier  
**Use Cases:** Text classification, spam detection, sentiment analysis, category prediction  
**Time Complexity:** O(n) training, O(1) prediction  
**Space Complexity:** O(features × classes)

#### Overview

Naïve Bayes is a probabilistic classifier based on Bayes' theorem with the assumption that features are conditionally independent given the class label.

**Formula:**

$$P(C|X) = \frac{P(X|C) \times P(C)}{P(X)}$$

Where:
- **P(C|X)** = Posterior probability of class C given features X
- **P(X|C)** = Likelihood of features X given class C
- **P(C)** = Prior probability of class C
- **P(X)** = Probability of observing features X

#### How It Works

1. **Training Phase:**
   - Count feature occurrences for each class
   - Calculate feature probabilities per class
   - Store class priors (how often each class appears)

2. **Prediction Phase:**
   - For input features, compute probability of each class
   - Multiply: P(class) × P(feature1|class) × P(feature2|class) × ...
   - Return class with highest probability

#### Example

```javascript
const BayesClassifier = require('./lib/apparatus/classifier/bayes_classifier');

const classifier = new BayesClassifier();

// Training: [feature_vector, label]
classifier.addExample([1, 0, 1, 0], 'spam');
classifier.addExample([0, 1, 0, 1], 'ham');
classifier.addExample([1, 1, 1, 0], 'spam');
classifier.train();

// Prediction
const result = classifier.classify([1, 0, 1, 0]);  // Returns 'spam'
```

#### Parameters

- **Smoothing:** Laplace smoothing factor (default: 1.0)
  - Prevents zero probabilities for unseen features
  - Higher values = more smoothing

---

### Logistic Regression Classifier

**Type:** Linear classifier (multiclass support)  
**Use Cases:** Multi-class classification, probability estimation, linear decision boundaries  
**Time Complexity:** O(n × iterations × classes) training, O(features × classes) prediction  
**Space Complexity:** O(features × classes)

#### Overview

Logistic Regression is a linear classification algorithm that uses the sigmoid function to map input to probabilities. It supports multiclass classification by training separate binary classifiers for each class (one-vs-all).

**Formula:**

$$P(y=1|x) = \frac{1}{1 + e^{-(\theta^T x)}}$$

Where:
- **θ** = Learned weights (one set per class)
- **x** = Feature vector
- Sigmoid function maps linear combination to [0, 1]

#### How It Works

1. **Initialization:**
   - Start with random weights (θ)
   - Define learning data and labels

2. **Training via Gradient Descent:**
   - Compute predictions using current weights
   - Calculate cost (loss) using cross-entropy
   - Update weights in direction of steepest descent
   - Repeat for max iterations

3. **Prediction:**
   - Compute linear combination: z = θ^T × x
   - Apply sigmoid: P = 1 / (1 + e^-z)
   - If P > 0.5 → class 1, else → class 0

#### Example

```javascript
const LogisticRegression = require('./lib/apparatus/classifier/logistic_regression_classifier');

const classifier = new LogisticRegression();

// Training: array of [features] and array of [labels]
const examples = [
  [2.1, 1.0],
  [2.0, 0.9],
  [8.0, 8.0],
  [8.1, 7.9]
];
const labels = [0, 0, 1, 1];

classifier.addExample(examples[0], labels[0]);
classifier.addExample(examples[1], labels[1]);
classifier.addExample(examples[2], labels[2]);
classifier.addExample(examples[3], labels[3]);
classifier.train();

// Prediction (returns probability or class)
const prob = classifier.classify([2.0, 1.0]);  // Close to 0
const prob2 = classifier.classify([8.0, 8.0]); // Close to 1
```

#### Parameters

- **Learning Rate:** Controls step size in gradient descent
- **Max Iterations:** Maximum training iterations
- **Regularization:** Optional L2 regularization to prevent overfitting

---

### Random Forest Classifier

**Type:** Ensemble classifier (decision trees) - **Binary classification only**  
**Use Cases:** Binary classification, feature importance, non-linear boundaries  
**Time Complexity:** O(n × k × log n) training, O(k × depth) prediction  
**Space Complexity:** O(k × tree_nodes)  
**Constraint:** Only supports binary classification with classes **1** and **-1**

#### Overview

Random Forest is an ensemble method that builds multiple decision trees and combines their predictions. Each tree is trained on a random subset of data and features.

**Important:** This implementation only supports binary classification where labels must be **1** or **-1**.

#### How It Works

1. **Forest Creation:**
   - Build k decision trees (configurable)
   - For each tree: bootstrap sample of training data
   - At each node: randomly select subset of features
   - Split on feature that maximizes information gain

2. **Prediction:**
   - Each tree predicts independently
   - Return majority vote (classification) or average (regression)

3. **Feature Importance:**
   - Track how often features are used for splits
   - Normalize by depth/impact

#### Example

```javascript
const RandomForest = require('./lib/apparatus/classifier/randomforest_classifier');

const classifier = new RandomForest();

// Add training examples - MUST use labels 1 or -1
classifier.addExample([2.0, 1.0], 1);
classifier.addExample([2.1, 0.9], 1);
classifier.addExample([2.0, 1.1], 1);
classifier.addExample([8.0, 8.0], -1);
classifier.addExample([8.1, 7.9], -1);
classifier.addExample([8.0, 8.1], -1);

// Train with configuration
classifier.train({ numTrees: 10, maxDepth: 4 });

// Predict - returns probability and classifications
const result = classifier.classify([2.0, 1.0]); // Class 1
const result2 = classifier.classify([8.0, 8.0]); // Class -1
```

#### Parameters

- **numTrees:** Number of trees to build (default: 100)
  - Higher = better accuracy, slower training
  - Typical: 10-100

- **maxDepth:** Maximum tree depth (default: 4)
  - Controls tree complexity
  - Prevents overfitting

- **numTries:** Random hypotheses at each node (default: 10)
  - Controls randomness in feature selection

---

## Clusterers

Clustering algorithms partition data into groups without labeled targets.

### K-Means Clustering

**Type:** Unsupervised clustering (centroid-based)  
**Use Cases:** Customer segmentation, image compression, document clustering, anomaly detection  
**Time Complexity:** O(n × k × iterations × d)  
**Space Complexity:** O(n × d)

#### Overview

K-Means is an iterative algorithm that partitions data into k clusters by minimizing within-cluster variance. The goal is to find cluster centers (centroids) that minimize the sum of squared distances to all points.

**Objective Function:**

$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

Where:
- **k** = Number of clusters
- **C_i** = Set of points in cluster i
- **μ_i** = Centroid of cluster i
- **||x - μ_i||²** = Squared Euclidean distance

#### Algorithm Steps

1. **Initialization:**
   - Select k initial centroids
   - Options: random, k-means++, or user-provided

2. **Assignment Step:**
   - For each point: assign to nearest centroid
   - Calculate distances to all k centroids

3. **Update Step:**
   - Recompute each centroid as mean of assigned points
   - Move centroids toward cluster center

4. **Convergence Check:**
   - If centroids changed → repeat from Assignment
   - Else → done!

#### Variants in Apparatus

**Legacy Implementation (kmeans-legacy.js):**
- Uses Sylvester matrix library
- Implemented circa 2011
- Slower but foundational

**Modernized Implementation (kmeans-modernized.js):**
- Pure JavaScript arrays (no external dependencies)
- K-means++ initialization (better starting points)
- Multiple restart support (finds better solutions)
- **7.6x faster** than legacy (see benchmark)
- Thread-safe and scalable

#### K-Means++ Initialization

Instead of random centroids, k-means++ selects initial centers probabilistically:

1. Choose first centroid randomly from data
2. For i = 2 to k:
   - Probability of selecting point x is proportional to D(x)²
   - D(x) = distance to nearest existing centroid
3. Result: initial centers spread out, converges faster

#### Example

```javascript
// Modernized version (recommended)
const KMeans = require('./lib/apparatus/clusterer/kmeans');

const data = [
  [1, 1],
  [2, 2],
  [10, 10],
  [11, 11]
];

// Legacy API (backwards compatible)
const kmeans = new KMeans(data);
const result = kmeans.cluster(2);

// Access results via Sylvester-compatible API
console.log(result.elements.length);    // 4 assignments
console.log(result.e(1));               // Cluster of point 1 (1-indexed)
console.log(result.e(2));               // Cluster of point 2
// etc.
```

#### Advanced Example with Modernized API

```javascript
const KMeansModernized = require('./lib/apparatus/clusterer/kmeans-modernized');

const data = [[1,1], [2,2], [10,10], [11,11]];

// More control with modernized API
const kmeans = new KMeansModernized(2, {
  maxIterations: 100,        // Max iterations
  tolerance: 0.0001,         // Convergence threshold
  initialization: 'kmeans++', // 'random' or 'kmeans++'
  seed: 42                   // For reproducible results
});

kmeans.fit(data);

// Get results
const assignments = kmeans.getAssignments();  // [0, 0, 1, 1]
const clusters = kmeans.getClusters();        // [[0, 1], [2, 3]]
const centroids = kmeans.getCentroids();      // [[1.5, 1.5], [10.5, 10.5]]
```

#### Performance Comparison

See `benchmark/kmeans_benchmark.js` for full results:

| Dataset | Points | Modernized | Legacy | Speedup |
|---------|--------|------------|--------|---------|
| Random | 100 | 0.17 ms | 0.81 ms | **4.8x** |
| Small | 30 | 0.01 ms | 0.10 ms | **6.9x** |
| Medium | 250 | 0.18 ms | 1.16 ms | **6.3x** |
| Large | 1,000 | 0.88 ms | 9.14 ms | **10.4x** |
| Very Large | 5,000 | 4.07 ms | 44.71 ms | **11.0x** |

**Average speedup: 7.6x faster, 6.3% more memory**

#### Parameters

**KMeansModernized Constructor:**

```javascript
new KMeansModernized(k, options)
```

- **k** (required): Number of clusters
- **options.maxIterations**: Max iterations (default: 100)
- **options.tolerance**: Convergence tolerance (default: 0.0001)
- **options.initialization**: 
  - `'random'` - Random centroid selection
  - `'kmeans++'` - K-means++ initialization (recommended)
- **options.seed**: Random seed for reproducibility
- **options.restarts**: Multiple runs to find better solution
- **options.distanceFunction**: Custom distance metric (default: Euclidean)


## Comparison Table

| Algorithm | Type | Classes | Training | Prediction | Complexity | Best Use Case |
|-----------|------|---------|----------|-----------|-----------|---------------|
| Naïve Bayes | Classification | Multi | O(n) | O(1) | Low | Text, fast baseline |
| Logistic Regression | Classification | Multi | O(n×iter×c) | O(d×c) | Low-Mid | Linear boundaries, multiclass |
| Random Forest | Classification | **Binary only (±1)** | O(n×k×log n) | O(k×depth) | High | Non-linear binary problems |
| K-Means | Clustering | N/A | O(n×k×iter×d) | O(k×d) | Mid | Segmentation, exploration |

**Note:** d = dimensions, n = samples, k = clusters/trees, c = classes, iter = iterations

---

## Installation & Usage

### Install Apparatus

```bash
npm install apparatus
```

### Basic Example

```javascript
const BayesClassifier = require('apparatus/lib/apparatus/classifier/bayes_classifier');
const KMeans = require('apparatus/lib/apparatus/clusterer/kmeans');

// Classification
const classifier = new BayesClassifier();
classifier.addExample([1, 0], 'class_a');
classifier.addExample([0, 1], 'class_b');
classifier.train();
console.log(classifier.classify([1, 0])); // 'class_a'

// Clustering
const kmeans = new KMeans([[1,1], [2,2], [10,10]]);
const result = kmeans.cluster(2);
console.log(result.elements); // [1, 1, 2]
```

---

## References & Further Reading

- **Naïve Bayes:** [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- **Logistic Regression:** [Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
- **Random Forest:** [Breiman, 2001](https://en.wikipedia.org/wiki/Random_forest)
- **K-Means:** [MacQueen, 1967](https://en.wikipedia.org/wiki/K-means_clustering)
- **K-Means++:** [Arthur & Vassilvitskii, 2007](https://en.wikipedia.org/wiki/K-means%2B%2B)

---

**Last Updated:** February 24, 2026  
**Apparatus Version:** 0.0.11

