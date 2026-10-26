/*
Copyright (c) 2026 Hugo W.L. ter Doest

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * Options for KMeansModernized clustering
 */
export interface KMeansOptions {
  /** Maximum number of iterations (default: 100) */
  maxIterations?: number
  /** Convergence tolerance (default: 0.0001) */
  tolerance?: number
  /** Initialization method: 'random' or 'kmeans++' (default: 'kmeans++') */
  initialization?: 'random' | 'kmeans++'
  /** Random seed for deterministic results (default: undefined) */
  seed?: number
  /** Number of restarts to pick the best run (default: 1) */
  restarts?: number
}

/**
 * KMeansModernized - Modern K-Means clustering implementation
 * 
 * Advanced features:
 * - K-means++ initialization (better than random)
 * - Multiple restarts with best result selection
 * - Deterministic seeding for reproducibility
 * - Tolerance-based convergence
 * 
 * Usage:
 *   const kmeans = new KMeansModernized(3, { restarts: 10 });
 *   kmeans.fit(data);
 *   const assignments = kmeans.getAssignments();
 */
export class KMeansModernized {
  /** Number of clusters */
  k: number
  /** Maximum number of iterations */
  maxIterations: number
  /** Convergence tolerance */
  tolerance: number
  /** Initialization method */
  initialization: string
  /** Random seed */
  seed: number | undefined
  /** Number of restarts */
  restarts: number
  /** Cluster centroids */
  centroids: number[][] | null
  /** Cluster assignments (indices of points in each cluster) */
  clusters: number[][] | null
  /** Cluster assignment for each data point */
  assignments: number[] | null
  /** Number of iterations in the final fit */
  iterations: number

  /**
   * Initialize KMeansModernized
   * @param k - Number of clusters
   * @param options - Configuration options
   */
  constructor (k: number, options?: KMeansOptions)

  /**
   * Initialize centroids using random selection
   * @param data - Data points
   * @param rng - Random number generator function
   * @returns Initial centroid positions
   */
  initializeRandomCentroids (data: number[][], rng: () => number): number[][]

  /**
   * Initialize centroids using k-means++ algorithm
   * @param data - Data points
   * @param rng - Random number generator function
   * @returns Initial centroid positions
   */
  initializeKMeansPlusPlusCentroids (data: number[][], rng: () => number): number[][]

  /**
   * Assign each data point to the nearest centroid
   * @param data - Data points
   * @param centroids - Centroid positions
   */
  assignPointsToClusters (
    data: number[][],
    centroids?: number[][]
  ): { clusters: number[][]; assignments: number[] }

  /**
   * Update centroids based on cluster means
   * @param data - Data points
   * @param clusters - Point indices for each cluster
   * @param currentCentroids - Current centroid positions
   * @returns Updated centroid positions
   */
  updateCentroids (
    data: number[][],
    clusters: number[][],
    currentCentroids: number[][]
  ): number[][]

  /**
   * Check if centroids have converged
   * @param oldCentroids - Previous centroid positions
   * @param newCentroids - New centroid positions
   * @returns True if converged within tolerance
   */
  hasConverged (oldCentroids: number[][], newCentroids: number[][]): boolean

  /**
   * Calculate total inertia (sum of squared distances to centroids)
   * @param data - Data points
   * @param centroids - Centroid positions
   * @param assignments - Cluster assignments
   * @returns Total inertia
   */
  calculateInertia (data: number[][], centroids: number[][], assignments: number[]): number

  /**
   * Run a single K-Means fit with a given RNG
   * @param data - Data points
   * @param rng - Random number generator function
   */
  runSingleFit (
    data: number[][],
    rng: () => number
  ): { centroids: number[][]; clusters: number[][]; assignments: number[]; iterations: number; inertia: number }

  /**
   * Fit the model to the data
   * @param data - Array of data points (vectors)
   * @returns Returns this for chaining
   */
  fit (data: number[][]): KMeansModernized

  /**
   * Predict cluster assignments for new data points
   * @param data - Data points to predict
   * @returns Array of cluster indices
   */
  predict (data: number[][] | number[]): number[]

  /**
   * Get the centroids
   * @returns Cluster centroids
   */
  getCentroids (): number[][]

  /**
   * Get cluster assignments for the training data
   * @returns Array of cluster indices
   */
  getAssignments (): number[]

  /**
   * Get clusters (indices of points in each cluster)
   * @returns Array of clusters
   */
  getClusters (): number[][]
}
