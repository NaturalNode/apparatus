/*
Backwards compatibility wrapper for the original Apparatus KMeans API

This module provides the original Apparatus API for K-Means clustering.
The implementation uses the modernized KMeans internally.

For the modern API with additional features, see kmeans-modernized.d.ts
*/

/**
 * KMeans - Original Apparatus API
 * 
 * Supports the old Apparatus API:
 *   new KMeans(observations)
 *   kmeans.cluster(k)
 *   kmeans.createCentroids(k)
 *   kmeans.distanceFrom(centroids)
 */
export class KMeans {
  Observations: number[][]

  /**
   * Initialize with observations (old Apparatus API)
   * @param observations - Data points
   */
  constructor (observations: number[][])

  /**
   * Cluster the observations into k clusters (old Apparatus API)
   * @param k - Number of clusters
   * @returns Cluster assignments for each observation
   */
  cluster (k: number): number[]

  /**
   * Create initial centroids (old Apparatus API)
   * @param k - Number of clusters
   * @returns Initial centroid positions
   */
  createCentroids (k: number): number[][]

  /**
   * Calculate distances from observations to centroids (old Apparatus API)
   * @param centroids - Centroid positions
   * @returns Distance matrix
   */
  distanceFrom (centroids: number[][]): number[][]
}

