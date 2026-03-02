/*
Copyright (c) 2026, Hugo W.L. ter Doest

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

/*
Backwards compatibility wrapper for the original Apparatus KMeans API

This module provides the original Apparatus API for K-Means clustering.
It extends KMeansModernized with a legacy-compatible interface.

Old API usage:
  const kmeans = new KMeans([[1,2], [3,4], [5,6]]);
  const assignments = kmeans.cluster(3);
*/

'use strict'

const KMeansModernized = require('./kmeans-modernized')
const VectorLike = require('./vector-like')

// Use utility functions from KMeansModernized
const { euclideanDistance, createRng } = KMeansModernized

/**
 * KMeans - Original Apparatus API wrapper
 * 
 * Extends KMeansModernized to provide backward-compatible API:
 *   new KMeans(observations)
 *   kmeans.cluster(k)
 *   kmeans.createCentroids(k)
 *   kmeans.distanceFrom(centroids)
 */
class KMeans extends KMeansModernized {
  /**
   * Initialize with observations (old Apparatus API)
   * @param {Array<Array<number>>} observations - Data points
   */
  constructor (observations) {
    // Call parent with placeholder k (will be set in cluster())
    super(1)
    
    if (!Array.isArray(observations)) {
      throw new Error('KMeans expects an array of observations')
    }
    
    if (observations.length === 0) {
      throw new Error('Observations cannot be empty')
    }
    
    if (!Array.isArray(observations[0])) {
      throw new Error('Observations must be an array of arrays')
    }
    
    this.Observations = observations
  }

  /**
   * Cluster the observations into k clusters (old Apparatus API)
   * @param {number} k - Number of clusters
   * @returns {VectorLike} - Cluster assignments wrapped in Sylvester-compatible object
   */
  cluster (k) {
    if (!isFinite(k) || k < 1) {
      throw new Error('k must be a positive integer')
    }
    
    if (k > this.Observations.length) {
      throw new Error(`k (${k}) cannot be greater than number of observations (${this.Observations.length})`)
    }
    
    // Set k and fit using parent's fit method
    this.k = k
    this.fit(this.Observations)
    
    // Return assignments wrapped in Sylvester-like Vector for backwards compatibility
    // VectorLike handles 0->1 index conversion lazily
    return new VectorLike(this.getAssignments())
  }

  /**
   * Create initial centroids (old Apparatus API)
   * @param {number} k - Number of clusters
   * @returns {Array<Array<number>>} - Initial centroid positions
   */
  createCentroids (k) {
    if (!isFinite(k) || k < 1) {
      throw new Error('k must be a positive integer')
    }
    
    const rng = createRng()
    return this.initializeRandomCentroids(this.Observations, rng)
  }

  /**
   * Calculate distances from observations to centroids (old Apparatus API)
   * @param {Array<Array<number>>} centroids - Centroid positions
   * @returns {Array<Array<number>>} - Distance matrix
   */
  distanceFrom (centroids) {
    if (!Array.isArray(centroids)) {
      throw new Error('centroids must be an array')
    }
    
    if (centroids.length === 0) {
      throw new Error('centroids cannot be empty')
    }
    
    const distances = []
    
    for (let i = 0; i < this.Observations.length; i++) {
      const distRow = []
      
      for (let j = 0; j < centroids.length; j++) {
        distRow.push(euclideanDistance(this.Observations[i], centroids[j]))
      }
      
      distances.push(distRow)
    }
    
    return distances
  }
}

module.exports = KMeans
