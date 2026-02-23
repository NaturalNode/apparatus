/*
Backwards compatibility wrapper for the original Apparatus KMeans API

This module provides the original Apparatus API for K-Means clustering.
The implementation uses the modernized KMeans internally.

Old API usage:
  const kmeans = new KMeans([[1,2], [3,4], [5,6]]);
  const assignments = kmeans.cluster(3);

For the modern API with additional features, see kmeans-modernized.js:
  const { KMeansModernized } = require('natural');
  const kmeans = new KMeansModernized(3, { restarts: 10 });
  kmeans.fit(data);
*/

'use strict'

const KMeansModernized = require('./kmeans-modernized')

/**
 * Euclidean distance calculation
 */
function euclideanDistance (a, b) {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i]
    sum += diff * diff
  }
  return Math.sqrt(sum)
}

/**
 * Create a deterministic RNG when a seed is provided
 */
function createRng (seed) {
  if (typeof seed !== 'number' || !isFinite(seed)) {
    return Math.random
  }

  let state = seed >>> 0
  return function () {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 4294967296
  }
}

/**
 * Simple wrapper to make arrays compatible with Sylvester Vector API
 * The legacy API used Sylvester Vectors which have .elements and .e(i)
 */
class VectorLike {
  constructor (zeroIndexedArray) {
    // Store 0-indexed array internally for efficiency
    this._array = zeroIndexedArray
    this._converted = null
  }
  
  /**
   * Get element at 1-based index (Sylvester compatibility)
   * Converts from 0-indexed to 1-indexed on the fly
   * @param {number} i - 1-based index
   * @returns {number} - Element value (1-indexed cluster assignment)
   */
  e (i) {
    return this._array[i - 1] + 1
  }
  
  /**
   * Lazy getter for elements array (for .length and direct access)
   * Only converts when accessed, caches the result
   */
  get elements () {
    if (!this._converted) {
      this._converted = this._array.map(a => a + 1)
    }
    return this._converted
  }
}

/**
 * KMeans - Original Apparatus API wrapper
 * 
 * Supports the old Apparatus API:
 *   new KMeans(observations)
 *   kmeans.cluster(k)
 *   kmeans.createCentroids(k)
 *   kmeans.distanceFrom(centroids)
 */
class KMeans {
  /**
   * Initialize with observations (old Apparatus API)
   * @param {Array<Array<number>>} observations - Data points
   */
  constructor (observations) {
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
    
    // Use modernized KMeans internally
    const kmeans = new KMeansModernized(k)
    kmeans.fit(this.Observations)
    
    // Return assignments wrapped in Sylvester-like Vector for backwards compatibility
    // VectorLike handles 0->1 index conversion lazily
    return new VectorLike(kmeans.getAssignments())
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
    const tempKmeans = new KMeansModernized(k, { initialization: 'random' })
    
    return tempKmeans.initializeRandomCentroids(this.Observations, rng)
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
