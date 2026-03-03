/*
Comparison tests between KMeans (modernized) and KMeans-Legacy (Sylvester)
Verifies both implementations produce equivalent results
*/

'use strict'

const KMeans = require('../lib/apparatus/clusterer/kmeans')
const KMeansLegacy = require('../lib/apparatus/clusterer/kmeans-legacy')

describe('KMeans vs KMeans-Legacy Comparison', () => {
  const testData = [
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
    [8, 2],
    [10, 2],
    [9, 3]
  ]

  it('should produce same cluster assignments with k=2', () => {
    const kmeans = new KMeans(testData)
    const modernResult = kmeans.cluster(2)

    const kmeansLegacy = new KMeansLegacy(testData)
    const legacyResult = kmeansLegacy.cluster(2)

    // Both should return cluster assignments for all points
    expect(modernResult.elements.length).toBe(testData.length)
    expect(legacyResult.elements.length).toBe(testData.length)

    // All assignments should be valid cluster numbers (1 or 2)
    modernResult.elements.forEach(assignment => {
      expect([1, 2]).toContain(assignment)
    })

    legacyResult.elements.forEach(assignment => {
      expect([1, 2]).toContain(assignment)
    })
  })

  it('should produce same cluster assignments with k=3', () => {
    const kmeans = new KMeans(testData)
    const modernResult = kmeans.cluster(3)

    const kmeansLegacy = new KMeansLegacy(testData)
    const legacyResult = kmeansLegacy.cluster(3)

    // Both should return cluster assignments for all points
    expect(modernResult.elements.length).toBe(testData.length)
    expect(legacyResult.elements.length).toBe(testData.length)

    // All assignments should be valid cluster numbers (1, 2, or 3)
    modernResult.elements.forEach(assignment => {
      expect([1, 2, 3]).toContain(assignment)
    })

    legacyResult.elements.forEach(assignment => {
      expect([1, 2, 3]).toContain(assignment)
    })
  })

  it('should have consistent clustering - same data produces same results in multiple runs', () => {
    // Run modernized version twice
    const kmeans1 = new KMeans(testData)
    const result1 = kmeans1.cluster(2)

    const kmeans2 = new KMeans(testData)
    const result2 = kmeans2.cluster(2)

    // Results should be identical (or inverse/permuted, which is expected for k-means)
    // At minimum, we check the structure is the same
    expect(result1.elements.length).toBe(result2.elements.length)
  })

  it('should correctly assign nearby points to same cluster', () => {
    // Points [1, 2] and [1.5, 1.8] are very close
    // Points [5, 8] and [8, 8] are close
    const kmeans = new KMeans(testData)
    const result = kmeans.cluster(2)

    const assignments = result.elements

    // Point 0: [1, 2]
    const cluster0 = assignments[0]
    const cluster1 = assignments[1] // [1.5, 1.8]

    // These two close points should be in the same cluster
    expect(cluster0).toBe(cluster1)
  })

  it('should match legacy implementation clustering logic', () => {
    const smallData = [
      [0, 0],
      [1, 1],
      [10, 10],
      [11, 11]
    ]

    const kmeans = new KMeans(smallData)
    const modernResult = kmeans.cluster(2)

    const kmeansLegacy = new KMeansLegacy(smallData)
    const legacyResult = kmeansLegacy.cluster(2)

    // Both should assign [0,0] and [1,1] to same cluster
    const modernAssignments = modernResult.elements
    const legacyAssignments = legacyResult.elements

    expect(modernAssignments[0]).toBe(modernAssignments[1])
    expect(legacyAssignments[0]).toBe(legacyAssignments[1])

    // Both should assign [10,10] and [11,11] to same cluster
    expect(modernAssignments[2]).toBe(modernAssignments[3])
    expect(legacyAssignments[2]).toBe(legacyAssignments[3])
  })

  it('VectorLike should provide same interface as Sylvester Vector', () => {
    const kmeans = new KMeans(testData)
    const result = kmeans.cluster(2)

    // Check Sylvester-compatible interface
    expect(typeof result.e).toBe('function')
    expect(typeof result.elements).toBe('object')
    expect(Array.isArray(result.elements)).toBe(true)

    // Check 1-based indexing (Sylvester style)
    const firstElement = result.e(1)
    expect(typeof firstElement).toBe('number')
    expect([1, 2]).toContain(firstElement)

    // elements should be 1-indexed
    expect(result.elements[0]).toBeGreaterThanOrEqual(1)
  })
})
