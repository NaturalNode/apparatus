/*
Comparison tests between LogisticRegressionClassifier (modernized) and LogisticRegressionClassifier-Legacy (Sylvester)
Verifies both implementations produce equivalent results
*/

'use strict'

const LogisticRegressionClassifier = require('../lib/apparatus/classifier/logistic_regression_classifier')
const LogisticRegressionClassifierLegacy = require('../lib/apparatus/classifier/logistic-regression-legacy')

describe('LogisticRegressionClassifier vs Legacy Comparison', () => {
  const trainingData = [
    { text: [1, 0, 1, 0], label: 'positive' },
    { text: [1, 1, 0, 0], label: 'positive' },
    { text: [0, 1, 1, 0], label: 'positive' },
    { text: [0, 0, 1, 1], label: 'negative' },
    { text: [0, 0, 0, 1], label: 'negative' },
    { text: [1, 0, 0, 1], label: 'negative' }
  ]

  it('should both train successfully on same data', () => {
    const modernClassifier = new LogisticRegressionClassifier()
    trainingData.forEach(item => {
      modernClassifier.addExample(item.text, item.label)
    })
    modernClassifier.train()

    const legacyClassifier = new LogisticRegressionClassifierLegacy()
    trainingData.forEach(item => {
      legacyClassifier.addExample(item.text, item.label)
    })
    legacyClassifier.train()

    // Both should have learned theta values
    expect(modernClassifier.theta).toBeDefined()
    expect(modernClassifier.theta.length).toBeGreaterThan(0)

    expect(legacyClassifier.theta).toBeDefined()
    expect(legacyClassifier.theta.length).toBeGreaterThan(0)
  })

  it('should both classify observations with consistent class labels', () => {
    const modernClassifier = new LogisticRegressionClassifier()
    trainingData.forEach(item => {
      modernClassifier.addExample(item.text, item.label)
    })
    modernClassifier.train()

    const legacyClassifier = new LogisticRegressionClassifierLegacy()
    trainingData.forEach(item => {
      legacyClassifier.addExample(item.text, item.label)
    })
    legacyClassifier.train()

    // Test observation
    const testObservation = [1, 0, 1, 0]

    const modernResult = modernClassifier.getClassifications(testObservation)
    const legacyResult = legacyClassifier.getClassifications(testObservation)

    // Both should return array of classifications
    expect(Array.isArray(modernResult)).toBe(true)
    expect(Array.isArray(legacyResult)).toBe(true)

    // Both should return results for same classes
    expect(modernResult.length).toBe(legacyResult.length)
    expect(modernResult.length).toBe(2) // positive and negative

    // Both should have label and value properties
    modernResult.forEach(result => {
      expect(result.label).toBeDefined()
      expect(result.value).toBeDefined()
      expect(typeof result.value).toBe('number')
      expect(result.value).toBeGreaterThanOrEqual(0)
      expect(result.value).toBeLessThanOrEqual(1)
    })

    legacyResult.forEach(result => {
      expect(result.label).toBeDefined()
      expect(result.value).toBeDefined()
      expect(typeof result.value).toBe('number')
    })
  })

  it('should return sorted results by confidence', () => {
    const modernClassifier = new LogisticRegressionClassifier()
    trainingData.forEach(item => {
      modernClassifier.addExample(item.text, item.label)
    })
    modernClassifier.train()

    const legacyClassifier = new LogisticRegressionClassifierLegacy()
    trainingData.forEach(item => {
      legacyClassifier.addExample(item.text, item.label)
    })
    legacyClassifier.train()

    const testObservation = [1, 0, 1, 0]

    const modernResult = modernClassifier.getClassifications(testObservation)
    const legacyResult = legacyClassifier.getClassifications(testObservation)

    // Both should return results sorted by value (descending)
    for (let i = 0; i < modernResult.length - 1; i++) {
      expect(modernResult[i].value).toBeGreaterThanOrEqual(modernResult[i + 1].value)
    }

    for (let i = 0; i < legacyResult.length - 1; i++) {
      expect(legacyResult[i].value).toBeGreaterThanOrEqual(legacyResult[i + 1].value)
    }
  })

  it('should have consistent class labels in both implementations', () => {
    const modernClassifier = new LogisticRegressionClassifier()
    trainingData.forEach(item => {
      modernClassifier.addExample(item.text, item.label)
    })
    modernClassifier.train()

    const legacyClassifier = new LogisticRegressionClassifierLegacy()
    trainingData.forEach(item => {
      legacyClassifier.addExample(item.text, item.label)
    })
    legacyClassifier.train()

    const testObservation = [1, 0, 1, 0]

    const modernResult = modernClassifier.getClassifications(testObservation)
    const legacyResult = legacyClassifier.getClassifications(testObservation)

    // Extract labels and sort for comparison
    const modernLabels = modernResult.map(r => r.label).sort()
    const legacyLabels = legacyResult.map(r => r.label).sort()

    expect(modernLabels).toEqual(legacyLabels)
  })

  it('should both classify training examples with reasonable confidence', () => {
    const modernClassifier = new LogisticRegressionClassifier()
    trainingData.forEach(item => {
      modernClassifier.addExample(item.text, item.label)
    })
    modernClassifier.train()

    const legacyClassifier = new LogisticRegressionClassifierLegacy()
    trainingData.forEach(item => {
      legacyClassifier.addExample(item.text, item.label)
    })
    legacyClassifier.train()

    // Test on a training example
    const positiveExample = trainingData[0].text

    const modernResult = modernClassifier.getClassifications(positiveExample)
    const legacyResult = legacyClassifier.getClassifications(positiveExample)

    // Top prediction should be the correct class
    expect(modernResult[0].label).toBe('positive')
    expect(legacyResult[0].label).toBe('positive')

    // Confidence should be reasonably high
    expect(modernResult[0].value).toBeGreaterThan(0.3)
    expect(legacyResult[0].value).toBeGreaterThan(0.3)
  })

  it('should produce results with value between 0 and 1 (sigmoid output)', () => {
    const modernClassifier = new LogisticRegressionClassifier()
    trainingData.forEach(item => {
      modernClassifier.addExample(item.text, item.label)
    })
    modernClassifier.train()

    const testObservation = [1, 1, 0, 0]
    const modernResult = modernClassifier.getClassifications(testObservation)

    modernResult.forEach(result => {
      expect(result.value).toBeGreaterThanOrEqual(0)
      expect(result.value).toBeLessThanOrEqual(1)
    })
  })

  it('should both handle multiple classes correctly', () => {
    const multiClassData = [
      { text: [1, 0, 0], label: 'class_a' },
      { text: [1, 1, 0], label: 'class_a' },
      { text: [0, 1, 0], label: 'class_b' },
      { text: [0, 1, 1], label: 'class_b' },
      { text: [0, 0, 1], label: 'class_c' },
      { text: [1, 0, 1], label: 'class_c' }
    ]

    const modernClassifier = new LogisticRegressionClassifier()
    multiClassData.forEach(item => {
      modernClassifier.addExample(item.text, item.label)
    })
    modernClassifier.train()

    const legacyClassifier = new LogisticRegressionClassifierLegacy()
    multiClassData.forEach(item => {
      legacyClassifier.addExample(item.text, item.label)
    })
    legacyClassifier.train()

    const testObservation = [1, 0, 0]

    const modernResult = modernClassifier.getClassifications(testObservation)
    const legacyResult = legacyClassifier.getClassifications(testObservation)

    // Both should return 3 class predictions
    expect(modernResult.length).toBe(3)
    expect(legacyResult.length).toBe(3)

    // Both should contain all three classes
    const modernLabels = modernResult.map(r => r.label).sort()
    const legacyLabels = legacyResult.map(r => r.label).sort()

    expect(modernLabels).toEqual(['class_a', 'class_b', 'class_c'])
    expect(legacyLabels).toEqual(['class_a', 'class_b', 'class_c'])
  })
})
