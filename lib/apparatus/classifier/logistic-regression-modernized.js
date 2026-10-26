/*
Copyright (c) 2011, Chris Umbel

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

const Classifier = require('./classifier')

/**
 * Sigmoid function
 */
function sigmoid (z) {
  return 1 / (1 + Math.exp(-z))
}

/**
 * Dot product of two vectors (arrays)
 */
function dotProduct (a, b) {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i]
  }
  return sum
}

/**
 * Matrix-vector multiplication
 * matrix: array of arrays (rows x cols)
 * vector: array (cols)
 * returns: array (rows)
 */
function matrixVectorMultiply (matrix, vector) {
  const result = new Array(matrix.length)
  for (let i = 0; i < matrix.length; i++) {
    result[i] = dotProduct(matrix[i], vector)
  }
  return result
}

/**
 * Matrix transpose
 * matrix: array of arrays (rows x cols)
 * returns: array of arrays (cols x rows)
 */
function matrixTranspose (matrix) {
  if (matrix.length === 0) return []

  const rows = matrix.length
  const cols = matrix[0].length
  const result = Array(cols)

  for (let j = 0; j < cols; j++) {
    result[j] = new Array(rows)
    for (let i = 0; i < rows; i++) {
      result[j][i] = matrix[i][j]
    }
  }
  return result
}

/**
 * Element-wise operations
 */
function elementWiseSubtract (a, b) {
  const result = new Array(a.length)
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] - b[i]
  }
  return result
}

function elementWiseMultiply (a, b) {
  const result = new Array(a.length)
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] * b[i]
  }
  return result
}

function elementWiseLog (a) {
  const result = new Array(a.length)
  for (let i = 0; i < a.length; i++) {
    result[i] = Math.log(a[i])
  }
  return result
}

function elementWiseApply (a, fn) {
  const result = new Array(a.length)
  for (let i = 0; i < a.length; i++) {
    result[i] = fn(a[i])
  }
  return result
}

/**
 * Sum of array elements
 */
function sum (arr) {
  let total = 0
  for (let i = 0; i < arr.length; i++) {
    total += arr[i]
  }
  return total
}

/**
 * Scalar multiply
 */
function scalarMultiply (arr, scalar) {
  const result = new Array(arr.length)
  for (let i = 0; i < arr.length; i++) {
    result[i] = arr[i] * scalar
  }
  return result
}

/**
 * Augment matrix with ones column
 */
function augmentWithOnes (matrix) {
  const result = new Array(matrix.length)
  for (let i = 0; i < matrix.length; i++) {
    result[i] = [1, ...matrix[i]]
  }
  return result
}

/**
 * Cost function
 */
function cost (theta, Examples, classifications) {
  const hypothesisResult = matrixVectorMultiply(Examples, theta)
  const sigmoidResult = elementWiseApply(hypothesisResult, sigmoid)

  const numExamples = Examples.length
  const ones = Array(numExamples).fill(1)

  // cost1 = (-classifications) .* log(sigmoidResult)
  const negClassifications = elementWiseMultiply(classifications, Array(numExamples).fill(-1))
  const cost1 = elementWiseMultiply(
    negClassifications,
    elementWiseLog(sigmoidResult)
  )

  // cost0 = (1 - classifications) .* log(1 - sigmoidResult)
  const oneMinusHypothesis = elementWiseSubtract(ones, sigmoidResult)
  const oneMinusClassifications = elementWiseSubtract(ones, classifications)
  const cost0 = elementWiseMultiply(
    oneMinusClassifications,
    elementWiseLog(oneMinusHypothesis)
  )

  const totalCost = sum(elementWiseSubtract(cost1, cost0))
  return totalCost / numExamples
}

/**
 * Descending gradient function - trains the model
 * Optimized for speed with pre-computed matrix transpose
 */
function descendGradient (theta, Examples, classifications) {
  const maxIterPerRate = 100
  let learningRate = 1.0
  let currentTheta = theta.slice()
  let bestTheta = theta.slice()
  let bestCost = cost(currentTheta, Examples, classifications)
  let learningRateFound = false
  const m = Examples.length

  // Pre-compute transpose once - this is the key optimization!
  const ExamplesTransposed = matrixTranspose(Examples)

  while (!learningRateFound && learningRate > 0.0001) {
    let iterationCount = 0
    let lastCost = bestCost
    let improvementCount = 0

    while (iterationCount < maxIterPerRate) {
      // Compute hypothesis and error
      const hypothesisResult = matrixVectorMultiply(Examples, currentTheta)
      const sigmoidResult = elementWiseApply(hypothesisResult, sigmoid)

      // Gradient = X^T * (h(X) - y) / m
      const error = elementWiseSubtract(sigmoidResult, classifications)
      const gradient = matrixVectorMultiply(ExamplesTransposed, error)
      const scaledGradient = scalarMultiply(gradient, learningRate / m)

      // Update theta
      currentTheta = elementWiseSubtract(currentTheta, scaledGradient)

      // Evaluate cost
      const currentCost = cost(currentTheta, Examples, classifications)

      if (currentCost < bestCost) {
        bestCost = currentCost
        bestTheta = currentTheta.slice()
        improvementCount++
      }

      // Check for convergence with improved logic
      if (lastCost - currentCost < 0.00001) {
        learningRateFound = true
        break
      }

      lastCost = currentCost
      iterationCount++
    }

    // If we made progress, accept this learning rate
    if (improvementCount > maxIterPerRate * 0.1) {
      learningRateFound = true
    } else {
      // Try smaller learning rate
      learningRate *= 0.5
      currentTheta = bestTheta.slice() // Reset to best known theta
    }
  }

  return bestTheta.slice(1) // Remove augmented 0 at the beginning
}

/**
 * Logistic Regression Classifier - Modernized version
 */
class LogisticRegressionModernized extends Classifier {
  constructor () {
    super()
    this.examples = {}
    this.features = []
    this.featurePositions = {}
    this.maxFeaturePosition = 0
    this.classifications = []
    this.exampleCount = 0
    this.theta = []
  }

  /**
   * Create classifications matrix
   */
  createClassifications () {
    const classifications = []

    for (let i = 0; i < this.exampleCount; i++) {
      const classification = []

      Object.keys(this.examples).forEach(() => {
        classification.push(0)
      })

      classifications.push(classification)
    }

    return classifications
  }

  /**
   * Compute theta parameters for each class
   */
  computeThetas (Examples, Classifications) {
    this.theta = []

    // each class will have its own theta
    const zeroVector = new Array(Examples[0].length).fill(0)

    for (let i = 0; i < this.classifications.length; i++) {
      // Extract column i from Classifications
      const classColumn = new Array(Classifications.length)
      for (let j = 0; j < Classifications.length; j++) {
        classColumn[j] = Classifications[j][i]
      }

      this.theta.push(descendGradient(zeroVector, Examples, classColumn))
    }
  }

  /**
   * Train the classifier
   */
  train () {
    const examples = []
    const classifications = this.createClassifications()
    let d = 0
    let c = 0

    for (const classification in this.examples) {
      for (let i = 0; i < this.examples[classification].length; i++) {
        const doc = this.examples[classification][i]
        examples.push(doc)
        classifications[d][c] = 1
        d++
      }

      c++
    }

    const augmentedExamples = augmentWithOnes(examples)
    this.computeThetas(augmentedExamples, classifications)
  }

  /**
   * Add example to training set
   */
  addExample (data, classification) {
    if (!this.examples[classification]) {
      this.examples[classification] = []
      this.classifications.push(classification)
    }

    this.examples[classification].push(data)
    this.exampleCount++
  }

  /**
   * Get classifications for an observation
   */
  getClassifications (observation) {
    const classifications = []

    for (let i = 0; i < this.theta.length; i++) {
      const score = dotProduct(observation, this.theta[i])
      classifications.push({
        label: this.classifications[i],
        value: sigmoid(score)
      })
    }

    return classifications.sort(function (x, y) {
      return y.value - x.value
    })
  }

  /**
   * Restore classifier from JSON
   */
  static restore (classifier) {
    classifier = Classifier.restore(classifier)
    Object.setPrototypeOf(classifier, LogisticRegressionModernized.prototype)

    return classifier
  }
}

module.exports = LogisticRegressionModernized
