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

const sylvester = require('sylvester')
const Matrix = sylvester.Matrix
const Vector = sylvester.Vector

function sigmoid (z) {
  return 1 / (1 + Math.exp(0 - z))
}

function hypothesis (theta, Observations) {
  return Observations.x(theta).map(sigmoid)
}

function cost (theta, Examples, classifications) {
  const hypothesisResult = hypothesis(theta, Examples)

  const ones = Vector.One(Examples.rows())
  const cost1 = Vector.Zero(Examples.rows()).subtract(classifications).elementMultiply(hypothesisResult.log())
  const cost0 = ones.subtract(classifications).elementMultiply(ones.subtract(hypothesisResult).log())

  return (1 / Examples.rows()) * cost1.subtract(cost0).sum()
}

function descendGradient (theta, Examples, classifications) {
  const maxIt = 500 * Examples.rows()
  let last
  let current
  let learningRate = 3
  let learningRateFound = false

  Examples = Matrix.One(Examples.rows(), 1).augment(Examples)
  theta = theta.augment([0])

  while (!learningRateFound && learningRate !== 0) {
    let i = 0
    last = null

    while (true) {
      const hypothesisResult = hypothesis(theta, Examples)
      theta = theta.subtract(Examples.transpose().x(
        hypothesisResult.subtract(classifications)).x(1 / Examples.rows()).x(learningRate))
      current = cost(theta, Examples, classifications)

      i++

      if (last) {
        if (current < last) { learningRateFound = true } else { break }

        if (last - current < 0.0001) { break }
      }

      if (i >= maxIt) {
        throw new Error('unable to find minimum')
      }

      last = current
    }

    learningRate /= 3
  }

  return theta.chomp(1)
}

class LogisticRegressionClassifier extends Classifier {
  constructor () {
    super()
    this.examples = {}
    this.features = []
    this.featurePositions = {}
    this.maxFeaturePosition = 0
    this.classifications = []
    this.exampleCount = 0
  }

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

  computeThetas (Examples, Classifications) {
    this.theta = []

    // each class will have it's own theta.
    const zero = function () { return 0 }
    for (let i = 1; i <= this.classifications.length; i++) {
      const theta = Examples.row(1).map(zero)
      this.theta.push(descendGradient(theta, Examples, Classifications.column(i)))
    }
  }

  train () {
    const examples = []
    const classifications = this.createClassifications()
    let d = 0; let c = 0

    for (const classification in this.examples) {
      for (let i = 0; i < this.examples[classification].length; i++) {
        const doc = this.examples[classification][i]
        const example = doc

        examples.push(example)
        classifications[d][c] = 1
        d++
      }

      c++
    }

    this.computeThetas(Matrix.create(examples), Matrix.create(classifications))
  }

  addExample (data, classification) {
    if (!this.examples[classification]) {
      this.examples[classification] = []
      this.classifications.push(classification)
    }

    this.examples[classification].push(data)
    this.exampleCount++
  }

  getClassifications (observation) {
    observation = Vector.create(observation)
    const classifications = []

    for (let i = 0; i < this.theta.length; i++) {
      classifications.push({ label: this.classifications[i], value: sigmoid(observation.dot(this.theta[i])) })
    }

    return classifications.sort(function (x, y) {
      return y.value - x.value
    })
  }

  static restore (classifier) {
    classifier = Classifier.restore(classifier)
    Object.setPrototypeOf(classifier, LogisticRegressionClassifier.prototype)

    return classifier
  }
}

module.exports = LogisticRegressionClassifier
