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

import { Classifier, Classification } from './classifier'

/**
 * Logistic Regression Classifier - Modernized Version
 * 
 * Supports multi-class logistic regression using gradient descent optimization
 */
export class LogisticRegressionModernized extends Classifier {
  /**
   * Initialize the classifier
   */
  constructor ()

  /**
   * Add an example to the training set
   * @param data - Feature vector
   * @param classification - Class label
   */
  addExample (data: number[], classification: string): void

  /**
   * Train the classifier on added examples
   */
  train (): void

  /**
   * Get classifications for an observation with confidence scores
   * @param observation - Feature vector to classify
   * @returns Array of {label, value} pairs sorted by confidence (sigmoid scores)
   */
  getClassifications (observation: number[]): Classification[]

  /**
   * Restore classifier from JSON
   * @param classifier - JSON representation
   * @returns Restored classifier
   */
  static restore (classifier: any): LogisticRegressionModernized
}

export default LogisticRegressionModernized
