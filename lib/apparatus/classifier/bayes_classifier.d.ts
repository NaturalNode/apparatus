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
 * Bayes Classifier
 * 
 * Naive Bayes classifier implementation
 */
export class BayesClassifier extends Classifier {
  /**
   * Initialize with optional smoothing parameter
   * @param smoothing - Smoothing constant (default: 1.0)
   */
  constructor (smoothing?: number)

  /**
   * Add an example to the training set
   * @param observation - Feature vector (array or sparse object)
   * @param label - Class label
   */
  addExample (observation: number[] | { [key: string]: any }, label: string): void

  /**
   * Train the classifier
   */
  train (): void

  /**
   * Get probability of a class given an observation
   * @param observation - Feature vector (array or sparse object)
   * @param label - Class label
   * @returns Log probability
   */
  probabilityOfClass (observation: number[] | { [key: string]: any }, label: string): number

  /**
   * Get classifications for an observation
   * @param observation - Feature vector (array or sparse object)
   * @returns Array of {label, value} pairs sorted by probability
   */
  getClassifications (observation: number[] | { [key: string]: any }): Classification[]

  /**
   * Restore classifier from JSON
   * @param classifier - JSON representation
   * @returns Restored classifier
   */
  static restore (classifier: any): BayesClassifier
}

export default BayesClassifier
