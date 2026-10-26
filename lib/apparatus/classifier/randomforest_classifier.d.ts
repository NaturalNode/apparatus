/*
Copyright (c) 2012 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

import { Classifier } from './classifier'

/**
 * Random Forest Classifier Options
 */
export interface RandomForestOptions {
  /**
   * Number of trees to train (default: 100)
   */
  numTrees?: number

  /**
   * Maximum depth of each tree in the forest (default: 4)
   */
  maxDepth?: number

  /**
   * Number of random hypotheses generated at each node during training (default: 10)
   */
  numTries?: number

  /**
   * Weak learner training function
   */
  trainFun?: (data: number[][], labels: number[], ix: number[], options: RandomForestOptions) => any

  /**
   * Weak learner test function
   */
  testFun?: (inst: number[], model: any) => number

  /**
   * Type of weak learner (0: decisionStump, 1: decision2DStump)
   */
  type?: number
}

/**
 * Random Forest Classifier
 * 
 * Ensemble classifier using multiple decision trees
 */
export class RandomForestClassifier extends Classifier {
  /**
   * Initialize with optional options
   * @param options - Configuration options for the forest
   */
  constructor (options?: RandomForestOptions)

  /**
   * Add an example to the training set
   * @param data - Feature vector
   * @param label - Classification label (typically 1 or -1 for binary classification)
   */
  addExample (data: number[], label: number): void

  /**
   * Train the classifier on added examples
   */
  train (): void

  /**
   * Classify an observation (returns the predicted label)
   * @param inst - Feature vector to classify
   * @returns Predicted label (1 or -1)
   */
  classify (inst: number[]): number

  /**
   * Get probability prediction for a single instance
   * @param inst - Feature vector to predict
   * @returns Probability/prediction value in range [0, 1]
   */
  predictOne (inst: number[]): number

  /**
   * Get probability predictions for multiple instances
   * @param data - Array of feature vectors
   * @returns Array of probability values
   */
  predict (data: number[][]): number[]
}

export default RandomForestClassifier
