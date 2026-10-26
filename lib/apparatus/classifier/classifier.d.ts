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

/**
 * Classification result
 */
export interface Classification {
  label: string | number
  value: number
}

/**
 * Base Classifier class
 * 
 * This is the base class for all classifiers in Apparatus.
 * Subclasses must implement addExample() and train() methods.
 */
export class Classifier {
  /**
   * Add an example to the training set
   * @param observation - The feature vector (array or object)
   * @param classification - The label/class for this example
   */
  addExample (observation: any, classification: string | number): void

  /**
   * Train the classifier on the added examples
   */
  train (): void

  /**
   * Classify an observation
   * @param observation - The feature vector to classify
   * @returns The predicted class label
   */
  classify (observation: any): string | number

  /**
   * Get classifications for an observation with confidence scores
   * @param observation - The feature vector to classify
   * @returns Array of {label, value} pairs sorted by confidence
   */
  getClassifications (observation: any): Classification[]

  /**
   * Restore a classifier from JSON
   * @param classifier - JSON representation of classifier or JSON string
   * @returns Restored classifier instance
   */
  static restore (classifier: any): Classifier
}

export default Classifier
