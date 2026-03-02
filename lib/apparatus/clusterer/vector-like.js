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

'use strict'

/**
 * VectorLike - Sylvester Vector API compatible wrapper
 * Makes arrays compatible with Sylvester Vector API without needing Sylvester
 * The legacy API used Sylvester Vectors which have .elements and .e(i)
 */
class VectorLike {
  /**
   * Create a VectorLike object
   * @param {Array<number>} zeroIndexedArray - 0-indexed array of cluster assignments
   */
  constructor (zeroIndexedArray) {
    // Store 0-indexed array internally for efficiency
    this._array = zeroIndexedArray
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
   * Getter for elements array
   * Converts from 0-indexed to 1-indexed for Sylvester compatibility
   */
  get elements () {
    return this._array.map(a => a + 1)
  }
}

module.exports = VectorLike
