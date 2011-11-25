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

var LogisticRegressionClassifier = new require('../lib/apparatus/classifier/logistic_regression_classifier');

describe('logistic regression', function() {    
    it('should classify with examples added in groups', function() {
        var logistic = new LogisticRegressionClassifier();
        logistic.addExample([1,1,1,0,0,0], 'one');
        logistic.addExample([1,0,1,0,0,0], 'one');
        logistic.addExample([1,1,1,0,0,0], 'one');
        logistic.addExample([0,0,0,1,1,1], 'two');
        logistic.addExample([0,0,0,1,0,1], 'two');
        logistic.addExample([0,0,0,1,1,0], 'two');

        logistic.train();
        
        expect(logistic.classify([0,1,1,0,0,0])).toBe('one');
        expect(logistic.classify([0,0,0,0,1,1])).toBe('two');
    });
});
