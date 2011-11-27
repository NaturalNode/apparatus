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

var sys = require('sys'),
Classifier = require('./classifier');

var BayesClassifier = function() {
    Classifier.call(this);
    this.classifications = {};
    this.classificationTotals = {};
    this.totalExamples = 0;
};

sys.inherits(BayesClassifier, Classifier);

function addExample(observation, classification) {
    var classifier = this;
    this.totalExamples++;
     
    if(!this.classifications[classification]) {
        this.classifications[classification] = {};
        this.classificationTotals[classification] = 0;
    }
     
    var i = observation.length;

    while(i--) {
	if(observation[i] > 0) {
            classifier.classificationTotals[classification]++;

            if(classifier.classifications[classification][i]) {
		classifier.classifications[classification][i]++;
            } else {
		classifier.classifications[classification][i] = 1;
            }
	}
    }
}

function train() {
    
}

function probabilityOfClass(observation, classification) {
    var probability = 0;
    var i = observation.length;

    while(i--) {
	if(observation[i]) {
            var count = this.classifications[classification][i] || 0;

            probability += Math.log((count + 1) / 
		    (this.classificationTotals[classification] + observation.length));
	}
    };

    probability = (this.classificationTotals[classification] / this.totalExamples) * Math.exp(probability);
    
    return probability;
}

function getClassifications(observation) {
    var classifier = this;
    var classifications = [];
    
    for(var className in this.classifications) {
	classifications.push({name: className,
			      value: classifier.probabilityOfClass(observation, className)});
    }
    
    return classifications.sort(function(x, y) {return y.value > x.value});
}

function load(filename, callback) {
     Classifier.load(filename, function(err, classifier) {
          callback(err, restore(classifier));
     });
}

function restore(classifier, stemmer) {
     classifier = Classifier.restore(classifier, stemmer);
     classifier.__proto__ = BayesClassifier.prototype;
     
     return classifier;
}

BayesClassifier.prototype.addExample = addExample;
BayesClassifier.prototype.train = train;
BayesClassifier.prototype.getClassifications = getClassifications;
BayesClassifier.prototype.probabilityOfClass = probabilityOfClass;

BayesClassifier.load = load;
BayesClassifier.restore = restore;

module.exports = BayesClassifier;