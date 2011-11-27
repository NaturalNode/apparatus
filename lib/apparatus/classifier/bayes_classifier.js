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
    this.classFeatures = {};
    this.classTotals = {};
    this.totalExamples = 0;
};

sys.inherits(BayesClassifier, Classifier);

function addExample(observation, label) {
    var classifier = this;
    this.totalExamples++;
     
    if(!this.classFeatures[label]) {
        this.classFeatures[label] = {};
        this.classTotals[label] = 0;
    }
     
    var i = observation.length;

    while(i--) {
	if(observation[i] > 0) {
            classifier.classTotals[label]++;

            if(classifier.classFeatures[label][i]) {
		classifier.classFeatures[label][i]++;
            } else {
		classifier.classFeatures[label][i] = 1;
            }
	}
    }
}

function train() {
    
}

function probabilityOfClass(observation, label) {
    var prob = 0;
    var i = observation.length;

    while(i--) {
	if(observation[i]) {
            var count = this.classFeatures[label][i] || 0;

            prob += Math.log((count + 1) / 
		    (this.classTotals[label] + observation.length));
	}
    };

    prob = (this.classTotals[label] / this.totalExamples) * Math.exp(prob);
    
    return prob;
}

function getClassifications(observation) {
    var classifier = this;
    var labels = [];
    
    for(var className in this.classFeatures) {
	labels.push({label: className,
	      value: classifier.probabilityOfClass(observation, className)});
    }
    
    return labels.sort(function(x, y) {return y.value > x.value});
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