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
Classifier = require('./classifier'),
Sylvester = require('sylvester'),
Vector = Sylvester.Vector;

var BayesClassifier = function() {
    Classifier.call(this);
    this.examples = [];
    this.exampleClassifications = [];
    this.classes = {};
    this.classNames = [];
};

sys.inherits(BayesClassifier, Classifier);

function addExample(observation, classification) {
    this.examples.push(observation);

    if(this.classes[classification])
	this.classes[classification].count++;
    else {
	var index = this.classNames.indexOf(classification);

	if(index < 0)
	    index = this.classNames.push(classification) - 1;

	this.classes[classification] = {
	    count: 1,
	    index: index 
	};
    }

    this.exampleClassifications.push(index);
}

function train() {
    this.Examples = $M(this.examples).transpose();
    this.classifications = $V(this.exampleClassifications);
}

function probabilityOfClass(observation, klass) {
    var classSize = klass.count;
    
    var probOfClass = classSize / this.classifications.cols();
    var classMask = this.classifications.map(function(v, i) {
        if(v == klass.index)
            return 1;
        else
            return 0;
    });

    var probElementsAreClass = this.Examples.x(classMask).x(1 / classSize);
    var probElementsAreNotClass = this.Examples.x(Vector.One(
        this.classifications.cols()).subtract(classMask)).x(
            1 / (this.classifications.cols() - classSize));
    var probExampleElementsAreClass = probElementsAreClass.elementMultiply(observation);
    var probExampleElementsAreNotClass = probElementsAreNotClass.elementMultiply(observation);

    return probExampleElementsAreClass.x(probOfClass).add(1).elementDivide(
	probExampleElementsAreClass.add(probExampleElementsAreNotClass).add(2)
    ).product();
    
}

function getClassifications(observation) {
    var classifier = this;
    var classifications = [];

    for(className in this.classes) {
	var klass = this.classes[className];
	classifications.push({name: className,
			      value: classifier.probabilityOfClass(observation, klass)});
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