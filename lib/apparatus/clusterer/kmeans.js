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

var Sylvester = require('sylvester'),
Matrix = Sylvester.Matrix,
Vector = Sylvester.Vector;

function KMeans(Observations) {
    this.Observations = Observations;
}

// create an initial centroid matrix with initial values between 
// 0 and the max of feature data X.
function createCentroid(k) {
    var Centroid = [];
    var max = this.Observations.max();

    for(var i = 1; i <= k; i++) {
	var centroid = [];
	
	for(var j = 1; j <= this.Observations.cols(); j++) {
	    centroid.push(max - (max / i));
	}

	Centroid.push(centroid);
    }
    
    return $M(Centroid);
}

// get the euclidian distance between the feature data X and
// a given centroid matrix C.
function distanceFrom(Centroid) {
    var Distance = [];

    for(var i = 1; i <= this.Observations.rows(); i++) {
	var distance = [];

	for(var j = 1; j <= Centroid.rows(); j++) {
	    distance.push(this.Observations.row(i).distanceFrom(Centroid.row(j)));
	}

	Distance.push(distance);
    }

    return $M(Distance);
}

// categorize the feature data X into k clusters. return a vector
// containing the results.
function cluster(k) {
    var Centroid = this.createCentroid(k);
    var LastDistance = Matrix.Zero(this.Observations.rows(), this.Observations.cols());
    var Distance = this.distanceFrom(Centroid);
    var Group;

    while(!(LastDistance.eql(Distance))) {
	Group = Distance.minColumnIndexes();
	LastDistance = Distance;

	var NewCentroid = [];

	for(var i = 1; i <= Centroid.rows(); i++) {
	    var centroid = [];

	    for(var j = 1; j <= Centroid.cols(); j++) {
		var sum = 0;
		var count = 0;

		for(var l = 1; l <= this.Observations.rows(); l++) {
		    if(Group.e(l) == i) {
			count++;
			sum += this.Observations.e(l, j);
		    }
		}

		centroid.push(sum / count);
	    }

	    NewCentroid.push(centroid);
	}
	
	Centroid = $M(NewCentroid);
	Distance = this.distanceFrom(Centroid);
    }

    return Group;
}

KMeans.prototype.createCentroid = createCentroid;
KMeans.prototype.distanceFrom = distanceFrom;
KMeans.prototype.cluster = cluster;

module.exports = KMeans;
