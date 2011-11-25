
var Sylvester = require('sylvester'),
Matrix = Sylvester.Matrix,
Vector = Sylvester.Vector;

function KMeans(X) {
    this.X = X;
}

// create an initial centroid matrix with initial values between 
// 0 and the max of feature data X.
function createC(k) {
    var C = [];
    var max = this.X.max();

    for(var i = 1; i <= k; i++) {
	var c = [];
	
	for(var j = 1; j <= this.X.cols(); j++) {
	    c.push(max - (max / i));
	}

	C.push(c);
    }
    
    return $M(C);
}

// get the euclidian distance between the feature data X and
// a given centroid matrix C.
function distanceFrom(C) {
    var D = [];

    for(var i = 1; i <= this.X.rows(); i++) {
	var d = [];

	for(var j = 1; j <= C.rows(); j++) {
	    d.push(this.X.row(i).distanceFrom(C.row(j)));
	}

	D.push(d);
    }

    return $M(D);
}

// categorize the feature data X into k clusters. return a vector
// containing the results.
function cluster(k) {
    var C = this.createC(k);
    var LastD = Matrix.Zero(this.X.rows(), this.X.cols());
    var D = this.distanceFrom(C);
    var G;

    while(!(LastD.eql(D))) {
	G = D.minColumnIndexes();
	LastD = D;

	var NewC = [];

	for(var i = 1; i <= C.rows(); i++) {
	    var c = [];

	    for(var j = 1; j <= C.cols(); j++) {
		var sum = 0;
		var count = 0;

		for(var k = 1; k <= this.X.rows(); k++) {
		    if(G.e(k) == i) {
			count++;
			sum += this.X.e(k, j);
		    }
		}

		c.push(sum / count);
	    }

	    NewC.push(c);
	}
	
	C = $M(NewC);
	D = this.distanceFrom(C);
    }

    return G;
}

KMeans.prototype.createC = createC;
KMeans.prototype.distanceFrom = distanceFrom;
KMeans.prototype.cluster = cluster;

module.exports = KMeans;
