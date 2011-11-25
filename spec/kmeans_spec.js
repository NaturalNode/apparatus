
var KMeans = require('.././lib/apparatus/clusterer/kmeans');

describe('kmeans', function() {
    it('should perform binary clustering', function() {
	var X = $M([[1, 1], [2, 1], [4, 3], [5, 4]]);
	var k = 2;
	var kmeans = new KMeans(X, k);

	expect(kmeans.cluster(2).eql($V([1, 1, 2, 2]))).toBeTruthy();
    });

    it('should cluster', function() {
	var X = $M([[1, 1], [2, 1], [4, 3], [5, 4], [11, 12], [11, 13]]);
	var k = 3;
	var kmeans = new KMeans(X, k);

	expect(kmeans.cluster(3).eql($V([1, 1, 2, 2, 3, 3]))).toBeTruthy();
    });
});
