
var PCA = new require('../lib/apparatus/util/pca');

describe('pca', function() {
    it('should pca', function() {
	PCA.pca($M([[1, 2], [5, 7]]));
    });

    it('should reduce', function() {
	expect(PCA.project($M([[1, 2], [5, 7]]), 1).eql($M([
	    [-2.2120098720461616],
	    [-8.601913944732665]
	])).toBeTruthy);
    });

    it('should recover', function() {
	var U = $M([[-0.5732529283807336, -0.819378471832714],
		    [-0.819378471832714, 0.5732529283807336]]);
	var Z = $M([[-2.2120098720461616],
		   [-8.601913944732665]]);
	expect(PCA.recover(Z, 1, U).eql($M([
	    [1.268041136757554, 1.812473268636061],
	    [4.931072358497068, 7.048223102871564]
	]))).toBeTruthy();
    });
});
