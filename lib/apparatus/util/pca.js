
var Sylvester = require('sylvester');

function pca(X) {
    var Sigma = X.transpose().x(X).x(1 / X.rows());
    var svd = Sigma.svd();
    return {U: svd.U, S: svd.S};
}

function project(X, k, U) {
    var U = U || pca(X).U;
    var Ureduce = U.slice(1, U.rows(), 1, k);

    return X.x(Ureduce);
}

function recover(Z, k, U) {
    var Ureduce = U.slice(1, U.rows(), 1, k);
    return Z.x(Ureduce.transpose());
}

exports.pca = pca;
exports.project = project;
exports.recover = recover;