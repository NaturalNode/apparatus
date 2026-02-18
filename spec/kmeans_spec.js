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

var KMeans = require('.././lib/apparatus/clusterer/kmeans');

describe('kmeans', function() {
    it('should return cluster assignments for all observations', function() {
        var observations = [
            [1, 1],
            [2, 2],
            [10, 10],
            [11, 11]
        ];

        var kmeans = new KMeans(observations);
        var clusters = kmeans.cluster(2);

        // Verify we got cluster assignments for all points
        expect(clusters.elements.length).toBe(4);
        
        // All cluster assignments should be valid numbers (1 or 2)
        for (var i = 1; i <= 4; i++) {
            expect(clusters.e(i)).toBeGreaterThan(0);
            expect(clusters.e(i)).toBeLessThanOrEqual(2);
        }
    });

    it('should handle single cluster', function() {
        var observations = [
            [1, 1],
            [1.5, 1.5],
            [2, 2]
        ];

        var kmeans = new KMeans(observations);
        var clusters = kmeans.cluster(1);

        // All points should be in the same cluster
        expect(clusters.e(1)).toBe(1);
        expect(clusters.e(2)).toBe(1);
        expect(clusters.e(3)).toBe(1);
    });

    it('should cluster nearby points together', function() {
        var observations = [
            [0, 0],
            [0.1, 0.1],
            [0.2, 0.2],
            [100, 100],
            [100.1, 100.1],
            [100.2, 100.2]
        ];

        var kmeans = new KMeans(observations);
        var clusters = kmeans.cluster(2);

        // Points 1-3 should be in the same cluster (they're very close)
        var cluster1 = clusters.e(1);
        expect(clusters.e(2)).toBe(cluster1);
        expect(clusters.e(3)).toBe(cluster1);
        
        // Points 4-6 should be in the same cluster (they're very close)
        var cluster2 = clusters.e(4);
        expect(clusters.e(5)).toBe(cluster2);
        expect(clusters.e(6)).toBe(cluster2);
        
        // The two clusters should be different
        expect(cluster1).not.toBe(cluster2);
    });
});
