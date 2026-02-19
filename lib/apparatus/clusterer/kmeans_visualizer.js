/*
KMeans Visualizer - Event-based visualization helper for K-means clustering
Listens to events emitted by KMeans and provides visualization/logging capabilities
*/

function KMeansVisualizer(options) {
    this.options = options || {};
    this.iterations = [];
    this.startData = null;
    this.completeData = null;
}

// Attach listener to a KMeans instance
KMeansVisualizer.prototype.attach = function(kmeansInstance) {
    var self = this;
    
    kmeansInstance.on('start', function(data) {
        self.onStart(data);
    });
    
    kmeansInstance.on('iteration', function(data) {
        self.onIteration(data);
    });
    
    kmeansInstance.on('complete', function(data) {
        self.onComplete(data);
    });
};

// Handle start event
KMeansVisualizer.prototype.onStart = function(data) {
    this.startData = data;
    if (this.options.logStart) {
        console.log('\n=== K-Means Clustering Started ===');
        console.log('Number of clusters (k):', data.k);
        console.log('Number of observations:', data.observations.length);
        console.log('Initial centroids:', data.initialCentroids);
    }
};

// Handle iteration event
KMeansVisualizer.prototype.onIteration = function(data) {
    this.iterations.push(data);
    
    if (this.options.logIterations) {
        console.log('\nIteration', data.iteration);
        console.log('Centroids:', data.centroids);
        console.log('Groups:', data.groups);
    }
    
    if (this.options.onIteration) {
        this.options.onIteration(data);
    }
};

// Handle complete event
KMeansVisualizer.prototype.onComplete = function(data) {
    this.completeData = data;
    
    if (this.options.logComplete) {
        console.log('\n=== Clustering Complete ===');
        console.log('Total iterations:', data.iterations);
        console.log('Final centroids:', data.finalCentroids);
        console.log('Cluster assignments:', data.groups);
    }
    
    if (this.options.onComplete) {
        this.options.onComplete(data);
    }
};

// Get iteration history
KMeansVisualizer.prototype.getIterationHistory = function() {
    return this.iterations;
};

// Get centroid movement (for 2D visualization)
KMeansVisualizer.prototype.getCentroidTrajectory = function() {
    var trajectories = {};
    
    if (this.startData) {
        this.startData.initialCentroids.forEach(function(centroid, idx) {
            trajectories[idx] = [centroid];
        });
    }
    
    this.iterations.forEach(function(iterData) {
        iterData.centroids.forEach(function(centroid, idx) {
            if (!trajectories[idx]) {
                trajectories[idx] = [];
            }
            trajectories[idx].push(centroid);
        });
    });
    
    return trajectories;
};

// Get cluster sizes over iterations
KMeansVisualizer.prototype.getClusterSizes = function() {
    var sizes = [];
    
    this.iterations.forEach(function(iterData) {
        var clusterSizes = {};
        iterData.groups.forEach(function(group) {
            clusterSizes[group] = (clusterSizes[group] || 0) + 1;
        });
        sizes.push(clusterSizes);
    });
    
    return sizes;
};

// Export as JSON for external visualization tools
KMeansVisualizer.prototype.exportToJSON = function() {
    return {
        start: this.startData,
        iterations: this.iterations,
        complete: this.completeData,
        summary: {
            totalIterations: this.completeData ? this.completeData.iterations : 0,
            iterationCount: this.iterations.length
        }
    };
};

// Simple 2D terminal visualization (for 2D data only)
KMeansVisualizer.prototype.visualize2DTerminal = function(width, height) {
    if (!this.completeData) {
        console.log('Clustering not complete');
        return;
    }
    
    console.log('\n=== 2D Visualization ===');
    console.log('(Note: This is a simplified text visualization)');
    
    var observations = this.startData.observations;
    var centroids = this.completeData.finalCentroids;
    
    // Find min/max values to normalize coordinates
    var minX = Infinity, maxX = -Infinity;
    var minY = Infinity, maxY = -Infinity;
    
    observations.forEach(function(obs) {
        if (obs.length >= 2) {
            if (obs[0] < minX) minX = obs[0];
            if (obs[0] > maxX) maxX = obs[0];
            if (obs[1] < minY) minY = obs[1];
            if (obs[1] > maxY) maxY = obs[1];
        }
    });
    
    centroids.forEach(function(c) {
        if (c.length >= 2) {
            if (c[0] < minX) minX = c[0];
            if (c[0] > maxX) maxX = c[0];
            if (c[1] < minY) minY = c[1];
            if (c[1] > maxY) maxY = c[1];
        }
    });
    
    // Add padding
    var padX = (maxX - minX) * 0.1 || 1;
    var padY = (maxY - minY) * 0.1 || 1;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;
    
    var rangeX = maxX - minX;
    var rangeY = maxY - minY;
    
    // Create canvas
    var canvas = [];
    for (var i = 0; i < height; i++) {
        canvas[i] = new Array(width).fill(' ');
    }
    
    // Helper function to convert data coordinates to canvas coordinates
    function toCanvasCoords(dataX, dataY) {
        var x = Math.floor(((dataX - minX) / rangeX) * (width - 1));
        var y = Math.floor((1 - (dataY - minY) / rangeY) * (height - 1)); // Flip Y axis
        return { x: x, y: y };
    }
    
    // Plot observations
    observations.forEach(function(obs, idx) {
        if (obs.length >= 2) {
            var coords = toCanvasCoords(obs[0], obs[1]);
            if (coords.x >= 0 && coords.x < width && coords.y >= 0 && coords.y < height) {
                canvas[coords.y][coords.x] = 'o';
            }
        }
    });
    
    // Plot centroids (overwrites observations if overlapping)
    centroids.forEach(function(centroid, idx) {
        if (centroid.length >= 2) {
            var coords = toCanvasCoords(centroid[0], centroid[1]);
            if (coords.x >= 0 && coords.x < width && coords.y >= 0 && coords.y < height) {
                canvas[coords.y][coords.x] = (idx + 1).toString();
            }
        }
    });
    
    // Print canvas
    console.log('Legend: o=observation, 1,2,3...=centroids');
    console.log('+' + new Array(width).fill('-').join('') + '+');
    canvas.forEach(function(row) {
        console.log('|' + row.join('') + '|');
    });
    console.log('+' + new Array(width).fill('-').join('') + '+');
    console.log('X range: [' + minX.toFixed(2) + ', ' + maxX.toFixed(2) + ']');
    console.log('Y range: [' + minY.toFixed(2) + ', ' + maxY.toFixed(2) + ']');
};

// Visualize a specific iteration's state
KMeansVisualizer.prototype.visualizeIteration = function(iterationNumber, width, height) {
    if (iterationNumber > this.iterations.length) {
        console.log('Iteration', iterationNumber, 'not found (max:', this.iterations.length, ')');
        return;
    }
    
    var iterData = iterationNumber > 0 ? this.iterations[iterationNumber - 1] : null;
    var observations = this.startData.observations;
    var centroids = iterData ? iterData.centroids : this.startData.initialCentroids;
    var groups = iterData ? iterData.groups : null;
    
    console.log('\n=== 2D Visualization - Iteration', iterationNumber || 'Initial', '===');
    
    // Find min/max values
    var minX = Infinity, maxX = -Infinity;
    var minY = Infinity, maxY = -Infinity;
    
    observations.forEach(function(obs) {
        if (obs.length >= 2) {
            if (obs[0] < minX) minX = obs[0];
            if (obs[0] > maxX) maxX = obs[0];
            if (obs[1] < minY) minY = obs[1];
            if (obs[1] > maxY) maxY = obs[1];
        }
    });
    
    centroids.forEach(function(c) {
        if (c.length >= 2) {
            if (c[0] < minX) minX = c[0];
            if (c[0] > maxX) maxX = c[0];
            if (c[1] < minY) minY = c[1];
            if (c[1] > maxY) maxY = c[1];
        }
    });
    
    var padX = (maxX - minX) * 0.1 || 1;
    var padY = (maxY - minY) * 0.1 || 1;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;
    
    var rangeX = maxX - minX;
    var rangeY = maxY - minY;
    
    var canvas = [];
    for (var i = 0; i < height; i++) {
        canvas[i] = new Array(width).fill(' ');
    }
    
    function toCanvasCoords(dataX, dataY) {
        var x = Math.floor(((dataX - minX) / rangeX) * (width - 1));
        var y = Math.floor((1 - (dataY - minY) / rangeY) * (height - 1));
        return { x: x, y: y };
    }
    
    // Plot observations with cluster coloring if available
    observations.forEach(function(obs, idx) {
        if (obs.length >= 2) {
            var coords = toCanvasCoords(obs[0], obs[1]);
            if (coords.x >= 0 && coords.x < width && coords.y >= 0 && coords.y < height) {
                if (groups) {
                    // Show cluster assignment as a number
                    canvas[coords.y][coords.x] = groups[idx] ? groups[idx].toString() : 'o';
                } else {
                    canvas[coords.y][coords.x] = 'o';
                }
            }
        }
    });
    
    // Plot centroids
    centroids.forEach(function(centroid, idx) {
        if (centroid.length >= 2) {
            var coords = toCanvasCoords(centroid[0], centroid[1]);
            if (coords.x >= 0 && coords.x < width && coords.y >= 0 && coords.y < height) {
                canvas[coords.y][coords.x] = String.fromCharCode(65 + idx); // A, B, C...
            }
        }
    });
    
    console.log('Legend: A,B,C...=centroids, 1,2,3...=cluster assignments');
    console.log('+' + new Array(width).fill('-').join('') + '+');
    canvas.forEach(function(row) {
        console.log('|' + row.join('') + '|');
    });
    console.log('+' + new Array(width).fill('-').join('') + '+');
};

module.exports = KMeansVisualizer;
