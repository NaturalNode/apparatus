/*
Example usage of KMeans with KMeansVisualizer for event-based visualization
*/

var KMeans = require('./kmeans');
var KMeansVisualizer = require('./kmeans_visualizer');

// Example: Basic clustering with logging
function exampleBasicVisualization() {
    // Sample observations (4 points with 2 features each)
    var observations = [
        [1, 2],
        [1.5, 1.8],
        [5, 8],
        [8, 8],
        [1, 0.6],
        [9, 11]
    ];
    
    // Create KMeans instance
    var kmeans = new KMeans(observations);
    
    // Create visualizer with logging options
    var visualizer = new KMeansVisualizer({
        logStart: true,       // Log when clustering starts
        logIterations: true,  // Log each iteration
        logComplete: true     // Log when clustering completes
    });
    
    // Attach visualizer to listen for events
    visualizer.attach(kmeans);
    
    // Run clustering with 2 clusters
    var groups = kmeans.cluster(2);
    
    // Access iteration history
    var history = visualizer.getIterationHistory();
    console.log('\nTotal iterations recorded:', history.length);
}

// Example: Custom event handling
function exampleCustomHandling() {
    var observations = [[1, 2], [1.5, 1.8], [5, 8], [8, 8]];
    var kmeans = new KMeans(observations);
    
    var visualizer = new KMeansVisualizer({
        onIteration: function(data) {
            // Custom handling per iteration
            console.log('Iteration', data.iteration, '- Centroid positions updated');
        },
        onComplete: function(data) {
            // Custom handling on completion
            console.log('Clustering done in', data.iterations, 'iterations');
        }
    });
    
    visualizer.attach(kmeans);
    kmeans.cluster(2);
}

// Example: Export data for external visualization
function exampleExportData() {
    var observations = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
    var kmeans = new KMeans(observations);
    
    var visualizer = new KMeansVisualizer();
    visualizer.attach(kmeans);
    kmeans.cluster(2);
    
    // Export all data to JSON
    var exportedData = visualizer.exportToJSON();
    console.log(JSON.stringify(exportedData, null, 2));
    
    // Or save to file:
    // var fs = require('fs');
    // fs.writeFileSync('clustering_data.json', JSON.stringify(exportedData, null, 2));
}

// Example: Track centroid movement
function exampleTrajectories() {
    var observations = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
    var kmeans = new KMeans(observations);
    
    var visualizer = new KMeansVisualizer({
        logIterations: true
    });
    
    visualizer.attach(kmeans);
    kmeans.cluster(2);
    
    // Get centroid trajectories
    var trajectories = visualizer.getCentroidTrajectory();
    console.log('\nCentroid movements:');
    Object.keys(trajectories).forEach(function(clusterIdx) {
        console.log('Cluster', clusterIdx, ':', trajectories[clusterIdx]);
    });
}

// Example: 2D terminal visualization
function exampleTerminalVisualization() {
    var observations = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
    var kmeans = new KMeans(observations);
    
    var visualizer = new KMeansVisualizer();
    visualizer.attach(kmeans);
    kmeans.cluster(2);
    
    // Simple 2D visualization in terminal
    visualizer.visualize2DTerminal(30, 15);
}

// Example: 2D visualization at each iteration
function exampleIterationVisualization() {
    var observations = [
        [1, 2],
        [1.5, 1.8],
        [5, 8],
        [8, 8],
        [1, 0.6],
        [9, 11]
    ];
    
    var kmeans = new KMeans(observations);
    var visualizer = new KMeansVisualizer();
    visualizer.attach(kmeans);
    
    // Run clustering
    kmeans.cluster(2);
    
    // Show initial state
    console.log('\n========== INITIAL STATE ==========');
    visualizer.visualizeIteration(0, 40, 15);
    
    // Show each iteration
    var history = visualizer.getIterationHistory();
    history.forEach(function(iterData, idx) {
        console.log('\n========== ITERATION', idx + 1, '==========');
        visualizer.visualizeIteration(idx + 1, 40, 15);
    });
    
    // Show final state
    console.log('\n========== FINAL STATE ==========');
    visualizer.visualize2DTerminal(40, 15);
}

// Uncomment one of the examples to run:
// exampleBasicVisualization();
// exampleCustomHandling();
// exampleExportData();
// exampleTrajectories();
// exampleTerminalVisualization();
exampleIterationVisualization();
