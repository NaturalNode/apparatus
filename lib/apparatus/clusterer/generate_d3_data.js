/*
Example: Generate K-Means data and export for D3.js visualization
Run this to create a JSON file, then paste the contents into the HTML visualization
*/

var KMeans = require('./kmeans');
var KMeansVisualizer = require('./kmeans_visualizer');
var fs = require('fs');

function generateVisualizationData() {
    // Create sample observations
    var observations = [
        [1, 2],
        [1.5, 1.8],
        [5, 8],
        [8, 8],
        [1, 0.6],
        [9, 11],
        [8.5, 9],
        [1.2, 1.5],
        [6, 9],
        [0.8, 2.2]
    ];
    
    console.log('Running K-Means clustering...');
    
    // Create KMeans instance
    var kmeans = new KMeans(observations);
    
    // Create visualizer
    var visualizer = new KMeansVisualizer({
        logStart: true,
        logIterations: true,
        logComplete: true
    });
    
    // Attach visualizer
    visualizer.attach(kmeans);
    
    // Run clustering with 2 clusters
    var groups = kmeans.cluster(2);
    
    // Export data
    var exportedData = visualizer.exportToJSON();
    
    // Save to file
    var filename = 'kmeans_data.json';
    fs.writeFileSync(filename, JSON.stringify(exportedData, null, 2));
    
    console.log('\n✓ Data exported to:', filename);
    console.log('✓ Total iterations:', exportedData.complete.iterations);
    console.log('\nNext steps:');
    console.log('1. Open kmeans_d3_visualization.html in your browser');
    console.log('2. Copy the contents of', filename);
    console.log('3. Paste into the textarea and click "Load Data"');
    console.log('4. Click "Play" to see the animation!\n');
    
    // Also print to console for quick copy-paste
    console.log('Or copy this JSON directly:\n');
    console.log(JSON.stringify(exportedData, null, 2));
}

// Run the example
generateVisualizationData();
