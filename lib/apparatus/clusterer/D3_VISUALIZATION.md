# K-Means D3.js Visualization

Interactive web-based visualization of K-Means clustering using D3.js

## Files Created

- **kmeans_d3_visualization.html** - Interactive D3.js visualization page
- **generate_d3_data.js** - Script to generate clustering data for visualization

## How to Use

### Method 1: Quick Start with Sample Data

1. Open `kmeans_d3_visualization.html` in your web browser
2. Click the **"Load Sample Data"** button
3. Click **"Play"** to watch the clustering animation

### Method 2: Use Your Own Data

1. Run the data generation script:
   ```bash
   node generate_d3_data.js
   ```

2. This will:
   - Run K-Means clustering
   - Export the data to `kmeans_data.json`
   - Print the JSON to the console

3. Open `kmeans_d3_visualization.html` in your browser

4. Copy the JSON from the file or console output

5. Paste it into the textarea on the web page

6. Click **"Load Data"**

### Method 3: Generate Custom Data

Create your own script:

```javascript
var KMeans = require('./kmeans');
var KMeansVisualizer = require('./kmeans_visualizer');

// Your custom observations
var observations = [
    [1, 2],
    [5, 8],
    // ... more points
];

var kmeans = new KMeans(observations);
var visualizer = new KMeansVisualizer();
visualizer.attach(kmeans);

kmeans.cluster(3); // Number of clusters

// Get the data
var data = visualizer.exportToJSON();
console.log(JSON.stringify(data, null, 2));
```

Then paste the output into the visualization HTML.

## Features

### Interactive Controls

- **Play/Pause** - Animate through iterations automatically
- **Next/Previous** - Step through iterations manually
- **Reset** - Return to initial state
- **Slider** - Jump to any iteration

### Visualization Elements

- **Colored circles** - Observations, colored by cluster assignment
- **Black circles** - Centroids (labeled A, B, C, etc.)
- **Axes** - Scaled to fit your data
- **Legend** - Shows cluster colors and centroid markers

### What You'll See

1. **Initial State** - Random initial centroids
2. **Iterations** - Watch centroids move and clusters form
3. **Convergence** - See when the algorithm stabilizes

## Requirements

- Modern web browser with JavaScript enabled
- Internet connection (to load D3.js from CDN)
- Node.js (to run the data generation scripts)

## Customization

### Change Animation Speed

Edit the interval in `kmeans_d3_visualization.html`:

```javascript
playInterval = setInterval(() => {
    // ...
}, 1000); // Change 1000 to desired milliseconds
```

### Adjust Visualization Size

Modify the SVG dimensions:

```javascript
const width = 800;  // Change width
const height = 600; // Change height
```

## Troubleshooting

**"Invalid JSON data" error**
- Make sure you copied the complete JSON output
- Check for any syntax errors in the pasted data

**Nothing appears**
- Ensure D3.js loaded (check browser console)
- Verify your observations have 2 dimensions (for 2D visualization)

**Centroids overlap**
- This is normal with small datasets
- Try larger datasets or adjust the visualization size
