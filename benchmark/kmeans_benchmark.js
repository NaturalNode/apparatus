/*
K-Means Performance Benchmark
Comparing legacy (Sylvester-based) vs modernized implementations
*/

const KMeansLegacy = require('../lib/apparatus/clusterer/kmeans-legacy');
const KMeansModernized = require('../lib/apparatus/clusterer/kmeans-modernized');

// Generate random dataset
function generateDataset(numPoints, dimensions) {
  const data = [];
  for (let i = 0; i < numPoints; i++) {
    const point = [];
    for (let j = 0; j < dimensions; j++) {
      point.push(Math.random() * 100);
    }
    data.push(point);
  }
  return data;
}

// Generate clustered dataset (more realistic)
function generateClusteredDataset(numClusters, pointsPerCluster, dimensions) {
  const data = [];
  
  for (let c = 0; c < numClusters; c++) {
    // Random cluster center
    const center = [];
    for (let d = 0; d < dimensions; d++) {
      center.push(Math.random() * 100);
    }
    
    // Generate points around center
    for (let p = 0; p < pointsPerCluster; p++) {
      const point = [];
      for (let d = 0; d < dimensions; d++) {
        point.push(center[d] + (Math.random() - 0.5) * 10);
      }
      data.push(point);
    }
  }
  
  return data;
}

// Benchmark function
function benchmark(name, fn, iterations = 1) {
  const start = process.hrtime.bigint();
  const startMem = process.memoryUsage().heapUsed;
  
  let result;
  for (let i = 0; i < iterations; i++) {
    result = fn();
  }
  
  const end = process.hrtime.bigint();
  const endMem = process.memoryUsage().heapUsed;
  
  const durationMs = Number(end - start) / 1000000 / iterations;
  const memoryDelta = (endMem - startMem) / 1024 / 1024;
  
  return {
    name,
    duration: durationMs,
    memory: memoryDelta,
    result
  };
}

// Run benchmark suite
function runBenchmark(datasetName, data, k, iterations = 5) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Benchmark: ${datasetName}`);
  console.log(`Dataset: ${data.length} points, ${data[0].length} dimensions, k=${k}`);
  console.log(`Iterations: ${iterations}`);
  console.log('='.repeat(60));
  
  // Warm up
  try {
    new KMeansLegacy(data).cluster(k);
  } catch (e) {
    console.log('Legacy warmup failed (expected if Sylvester not installed)');
  }
  
  const kmeansModern = new KMeansModernized(k);
  kmeansModern.fit(data);
  
  // Benchmark Legacy
  let legacyResult;
  try {
    legacyResult = benchmark('KMeans Legacy', () => {
      const kmeans = new KMeansLegacy(data);
      return kmeans.cluster(k);
    }, iterations);
  } catch (error) {
    legacyResult = { 
      name: 'KMeans Legacy', 
      duration: null, 
      memory: null,
      error: error.message 
    };
  }
  
  // Benchmark Modernized
  const modernizedResult = benchmark('KMeans Modernized', () => {
    const kmeans = new KMeansModernized(k, { maxIterations: 100 });
    kmeans.fit(data);
    return kmeans.getAssignments();
  }, iterations);
  
  // Results
  console.log('\nResults:');
  console.log('-'.repeat(60));
  
  if (legacyResult.error) {
    console.log(`Legacy:      ERROR - ${legacyResult.error}`);
  } else {
    console.log(`Legacy:      ${legacyResult.duration.toFixed(2)} ms, Memory: ${legacyResult.memory.toFixed(2)} MB`);
  }
  
  console.log(`Modernized:  ${modernizedResult.duration.toFixed(2)} ms, Memory: ${modernizedResult.memory.toFixed(2)} MB`);
  
  if (!legacyResult.error) {
    const speedup = (legacyResult.duration / modernizedResult.duration).toFixed(2);
    const faster = speedup > 1 ? 'Modernized' : 'Legacy';
    const ratio = speedup > 1 ? speedup : (1 / speedup).toFixed(2);
    
    const memDiff = modernizedResult.memory - legacyResult.memory;
    const memComparison = memDiff < 0 
      ? `${Math.abs(memDiff).toFixed(2)} MB less memory`
      : `${memDiff.toFixed(2)} MB more memory`;
    
    console.log(`\nComparison:  ${faster} is ${ratio}x faster, ${memComparison}`);
  }
  
  return { legacy: legacyResult, modernized: modernizedResult };
}

// Main benchmark suite
console.log('\n' + '='.repeat(60));
console.log('K-MEANS PERFORMANCE BENCHMARK');
console.log('='.repeat(60));

const results = [];

// Test 1: Random dataset (worst case - no natural clusters)
const random = generateDataset(100, 2);
results.push(runBenchmark('Random Dataset (100 points, 2D)', random, 5, 5));

// Test 2: Small dataset
const small = generateClusteredDataset(3, 10, 2);
results.push(runBenchmark('Small Clustered Dataset (30 points, 2D)', small, 3, 10));

// Test 3: Medium dataset
const medium = generateClusteredDataset(5, 50, 2);
results.push(runBenchmark('Medium Clustered Dataset (250 points, 2D)', medium, 5, 5));

// Test 4: Large dataset
const large = generateClusteredDataset(10, 100, 2);
results.push(runBenchmark('Large Clustered Dataset (1000 points, 2D)', large, 10, 3));

// Test 5: High dimensional
const highDim = generateClusteredDataset(5, 50, 10);
results.push(runBenchmark('High Dimensional Clustered (250 points, 10D)', highDim, 5, 5));

// Test 6: Very large dataset
const veryLarge = generateClusteredDataset(10, 500, 2);
results.push(runBenchmark('Very Large Clustered Dataset (5000 points, 2D)', veryLarge, 10, 1));

// Summary
console.log('\n' + '='.repeat(60));
console.log('SUMMARY');
console.log('='.repeat(60));

let modernizedWins = 0;
let totalSpeedup = 0;
let totalMemoryLegacy = 0;
let totalMemoryModernized = 0;
let validComparisons = 0;

results.forEach((r, i) => {
  if (!r.legacy.error && r.modernized.duration > 0) {
    const speedup = r.legacy.duration / r.modernized.duration;
    if (speedup > 1) {
      modernizedWins++;
    }
    totalSpeedup += speedup;
    totalMemoryLegacy += Math.abs(r.legacy.memory);
    totalMemoryModernized += Math.abs(r.modernized.memory);
    validComparisons++;
  }
});

if (validComparisons > 0) {
  const avgSpeedup = (totalSpeedup / validComparisons).toFixed(2);
  const avgMemLegacy = (totalMemoryLegacy / validComparisons).toFixed(2);
  const avgMemModernized = (totalMemoryModernized / validComparisons).toFixed(2);
  
  console.log(`\nPerformance:`);
  console.log(`  Modernized won ${modernizedWins}/${validComparisons} benchmarks`);
  console.log(`  Average speedup: ${avgSpeedup}x faster`);
  
  console.log(`\nMemory Usage (average absolute delta):`);
  console.log(`  Legacy:      ${avgMemLegacy} MB`);
  console.log(`  Modernized:  ${avgMemModernized} MB`);
  
  if (totalMemoryModernized < totalMemoryLegacy) {
    const memSavings = ((1 - totalMemoryModernized / totalMemoryLegacy) * 100).toFixed(1);
    console.log(`  Modernized uses ${memSavings}% less memory on average`);
  } else {
    const memIncrease = ((totalMemoryModernized / totalMemoryLegacy - 1) * 100).toFixed(1);
    console.log(`  Modernized uses ${memIncrease}% more memory on average`);
  }
  
  console.log(`\nOverall: Modernized is ${avgSpeedup}x faster`);
} else {
  console.log('Legacy implementation requires Sylvester library to be installed.');
  console.log('Install with: npm install sylvester');
  console.log('\nModernized implementation:');
  results.forEach((r, i) => {
    console.log(`  Test ${i + 1}: ${r.modernized.duration.toFixed(2)} ms (${r.modernized.memory.toFixed(2)} MB)`);
  });
}

console.log('\n' + '='.repeat(60));
