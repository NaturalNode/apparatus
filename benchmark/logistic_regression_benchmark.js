/*
Benchmark comparing Logistic Regression Classifier implementations
Original (Sylvester-based) vs Modernized (direct array manipulation)
*/

'use strict';

const LogisticRegressionLegacy = require('../lib/apparatus/classifier/logistic-regression-legacy');
const LogisticRegressionModernized = require('../lib/apparatus/classifier/logistic-regression-modernized');

/**
 * Generate random training data
 */
function generateTrainingData(numSamples, numFeatures, numClasses) {
    const data = [];
    
    for (let classIdx = 0; classIdx < numClasses; classIdx++) {
        const samplesPerClass = numSamples / numClasses;
        for (let i = 0; i < samplesPerClass; i++) {
            const example = new Array(numFeatures);
            for (let j = 0; j < numFeatures; j++) {
                // Generate features biased by class
                example[j] = Math.random() + (classIdx * 0.5);
            }
            data.push({
                features: example,
                label: `class_${classIdx}`
            });
        }
    }
    
    return data;
}

/**
 * Measure memory usage in MB
 */
function getMemoryUsage() {
    if (global.gc) {
        global.gc();
    }
    const used = process.memoryUsage();
    return {
        heapUsed: Math.round(used.heapUsed / 1024 / 1024 * 100) / 100,
        heapTotal: Math.round(used.heapTotal / 1024 / 1024 * 100) / 100,
        external: Math.round(used.external / 1024 / 1024 * 100) / 100
    };
}

/**
 * Benchmark training phase
 */
function benchmarkTraining(ClassifierClass, data, label) {
    const classifier = new ClassifierClass();
    
    // Measure memory before training
    const memBefore = getMemoryUsage();
    const startTime = process.hrtime.bigint();
    
    // Add examples
    for (let i = 0; i < data.length; i++) {
        classifier.addExample(data[i].features, data[i].label);
    }
    
    // Train
    classifier.train();
    
    const endTime = process.hrtime.bigint();
    const memAfter = getMemoryUsage();
    
    const timeMs = Number(endTime - startTime) / 1000000; // Convert to milliseconds
    const heapUsedDiff = memAfter.heapUsed - memBefore.heapUsed;
    
    return {
        label,
        trainingTime: Math.round(timeMs * 100) / 100,
        memoryUsed: Math.round(heapUsedDiff * 100) / 100,
        classifier
    };
}

/**
 * Benchmark classification phase
 */
function benchmarkClassification(classifier, testData, label) {
    const memBefore = getMemoryUsage();
    const startTime = process.hrtime.bigint();
    
    // Classify all test samples
    let correctCount = 0;
    for (let i = 0; i < testData.length; i++) {
        const result = classifier.getClassifications(testData[i].features);
        // Check if top prediction matches actual label
        if (result[0].label === testData[i].label) {
            correctCount++;
        }
    }
    
    const endTime = process.hrtime.bigint();
    const memAfter = getMemoryUsage();
    
    const timeMs = Number(endTime - startTime) / 1000000;
    const heapUsedDiff = memAfter.heapUsed - memBefore.heapUsed;
    const accuracy = Math.round((correctCount / testData.length) * 10000) / 100;
    
    return {
        label,
        classificationTime: Math.round(timeMs * 100) / 100,
        memoryUsed: Math.round(heapUsedDiff * 100) / 100,
        accuracy,
        sampleCount: testData.length
    };
}

/**
 * Main benchmark
 */
function runBenchmark() {
    console.log('='.repeat(80));
    console.log('Logistic Regression Classifier - Performance Benchmark');
    console.log('='.repeat(80));
    console.log();
    
    const testConfigs = [
        { samples: 100, features: 10, classes: 2, name: 'Small (100 samples)' },
        { samples: 500, features: 20, classes: 3, name: 'Medium (500 samples)' },
        { samples: 2000, features: 50, classes: 5, name: 'Large (2000 samples)' }
    ];
    
    for (const config of testConfigs) {
        console.log(`\n${'='.repeat(80)}`);
        console.log(`Dataset: ${config.name}`);
        console.log(`Features: ${config.features}, Classes: ${config.classes}`);
        console.log('-'.repeat(80));
        
        // Generate data split
        const allData = generateTrainingData(config.samples, config.features, config.classes);
        const trainSize = Math.floor(allData.length * 0.8);
        const trainData = allData.slice(0, trainSize);
        const testData = allData.slice(trainSize);
        
        console.log(`Training samples: ${trainData.length}, Test samples: ${testData.length}`);
        console.log();
        
        // Benchmark legacy (Sylvester)
        console.log('Legacy Classifier (Sylvester):');
        const legacyTrain = benchmarkTraining(LogisticRegressionLegacy, trainData, 'Legacy');
        console.log(`  Training time:  ${legacyTrain.trainingTime} ms`);
        console.log(`  Memory used:    ${legacyTrain.memoryUsed} MB`);
        
        const legacyClass = benchmarkClassification(legacyTrain.classifier, testData, 'Legacy');
        console.log(`  Classification time: ${legacyClass.classificationTime} ms (${testData.length} samples)`);
        console.log(`  Memory used:         ${legacyClass.memoryUsed} MB`);
        console.log(`  Accuracy:           ${legacyClass.accuracy}%`);
        console.log();
        
        // Benchmark modernized
        console.log('Modernized Classifier (Direct arrays):');
        const modernTrain = benchmarkTraining(LogisticRegressionModernized, trainData, 'Modernized');
        console.log(`  Training time:  ${modernTrain.trainingTime} ms`);
        console.log(`  Memory used:    ${modernTrain.memoryUsed} MB`);
        
        const modernClass = benchmarkClassification(modernTrain.classifier, testData, 'Modernized');
        console.log(`  Classification time: ${modernClass.classificationTime} ms (${testData.length} samples)`);
        console.log(`  Memory used:         ${modernClass.memoryUsed} MB`);
        console.log(`  Accuracy:           ${modernClass.accuracy}%`);
        console.log();
        
        // Calculate improvements
        const speedupTrain = Math.round((legacyTrain.trainingTime / modernTrain.trainingTime) * 100) / 100;
        const speedupClass = Math.round((legacyClass.classificationTime / modernClass.classificationTime) * 100) / 100;
        const memSavingsTrain = Math.round((legacyTrain.memoryUsed - modernTrain.memoryUsed) * 100) / 100;
        const memSavingsClass = Math.round((legacyClass.memoryUsed - modernClass.memoryUsed) * 100) / 100;
        
        console.log('IMPROVEMENTS (Modernized vs Legacy):');
        console.log(`  Training speedup:        ${speedupTrain}x faster`);
        console.log(`  Classification speedup:  ${speedupClass}x faster`);
        console.log(`  Training memory saving:  ${memSavingsTrain} MB (${Math.round((memSavingsTrain / Math.abs(legacyTrain.memoryUsed)) * 100)}%)`);
        console.log(`  Classification memory saving: ${memSavingsClass} MB (${Math.round((memSavingsClass / Math.abs(legacyClass.memoryUsed)) * 100)}%)`);
    }
    
    console.log();
    console.log('='.repeat(80));
    console.log('Benchmark Complete');
    console.log('='.repeat(80));
}

// Run benchmark
console.log('\nNote: For accurate memory measurements, run with: node --expose-gc benchmark/logistic_regression_benchmark.js\n');
runBenchmark();
