// Package gofann - Reflective Training System
// Inspired by Lane Cunningham's revolutionary approach to self-improving neural networks
// https://www.linkedin.com/in/lanecunningham/ 
//
// "Instead of 'bigger is better', we prove: 'A tiny model that understands itself 
// beats a large model that doesn't'" - Lane Cunningham
//
// This system implements Lane's breakthrough concept of neural networks that can:
// 1. Analyze their own confusion patterns
// 2. Generate targeted training data for weaknesses  
// 3. Adapt learning rates based on self-reflection
// 4. Continuously evolve through metacognitive loops

package gofann

import (
	"fmt"
	"math"
	"sort"
)

// ReflectiveTrainer implements Lane Cunningham's self-improving training methodology
type ReflectiveTrainer[T Numeric] struct {
	network *Fann[T]
	
	// Self-reflection components
	confusionMatrix *ConfusionMatrix[T]
	weaknessAnalyzer *WeaknessAnalyzer[T]
	dataAugmenter *DataAugmenter[T]
	adaptiveLR *AdaptiveLearningRate[T]
	
	// Training state
	cycle int
	targetAccuracy T
	improvementThreshold T
	maxCycles int
	
	// Callbacks for introspection
	onCycleComplete func(cycle int, metrics ReflectionMetrics[T])
	onWeaknessDetected func(weaknesses []Weakness[T])
	onImprovement func(oldAcc, newAcc T)
}

// ReflectionMetrics captures the model's self-understanding
type ReflectionMetrics[T Numeric] struct {
	Cycle int
	Accuracy T
	Loss T
	ConfusionEntropy T  // How confused is the model?
	WeaknessCount int   // How many weak spots identified?
	ImprovementRate T   // Rate of self-improvement
	LearningRate T      // Current adaptive learning rate
	TargetedSamples int // Samples generated for weaknesses
}

// ConfusionMatrix tracks what the model gets wrong and why
type ConfusionMatrix[T Numeric] struct {
	matrix map[string]map[string]int  // predicted -> actual -> count
	patterns map[string][]string      // pattern -> examples that confuse it
	confidence map[string]T           // pattern -> average confidence
}

// Weakness represents a specific area where the model struggles
type Weakness[T Numeric] struct {
	Pattern string      // What pattern is confusing (e.g., "COUNT vs SUM")
	ConfusionRate T     // How often it gets this wrong
	Examples []string   // Specific examples it failed on  
	Confidence T        // How confident it was when wrong
	Priority T          // How important to fix this
}

// WeaknessAnalyzer identifies patterns in the model's mistakes
type WeaknessAnalyzer[T Numeric] struct {
	minConfusionRate T
	maxWeaknesses int
	priorityThreshold T
}

// DataAugmenter generates targeted training data for identified weaknesses
type DataAugmenter[T Numeric] struct {
	variationStrategies []VariationStrategy[T]
	samplesPerWeakness int
	diversityFactor T
}

// VariationStrategy defines how to create variations of training examples
type VariationStrategy[T Numeric] interface {
	GenerateVariations(original string, weakness Weakness[T]) []string
	GetName() string
}

// AdaptiveLearningRate adjusts learning based on progress and plateau detection
type AdaptiveLearningRate[T Numeric] struct {
	baseLR T
	currentLR T
	decayFactor T
	plateauThreshold T
	plateauPatience int
	plateauCounter int
	lastImprovement T
}

// NewReflectiveTrainer creates a trainer using Lane Cunningham's methodology
func NewReflectiveTrainer[T Numeric](network *Fann[T]) *ReflectiveTrainer[T] {
	return &ReflectiveTrainer[T]{
		network: network,
		confusionMatrix: &ConfusionMatrix[T]{
			matrix: make(map[string]map[string]int),
			patterns: make(map[string][]string),
			confidence: make(map[string]T),
		},
		weaknessAnalyzer: &WeaknessAnalyzer[T]{
			minConfusionRate: T(0.1),  // 10% confusion rate to be considered weakness
			maxWeaknesses: 10,         // Focus on top weaknesses
			priorityThreshold: T(0.3), // High priority threshold
		},
		dataAugmenter: &DataAugmenter[T]{
			samplesPerWeakness: 20,    // Generate 20 samples per weakness
			diversityFactor: T(0.8),   // High diversity in generated samples
		},
		adaptiveLR: &AdaptiveLearningRate[T]{
			baseLR: T(0.01),
			currentLR: T(0.01),
			decayFactor: T(0.95),
			plateauThreshold: T(0.001), // 0.1% improvement threshold
			plateauPatience: 3,         // Wait 3 cycles before decay
		},
		targetAccuracy: T(0.95),        // Aim for 95% accuracy
		improvementThreshold: T(0.001), // 0.1% improvement threshold
		maxCycles: 100,                 // Maximum reflection cycles
	}
}

// TrainWithReflection implements Lane's revolutionary training loop:
// Train → Reflect → Target Weaknesses → Evolve → Repeat
func (rt *ReflectiveTrainer[T]) TrainWithReflection(data *TrainData[T]) *ReflectionMetrics[T] {
	fmt.Printf("🧠 Starting Reflective Training (inspired by Lane Cunningham)\n")
	fmt.Printf("   Target Accuracy: %.2f%%\n", float64(rt.targetAccuracy*100))
	fmt.Printf("   Max Cycles: %d\n", rt.maxCycles)
	fmt.Printf("   Philosophy: 'A tiny model that understands itself beats a large model that doesn't'\n\n")
	
	var lastMetrics *ReflectionMetrics[T]
	
	for rt.cycle = 0; rt.cycle < rt.maxCycles; rt.cycle++ {
		fmt.Printf("🔄 Reflection Cycle %d\n", rt.cycle+1)
		
		// Phase 1: Train on current data
		rt.network.SetLearningRate(float32(rt.adaptiveLR.currentLR))
		rt.network.TrainOnData(data, 10, 0, 0.001) // Quick training burst
		
		// Phase 2: Self-Reflection - "What am I getting wrong?"
		metrics := rt.analyzePerformance(data)
		
		// Phase 3: Weakness Analysis - "Why am I confusing these patterns?"
		weaknesses := rt.identifyWeaknesses()
		
		// Phase 4: Targeted Data Generation - "Create examples for my weak spots"
		if len(weaknesses) > 0 {
			targetedData := rt.generateTargetedTrainingData(weaknesses)
			rt.network.TrainOnData(targetedData, 5, 0, 0.001)
			metrics.TargetedSamples = targetedData.numData
		}
		
		// Phase 5: Adaptive Learning - "Should I slow down or speed up?"
		rt.adaptLearningRate(metrics)
		
		// Phase 6: Progress Evaluation - "Am I getting better?"
		if rt.checkConvergence(metrics) {
			fmt.Printf("🎯 Target accuracy reached: %.2f%%\n", float64(metrics.Accuracy*100))
			break
		}
		
		// Callbacks for introspection
		if rt.onCycleComplete != nil {
			rt.onCycleComplete(rt.cycle, *metrics)
		}
		if len(weaknesses) > 0 && rt.onWeaknessDetected != nil {
			rt.onWeaknessDetected(weaknesses)
		}
		
		lastMetrics = metrics
		rt.printCycleReport(*metrics, weaknesses)
	}
	
	if lastMetrics == nil {
		lastMetrics = rt.analyzePerformance(data)
	}
	
	fmt.Printf("\n🏁 Reflective Training Complete!\n")
	fmt.Printf("   Final Accuracy: %.2f%%\n", float64(lastMetrics.Accuracy*100))
	fmt.Printf("   Cycles Completed: %d\n", rt.cycle)
	fmt.Printf("   Self-Improvements: %d weakness patterns addressed\n", lastMetrics.WeaknessCount)
	
	return lastMetrics
}

// analyzePerformance implements the self-reflection mechanism
func (rt *ReflectiveTrainer[T]) analyzePerformance(data *TrainData[T]) *ReflectionMetrics[T] {
	correct := 0
	totalLoss := T(0)
	
	// Reset confusion matrix for this cycle
	rt.confusionMatrix.matrix = make(map[string]map[string]int)
	rt.confusionMatrix.patterns = make(map[string][]string)
	rt.confusionMatrix.confidence = make(map[string]T)
	
	for i := 0; i < data.numData; i++ {
		output := rt.network.Run(data.inputs[i])
		expected := data.outputs[i]
		
		// Calculate loss
		sampleLoss := T(0)
		for j := range output {
			diff := output[j] - expected[j]
			sampleLoss += diff * diff
		}
		totalLoss += sampleLoss
		
		// Determine prediction and actual for confusion matrix
		predictedClass := rt.classifyOutput(output)
		actualClass := rt.classifyOutput(expected)
		
		if predictedClass == actualClass {
			correct++
		} else {
			// Record the confusion for analysis
			rt.recordConfusion(predictedClass, actualClass, data.inputs[i], output)
		}
	}
	
	accuracy := T(correct) / T(data.numData)
	avgLoss := totalLoss / T(data.numData)
	
	return &ReflectionMetrics[T]{
		Cycle: rt.cycle,
		Accuracy: accuracy,
		Loss: avgLoss,
		ConfusionEntropy: rt.calculateConfusionEntropy(),
		LearningRate: rt.adaptiveLR.currentLR,
	}
}

// identifyWeaknesses finds patterns the model consistently gets wrong
func (rt *ReflectiveTrainer[T]) identifyWeaknesses() []Weakness[T] {
	weaknesses := make([]Weakness[T], 0)
	
	for predicted, actualMap := range rt.confusionMatrix.matrix {
		for actual, count := range actualMap {
			if predicted != actual && count > 0 {
				// Calculate confusion rate for this pattern
				totalForActual := 0
				for _, c := range rt.confusionMatrix.matrix[predicted] {
					totalForActual += c
				}
				
				if totalForActual > 0 {
					confusionRate := T(count) / T(totalForActual)
					
					if confusionRate >= rt.weaknessAnalyzer.minConfusionRate {
						weakness := Weakness[T]{
							Pattern: fmt.Sprintf("%s confused with %s", actual, predicted),
							ConfusionRate: confusionRate,
							Examples: rt.confusionMatrix.patterns[predicted+"->"+actual],
							Confidence: rt.confusionMatrix.confidence[predicted+"->"+actual],
							Priority: confusionRate * T(count), // Higher confusion rate + more instances = higher priority
						}
						weaknesses = append(weaknesses, weakness)
					}
				}
			}
		}
	}
	
	// Sort by priority and take top weaknesses
	sort.Slice(weaknesses, func(i, j int) bool {
		return weaknesses[i].Priority > weaknesses[j].Priority
	})
	
	if len(weaknesses) > rt.weaknessAnalyzer.maxWeaknesses {
		weaknesses = weaknesses[:rt.weaknessAnalyzer.maxWeaknesses]
	}
	
	return weaknesses
}

// generateTargetedTrainingData creates specific examples to address weaknesses
func (rt *ReflectiveTrainer[T]) generateTargetedTrainingData(weaknesses []Weakness[T]) *TrainData[T] {
	// This is a simplified version - in reality, this would use sophisticated
	// data augmentation techniques based on the specific domain
	
	// For now, we'll create a placeholder that demonstrates the concept
	inputs := make([][]T, 0)
	outputs := make([][]T, 0)
	
	for _, weakness := range weaknesses {
		for i := 0; i < rt.dataAugmenter.samplesPerWeakness; i++ {
			// Generate synthetic training examples targeting this weakness
			// This would be domain-specific (e.g., SQL variations, typos, etc.)
			syntheticInput := rt.generateSyntheticInput(weakness)
			syntheticOutput := rt.generateSyntheticOutput(weakness)
			
			inputs = append(inputs, syntheticInput)
			outputs = append(outputs, syntheticOutput)
		}
	}
	
	if len(inputs) == 0 {
		// Return empty training data if no targeted samples generated
		return CreateTrainDataArray([][]T{}, [][]T{})
	}
	
	return CreateTrainDataArray(inputs, outputs)
}

// Helper methods (simplified implementations for demonstration)

func (rt *ReflectiveTrainer[T]) classifyOutput(output []T) string {
	// Handle empty output gracefully
	if len(output) == 0 {
		return "empty_output"
	}
	
	// Simple classification - find the index with highest value
	maxIdx := 0
	maxVal := output[0]
	for i, val := range output {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return fmt.Sprintf("class_%d", maxIdx)
}

func (rt *ReflectiveTrainer[T]) recordConfusion(predicted, actual string, input, output []T) {
	key := predicted + "->" + actual
	
	if rt.confusionMatrix.matrix[predicted] == nil {
		rt.confusionMatrix.matrix[predicted] = make(map[string]int)
	}
	rt.confusionMatrix.matrix[predicted][actual]++
	
	// Store example that caused confusion
	rt.confusionMatrix.patterns[key] = append(rt.confusionMatrix.patterns[key], fmt.Sprintf("%v", input))
	
	// Calculate confidence (how sure the model was when it was wrong)
	confidence := T(0)
	for _, val := range output {
		if val > confidence {
			confidence = val
		}
	}
	rt.confusionMatrix.confidence[key] = confidence
}

func (rt *ReflectiveTrainer[T]) calculateConfusionEntropy() T {
	// Calculate entropy of confusion matrix (how chaotic are the mistakes?)
	entropy := T(0)
	total := 0
	
	for _, actualMap := range rt.confusionMatrix.matrix {
		for _, count := range actualMap {
			total += count
		}
	}
	
	if total == 0 {
		return 0
	}
	
	for _, actualMap := range rt.confusionMatrix.matrix {
		for _, count := range actualMap {
			if count > 0 {
				p := T(count) / T(total)
				entropy -= p * T(math.Log(float64(p)))
			}
		}
	}
	
	return entropy
}

func (rt *ReflectiveTrainer[T]) adaptLearningRate(metrics *ReflectionMetrics[T]) {
	// Check if we're plateauing
	improvement := metrics.Accuracy - rt.adaptiveLR.lastImprovement
	
	if improvement < rt.adaptiveLR.plateauThreshold {
		rt.adaptiveLR.plateauCounter++
		if rt.adaptiveLR.plateauCounter >= rt.adaptiveLR.plateauPatience {
			// Decay learning rate
			rt.adaptiveLR.currentLR *= rt.adaptiveLR.decayFactor
			rt.adaptiveLR.plateauCounter = 0
			fmt.Printf("   📉 Learning rate decayed to %.6f (plateau detected)\n", rt.adaptiveLR.currentLR)
		}
	} else {
		rt.adaptiveLR.plateauCounter = 0
		rt.adaptiveLR.lastImprovement = metrics.Accuracy
	}
}

func (rt *ReflectiveTrainer[T]) checkConvergence(metrics *ReflectionMetrics[T]) bool {
	return metrics.Accuracy >= rt.targetAccuracy
}

func (rt *ReflectiveTrainer[T]) generateSyntheticInput(weakness Weakness[T]) []T {
	// Placeholder - would be domain-specific
	return make([]T, rt.network.numInput)
}

func (rt *ReflectiveTrainer[T]) generateSyntheticOutput(weakness Weakness[T]) []T {
	// Placeholder - would be domain-specific  
	return make([]T, rt.network.numOutput)
}

func (rt *ReflectiveTrainer[T]) printCycleReport(metrics ReflectionMetrics[T], weaknesses []Weakness[T]) {
	fmt.Printf("   📊 Accuracy: %.2f%% | Loss: %.4f | LR: %.6f\n", 
		float64(metrics.Accuracy*100), float64(metrics.Loss), float64(metrics.LearningRate))
	
	if len(weaknesses) > 0 {
		fmt.Printf("   🎯 Top Weaknesses:\n")
		for i, w := range weaknesses {
			if i >= 3 { // Show top 3
				break
			}
			fmt.Printf("      • %s (%.1f%% confusion)\n", w.Pattern, float64(w.ConfusionRate*100))
		}
	}
	fmt.Println()
}

// Callback setters for introspection and monitoring
func (rt *ReflectiveTrainer[T]) OnCycleComplete(callback func(cycle int, metrics ReflectionMetrics[T])) {
	rt.onCycleComplete = callback
}

func (rt *ReflectiveTrainer[T]) OnWeaknessDetected(callback func(weaknesses []Weakness[T])) {
	rt.onWeaknessDetected = callback
}

func (rt *ReflectiveTrainer[T]) OnImprovement(callback func(oldAcc, newAcc T)) {
	rt.onImprovement = callback
}