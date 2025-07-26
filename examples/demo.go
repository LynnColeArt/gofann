package main

import (
	"fmt"
	"github.com/lynncoleart/gofann"
)

func main() {
	fmt.Println("🧠 GoFANN Manual Testing")
	fmt.Println("========================\n")

	// Test 1: Basic XOR Network
	testBasicXOR()
	
	// Test 2: Reflective Training
	testReflectiveTraining()
	
	// Test 3: Concurrent Training
	testConcurrentTraining()
	
	// Test 4: CLI Error Assistant
	testCLIAssistant()
}

func testBasicXOR() {
	fmt.Println("1️⃣ Testing Basic XOR Network")
	fmt.Println("----------------------------")
	
	// Create network
	net := gofann.CreateStandard[float32](2, 4, 1)
	net.SetActivationFunctionHidden(gofann.SigmoidSymmetric)
	net.SetActivationFunctionOutput(gofann.Sigmoid)
	net.RandomizeWeights(-1, 1)
	net.SetLearningRate(0.7)
	net.SetTrainingAlgorithm(gofann.TrainRPROP)
	
	// Create XOR training data
	inputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	outputs := [][]float32{{0}, {1}, {1}, {0}}
	trainData := gofann.CreateTrainDataArray(inputs, outputs)
	
	// Test before training
	fmt.Println("Before training:")
	for i, input := range inputs {
		result := net.Run(input)
		fmt.Printf("  XOR(%v) = %.3f (expected %.0f)\n", input, result[0], outputs[i][0])
	}
	
	// Train
	fmt.Println("\nTraining for 1000 epochs...")
	net.TrainOnData(trainData, 1000, 100, 0.01)
	
	// Test after training
	fmt.Println("\nAfter training:")
	for i, input := range inputs {
		result := net.Run(input)
		fmt.Printf("  XOR(%v) = %.3f (expected %.0f)\n", input, result[0], outputs[i][0])
	}
	
	// Save the network
	err := net.Save("xor_network.net")
	if err == nil {
		fmt.Println("\n✅ Network saved to xor_network.net")
	}
	
	fmt.Println()
}

func testReflectiveTraining() {
	fmt.Println("2️⃣ Testing Reflective Training (Lane Cunningham's Method)")
	fmt.Println("--------------------------------------------------------")
	
	// Create network
	net := gofann.CreateStandard[float32](2, 5, 2)
	trainer := gofann.NewReflectiveTrainer(net)
	
	// Create multi-class data (4 classes)
	inputs := [][]float32{
		{0, 0}, {0, 0.5}, {0, 1},    // Class 0
		{0.5, 0}, {0.5, 0.5}, {0.5, 1}, // Class 1
		{1, 0}, {1, 0.5}, {1, 1},    // Class 2
	}
	outputs := [][]float32{
		{1, 0}, {1, 0}, {1, 0},  // Class 0
		{0, 1}, {0, 1}, {0, 1},  // Class 1
		{0, 0}, {0, 0}, {0, 0},  // Class 2 (neither)
	}
	
	trainData := gofann.CreateTrainDataArray(inputs, outputs)
	
	// Set up monitoring
	trainer.OnCycleComplete(func(cycle int, metrics gofann.ReflectionMetrics[float32]) {
		fmt.Printf("  Cycle %d: Accuracy=%.2f%%, MSE=%.4f, Weaknesses=%d\n",
			cycle+1, metrics.Accuracy*100, metrics.Loss, metrics.WeaknessCount)
	})
	
	// Train with reflection
	fmt.Println("\nStarting reflective training...")
	metrics := trainer.TrainWithReflection(trainData)
	
	fmt.Printf("\n✅ Final Results: Accuracy=%.2f%%, MSE=%.4f\n", 
		metrics.Accuracy*100, metrics.Loss)
	fmt.Println()
}

func testConcurrentTraining() {
	fmt.Println("3️⃣ Testing Concurrent Multi-Expert Training")
	fmt.Println("------------------------------------------")
	
	// Create different logical gate experts
	xorData := gofann.CreateTrainDataArray[float32](
		[][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
		[][]float32{{0}, {1}, {1}, {0}},
	)
	
	andData := gofann.CreateTrainDataArray[float32](
		[][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
		[][]float32{{0}, {0}, {0}, {1}},
	)
	
	orData := gofann.CreateTrainDataArray[float32](
		[][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
		[][]float32{{0}, {1}, {1}, {1}},
	)
	
	// Create experts
	experts := []*gofann.ReflectiveExpert[float32]{
		gofann.NewReflectiveExpert[float32]("XORMaster", "xor", []int{2, 4, 1}),
		gofann.NewReflectiveExpert[float32]("ANDExpert", "and", []int{2, 3, 1}),
		gofann.NewReflectiveExpert[float32]("ORWizard", "or", []int{2, 3, 1}),
	}
	
	// Networks are already initialized in NewReflectiveExpert
	// Just need to set training parameters
	for _, expert := range experts {
		expert.network.RandomizeWeights(-1, 1)
		expert.network.SetLearningRate(0.7)
		expert.network.SetTrainingAlgorithm(gofann.TrainRPROP)
	}
	
	// Prepare data
	dataMap := map[string]*gofann.TrainData[float32]{
		"xor": xorData,
		"and": andData,
		"or":  orData,
	}
	
	// Train concurrently
	trainer := gofann.NewConcurrentTrainer[float32](3)
	fmt.Println("\nTraining 3 experts concurrently...")
	results := trainer.TrainExperts(experts, dataMap)
	
	// Show results
	fmt.Println("\nTraining Results:")
	for _, result := range results {
		status := "❌ Failed"
		if result.Success {
			status = "✅ Success"
		}
		fmt.Printf("  %s: %s (MSE=%.6f, Epochs=%d)\n", 
			result.ID, status, result.FinalMSE, result.EpochsTrained)
	}
	
	// Test the experts
	fmt.Println("\nTesting trained experts:")
	testInputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	
	for _, expert := range experts {
		fmt.Printf("\n  %s results:\n", expert.GetName())
		for _, input := range testInputs {
			output := expert.GetNetwork().Run(input)
			fmt.Printf("    %v -> %.3f\n", input, output[0])
		}
	}
	fmt.Println()
}

func testCLIAssistant() {
	fmt.Println("4️⃣ Testing CLI Error Assistant Demo")
	fmt.Println("-----------------------------------")
	
	// Create a simple error classifier
	net := gofann.CreateStandard[float32](10, 20, 4) // 10 features, 4 error types
	net.SetActivationFunctionHidden(gofann.SigmoidSymmetric)
	net.SetActivationFunctionOutput(gofann.Sigmoid)
	net.RandomizeWeights(-1, 1)
	
	// Simulate error pattern features (simplified)
	// Features: has_merge, has_conflict, has_permission, has_module, etc.
	errorPatterns := map[string][]float32{
		"merge conflict": {1, 1, 0, 0, 0, 1, 0, 0, 0, 0},
		"permission denied": {0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
		"module not found": {0, 0, 0, 1, 1, 0, 0, 0, 1, 0},
		"syntax error": {0, 0, 0, 0, 0, 0, 0, 1, 1, 1},
	}
	
	// Simulate a trained network response
	fmt.Println("\nSimulating CLI error diagnosis:")
	
	for errorType, features := range errorPatterns {
		output := net.Run(features)
		
		// Find highest confidence
		maxIdx := 0
		maxVal := output[0]
		for i, val := range output {
			if val > maxVal {
				maxVal = val
				maxIdx = i
			}
		}
		
		errorTypes := []string{"Git Conflict", "Permission", "Module", "Syntax"}
		fmt.Printf("\nError: '%s'\n", errorType)
		fmt.Printf("  Detected as: %s (confidence: %.2f%%)\n", 
			errorTypes[maxIdx], maxVal*100)
		
		// Provide recommendations
		switch maxIdx {
		case 0:
			fmt.Println("  💡 Try: git status, then resolve conflicts in files")
		case 1:
			fmt.Println("  💡 Try: Check file permissions with ls -la")
		case 2:
			fmt.Println("  💡 Try: npm install or check package.json")
		case 3:
			fmt.Println("  💡 Try: Check for missing brackets or semicolons")
		}
	}
	
	fmt.Println("\n✅ CLI Assistant demo complete!")
}