package main

import (
	"fmt"
	"github.com/lynncoleart/gofann"
)

func main() {
	fmt.Println("🧠 GoFANN Simple Demo")
	fmt.Println("====================\n")

	// Demo 1: Basic XOR Network
	fmt.Println("1️⃣ Testing XOR Network")
	fmt.Println("----------------------")
	
	// Create network: 2 inputs, 4 hidden, 1 output
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
	
	// Train with callback to show progress
	fmt.Println("\nTraining...")
	epochCount := 0
	net.SetCallback(func(ann *gofann.Fann[float32], epochs int, mse float32) bool {
		epochCount = epochs
		if epochs % 100 == 0 {
			fmt.Printf("  Epoch %d: MSE = %.6f\n", epochs, mse)
		}
		return true // Continue training
	})
	
	net.TrainOnData(trainData, 500, 10, 0.01)
	
	// Test after training
	fmt.Println("\nAfter training:")
	correct := 0
	for i, input := range inputs {
		result := net.Run(input)
		fmt.Printf("  XOR(%v) = %.3f (expected %.0f)", input, result[0], outputs[i][0])
		
		// Check if correct (within threshold)
		expected := outputs[i][0]
		if (expected < 0.5 && result[0] < 0.5) || (expected >= 0.5 && result[0] >= 0.5) {
			fmt.Printf(" ✓\n")
			correct++
		} else {
			fmt.Printf(" ✗\n")
		}
	}
	
	fmt.Printf("\nAccuracy: %d/%d (%.0f%%)\n", correct, len(inputs), float64(correct)/float64(len(inputs))*100)
	fmt.Printf("Final MSE: %.6f\n", net.GetMSE())
	fmt.Printf("Total epochs: %d\n", epochCount)
	
	// Demo 2: Save and Load
	fmt.Println("\n2️⃣ Testing Save/Load")
	fmt.Println("--------------------")
	
	// Save the trained network
	err := net.Save("xor_trained.net")
	if err != nil {
		fmt.Printf("Error saving: %v\n", err)
	} else {
		fmt.Println("Network saved to xor_trained.net")
	}
	
	// Load it back
	loaded, err := gofann.CreateFromFile[float32]("xor_trained.net")
	if err != nil {
		fmt.Printf("Error loading: %v\n", err)
	} else {
		fmt.Println("Network loaded successfully")
		
		// Test loaded network
		fmt.Println("\nTesting loaded network:")
		for _, input := range inputs {
			result := loaded.Run(input)
			fmt.Printf("  XOR(%v) = %.3f\n", input, result[0])
		}
	}
	
	// Demo 3: Different Network Types
	fmt.Println("\n3️⃣ Testing Network Types")
	fmt.Println("------------------------")
	
	// Shortcut network
	shortcut := gofann.CreateShortcut[float32](2, 3, 1)
	fmt.Printf("Shortcut network: %d connections\n", shortcut.GetTotalConnections())
	
	// Standard network for comparison
	standard := gofann.CreateStandard[float32](2, 3, 1)
	fmt.Printf("Standard network: %d connections\n", standard.GetTotalConnections())
	
	// Sparse network
	sparse := gofann.CreateSparse[float32](0.5, 2, 3, 1)
	fmt.Printf("Sparse network (50%%): %d connections\n", sparse.GetTotalConnections())
	
	// Demo 4: Training Algorithms Comparison
	fmt.Println("\n4️⃣ Comparing Training Algorithms")
	fmt.Println("--------------------------------")
	
	algorithms := []struct{
		name string
		algo gofann.TrainAlgorithm
	}{
		{"RPROP", gofann.TrainRPROP},
		{"Batch", gofann.TrainBatch},
		{"Quickprop", gofann.TrainQuickprop},
	}
	
	for _, alg := range algorithms {
		// Create fresh network
		testNet := gofann.CreateStandard[float32](2, 4, 1)
		testNet.SetActivationFunctionHidden(gofann.SigmoidSymmetric)
		testNet.SetActivationFunctionOutput(gofann.Sigmoid)
		testNet.RandomizeWeights(-1, 1)
		testNet.SetLearningRate(0.7)
		testNet.SetTrainingAlgorithm(alg.algo)
		
		// Train for fixed epochs
		testNet.TrainOnData(trainData, 200, 0, 0.01)
		
		fmt.Printf("%s: Final MSE = %.6f\n", alg.name, testNet.TestData(trainData))
	}
	
	fmt.Println("\n✅ Demo complete!")
	fmt.Println("\nNext steps:")
	fmt.Println("- Try reflective training with gofann.NewReflectiveTrainer")
	fmt.Println("- Test concurrent training with gofann.NewConcurrentTrainer")
	fmt.Println("- Build custom error patterns for your domain")
}