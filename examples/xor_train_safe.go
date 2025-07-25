// XOR training example with CPU-friendly settings
package main

import (
	"fmt"
	"runtime"
	"time"
	"github.com/lynncoleart/gofann"
)

func main() {
	fmt.Println("GoFANN XOR Training Example (CPU-Safe)")
	fmt.Println("======================================")
	
	// Allow using multiple CPU cores
	runtime.GOMAXPROCS(runtime.NumCPU())
	
	// Create a standard fully connected network
	// 2 inputs, 4 hidden neurons, 1 output
	net := gofann.CreateStandard[float32](2, 4, 1)
	if net == nil {
		panic("Failed to create network")
	}
	
	fmt.Printf("Created network with %d neurons and %d connections\n", 
		net.GetTotalNeurons(), net.GetTotalConnections())
	fmt.Printf("Using %d CPU cores\n", runtime.NumCPU())
	
	// Create XOR training data
	inputs := [][]float32{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	
	outputs := [][]float32{
		{0},
		{1},
		{1},
		{0},
	}
	
	trainData := gofann.CreateTrainDataArray(inputs, outputs)
	
	// Set training parameters
	net.SetTrainingAlgorithm(gofann.TrainRPROP) // RPROP should work now
	net.SetLearningRate(0.7)
	
	// Test before training
	fmt.Println("\nBefore training:")
	for i := 0; i < trainData.GetNumData(); i++ {
		input := trainData.GetInput(i)
		output := net.Run(input)
		expected := trainData.GetOutput(i)
		fmt.Printf("Input: %v => Output: %.4f (expected: %.0f)\n", 
			input, output[0], expected[0])
	}
	
	// Set up training callback to monitor progress
	startTime := time.Now()
	net.SetCallback(func(ann *gofann.Fann[float32], epochs int, mse float32) bool {
		elapsed := time.Since(startTime)
		fmt.Printf("Epoch %5d - MSE: %.6f - Time: %s\n", epochs, mse, elapsed)
		
		// Stop if taking too long
		if elapsed > 30*time.Second {
			fmt.Println("Training timeout reached!")
			return false
		}
		
		return true // Continue training
	})
	
	// Train the network with reasonable limits
	fmt.Println("\nTraining (max 2000 epochs, reporting every 200)...")
	net.TrainOnData(trainData, 2000, 200, 0.001)
	
	// Test after training
	fmt.Println("\nAfter training:")
	testMSE := net.TestData(trainData)
	for i := 0; i < trainData.GetNumData(); i++ {
		input := trainData.GetInput(i)
		output := net.Run(input)
		expected := trainData.GetOutput(i)
		fmt.Printf("Input: %v => Output: %.4f (expected: %.0f)\n", 
			input, output[0], expected[0])
	}
	
	totalTime := time.Since(startTime)
	fmt.Printf("\nFinal MSE: %.6f\n", testMSE)
	fmt.Printf("Bit failures: %d\n", net.GetBitFail())
	fmt.Printf("Total training time: %s\n", totalTime)
	
	if testMSE < 0.01 {
		fmt.Println("\nXOR truth table learned successfully!")
	} else {
		fmt.Println("\nTraining needs more epochs or different parameters.")
	}
}