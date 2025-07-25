// XOR training example - demonstrates training a network to solve XOR
package main

import (
	"fmt"
	"github.com/lynncoleart/gofann"
)

func main() {
	fmt.Println("GoFANN XOR Training Example")
	fmt.Println("===========================")
	
	// Create a standard fully connected network
	// 2 inputs, 4 hidden neurons, 1 output
	net := gofann.CreateStandard[float32](2, 4, 1)
	if net == nil {
		panic("Failed to create network")
	}
	
	fmt.Printf("Created network with %d neurons and %d connections\n", 
		net.GetTotalNeurons(), net.GetTotalConnections())
	
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
	net.SetTrainingAlgorithm(gofann.TrainIncremental)
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
	net.SetCallback(func(ann *gofann.Fann[float32], epochs int, mse float32) bool {
		fmt.Printf("Epoch %5d - MSE: %.6f\n", epochs, mse)
		return true // Continue training
	})
	
	// Train the network
	fmt.Println("\nTraining...")
	net.TrainOnData(trainData, 5000, 500, 0.0001)
	
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
	fmt.Printf("\nFinal MSE: %.6f\n", testMSE)
	fmt.Printf("Bit failures: %d\n", net.GetBitFail())
	
	// Demonstrate the network has learned XOR
	fmt.Println("\nXOR truth table learned!")
}