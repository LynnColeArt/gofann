// XOR example - the classic neural network problem
package main

import (
	"fmt"
	"github.com/lynncoleart/gofann"
)

func main() {
	fmt.Println("GoFANN XOR Example")
	fmt.Println("==================")
	
	// Create a standard fully connected network
	// 2 inputs, 3 hidden neurons, 1 output
	net := gofann.CreateStandard[float32](2, 3, 1)
	if net == nil {
		panic("Failed to create network")
	}
	
	fmt.Printf("Created network with %d neurons and %d connections\n", 
		net.GetTotalNeurons(), net.GetTotalConnections())
	
	// Test untrained network
	fmt.Println("\nUntrained network outputs:")
	testInputs := [][]float32{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	
	for _, input := range testInputs {
		output := net.Run(input)
		fmt.Printf("Input: %v => Output: %.4f\n", input, output[0])
	}
	
	// The network should produce random outputs since it's untrained
	// In the next iteration, we'll add training functionality
	
	fmt.Println("\nNote: Network is untrained, so outputs are random.")
	fmt.Println("Training functionality coming next!")
}