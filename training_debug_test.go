package gofann

import (
	"fmt"
	"testing"
)

func TestTrainingDebug(t *testing.T) {
	// Simple 2D dataset
	inputs := [][]float32{{0, 0}, {1, 1}}
	outputs := [][]float32{{0}, {1}}
	trainData := CreateTrainDataArray(inputs, outputs)
	
	// Test incremental which we know works
	net := CreateStandard[float32](2, 2, 1)
	net.SetTrainingAlgorithm(TrainIncremental)
	net.SetLearningRate(0.5)
	
	// Train one epoch at a time
	for epoch := 1; epoch <= 5; epoch++ {
		mse := net.TrainEpoch(trainData)
		fmt.Printf("Epoch %d: MSE=%.4f\n", epoch, mse)
		
		// Check outputs
		for i, input := range inputs {
			output := net.Run(input)
			fmt.Printf("  %v -> %.4f (expected %.0f)\n", 
				input, output[0], outputs[i][0])
		}
	}
}

func TestQuickpropSimple(t *testing.T) {
	// Even simpler - just learn to output 1
	inputs := [][]float32{{1}}
	outputs := [][]float32{{1}}
	trainData := CreateTrainDataArray(inputs, outputs)
	
	net := CreateStandard[float32](1, 2, 1)
	net.SetTrainingAlgorithm(TrainQuickprop)
	net.SetLearningRate(0.5)
	
	fmt.Println("Testing Quickprop on simple dataset:")
	for epoch := 1; epoch <= 10; epoch++ {
		mse := net.TrainEpoch(trainData)
		output := net.Run(inputs[0])
		fmt.Printf("Epoch %d: output=%.4f, MSE=%.4f\n", 
			epoch, output[0], mse)
		
		if mse < 0.001 {
			fmt.Println("Converged!")
			return
		}
	}
}