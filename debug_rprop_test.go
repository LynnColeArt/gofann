package gofann

import (
	"fmt"
	"testing"
)

func TestRPROPDebug(t *testing.T) {
	// Create simple network
	net := CreateStandard[float32](2, 3, 1)
	net.SetTrainingAlgorithm(TrainRPROP)
	
	// Create XOR training data
	inputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	outputs := [][]float32{{0}, {1}, {1}, {0}}
	trainData := CreateTrainDataArray(inputs, outputs)
	
	// Check initial state
	fmt.Println("Initial weights sample:")
	for i := 0; i < 5 && i < len(net.weights); i++ {
		fmt.Printf("  weight[%d] = %.4f\n", i, net.weights[i])
	}
	
	// Train for just a few epochs
	for epoch := 1; epoch <= 5; epoch++ {
		mse := net.TrainEpoch(trainData)
		fmt.Printf("\nEpoch %d - MSE: %.4f\n", epoch, mse)
		
		// Check a few slopes
		if net.trainSlopes != nil && len(net.trainSlopes) > 0 {
			fmt.Println("Slopes sample:")
			for i := 0; i < 5 && i < len(net.trainSlopes); i++ {
				fmt.Printf("  slope[%d] = %.4f\n", i, net.trainSlopes[i])
			}
		}
		
		// Check outputs
		fmt.Println("Network outputs:")
		for i, input := range inputs {
			output := net.Run(input)
			fmt.Printf("  %v -> %.4f (expected %.0f)\n", input, output[0], outputs[i][0])
		}
		
		// Check some neuron values
		outputLayer := net.layers[len(net.layers)-1]
		outputNeuron := &net.neurons[outputLayer.firstNeuron]
		fmt.Printf("Output neuron: sum=%.4f, value=%.4f\n", outputNeuron.sum, outputNeuron.value)
	}
	
	// Check final weights
	fmt.Println("\nFinal weights sample:")
	for i := 0; i < 5 && i < len(net.weights); i++ {
		fmt.Printf("  weight[%d] = %.4f\n", i, net.weights[i])
	}
}