package gofann

import (
	"testing"
)

func TestDebugSimpleTraining(t *testing.T) {
	// Create a simple 2-2-1 network
	net := CreateStandard[float32](2, 2, 1)
	net.SetActivationFunctionHidden(SigmoidSymmetric)
	net.SetActivationFunctionOutput(Sigmoid)
	net.RandomizeWeights(-1, 1)
	net.SetLearningRate(0.7)
	net.SetTrainingAlgorithm(TrainIncremental)
	
	// Simple AND data
	inputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	outputs := [][]float32{{0}, {0}, {0}, {1}}
	data := CreateTrainDataArray(inputs, outputs)
	
	// Test initial state
	t.Logf("Initial network state:")
	for i, input := range inputs {
		output := net.Run(input)
		t.Logf("  %v -> %.3f (expected %.0f)", input, output[0], outputs[i][0])
	}
	
	// Train for a few epochs manually
	for epoch := 0; epoch < 10; epoch++ {
		mse := net.TrainEpoch(data)
		t.Logf("Epoch %d: MSE=%.6f", epoch, mse)
	}
	
	// Test after training
	t.Logf("\nAfter 10 epochs:")
	for i, input := range inputs {
		output := net.Run(input)
		t.Logf("  %v -> %.3f (expected %.0f)", input, output[0], outputs[i][0])
	}
	
	// Check weights
	weights := net.GetWeights()
	t.Logf("\nNetwork has %d weights", len(weights))
	t.Logf("First few weights: %v", weights[:min(5, len(weights))])
	
	// Check network structure
	t.Logf("\nNetwork structure:")
	t.Logf("  Input neurons: %d", net.numInput)
	t.Logf("  Output neurons: %d", net.numOutput)
	t.Logf("  Total neurons: %d", net.totalNeurons)
	t.Logf("  Total connections: %d", net.totalConnections)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}