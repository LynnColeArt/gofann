package gofann

import (
	"math"
	"testing"
)

// TestCascadeDebugNew tests the new cascade implementation
func TestCascadeDebugNew(t *testing.T) {
	// Simple XOR problem
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
	data := CreateTrainDataArray(inputs, outputs)
	
	net := CreateCascade[float32](2, 1)
	
	// Set smaller limits for debugging
	net.SetCascadeMaxOutEpochs(50)
	net.SetCascadeMaxCandEpochs(50)
	net.SetCascadeOutputStagnationEpochs(10)
	net.SetCascadeCandidateStagnationEpochs(10)
	
	// Get initial state
	initialMSE := net.TestData(data)
	t.Logf("Initial MSE: %f", initialMSE)
	
	// Manually add one candidate to debug
	net.cascadeAddCandidate(data, 0.001)
	
	// Check results
	finalMSE := net.TestData(data)
	t.Logf("After adding candidate: MSE=%f", finalMSE)
	
	// Test outputs
	for i, input := range inputs {
		out := net.Run(input)
		t.Logf("  %v -> %.3f (expected %.0f)", input, out[0], outputs[i][0])
	}
}

// TestCascadeScoreCalculation tests if scores are calculated correctly
func TestCascadeScoreCalculation(t *testing.T) {
	// Single pattern to make debugging easier
	inputs := [][]float32{{0.5}}
	outputs := [][]float32{{0.8}}
	data := CreateTrainDataArray(inputs, outputs)
	
	net := CreateCascade[float32](1, 1)
	
	// Get initial MSE
	initialMSE := net.TestData(data)
	initialOut := net.Run([]float32{0.5})[0]
	t.Logf("Initial: output=%f, MSE=%f", initialOut, initialMSE)
	
	// Create a simple candidate
	_ = &cascadeCandidate[float32]{
		neuron: neuron[float32]{
			activationFunction: Sigmoid,
			activationSteepness: 0.5,
		},
		inputWeights:  []float32{1.0, 1.0}, // weight for input and bias
		outputWeights: []float32{0.1},      // small positive weight
		inputSlopes:   make([]float32, 2),
		outputSlopes:  make([]float32, 1),
	}
	
	// Calculate what the score should be
	// Network output
	netOut := initialOut
	// Error
	error := outputs[0][0] - netOut
	t.Logf("Network error: %f", error)
	
	// Candidate activation (assuming bias=1)
	candidateSum := 0.5*1.0 + 1.0*1.0 // input*weight + bias*weight = 1.5
	candidateAct := 1.0 / (1.0 + float32(math.Exp(-float64(candidateSum))))
	t.Logf("Candidate activation: %f", candidateAct)
	
	// Difference = (activation * weight) - error
	diff := (candidateAct * 0.1) - error
	t.Logf("Difference: %f", diff)
	
	// Score starts with MSE and subtracts diff²
	expectedScore := initialMSE - diff*diff
	t.Logf("Expected score: %f", expectedScore)
}