package gofann

import (
	"testing"
)

// TestCascadeFinal tests cascade with optimized parameters
func TestCascadeFinal(t *testing.T) {
	// XOR problem
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
	
	// Optimize parameters for XOR
	net.SetCascadeWeightMultiplier(1.0) // Larger initial weights
	net.SetLearningRate(2.0)            // Higher learning rate
	net.SetCascadeMaxOutEpochs(100)
	net.SetCascadeMaxCandEpochs(100)
	net.SetCascadeOutputStagnationEpochs(20)
	net.SetCascadeCandidateStagnationEpochs(20)
	
	// Track progress
	epochCount := 0
	net.SetCallback(func(ann *Fann[float32], epochs int, mse float32) bool {
		epochCount++
		if epochCount%100 == 0 {
			t.Logf("Epoch %d: MSE=%f, neurons=%d", epochCount, mse, ann.GetTotalNeurons())
		}
		return true
	})
	
	// Get initial state
	initialMSE := net.TestData(data)
	t.Logf("Initial: MSE=%f, neurons=%d", initialMSE, net.GetTotalNeurons())
	
	// Train
	net.CascadetrainOnData(data, 5, 1, 0.01)
	
	// Final results
	finalMSE := net.TestData(data)
	t.Logf("Final: MSE=%f, neurons=%d", finalMSE, net.GetTotalNeurons())
	
	// Test all patterns
	correct := 0
	t.Log("Final outputs:")
	for i, input := range inputs {
		out := net.Run(input)
		expected := outputs[i][0]
		isCorrect := (out[0] > 0.5 && expected > 0.5) || (out[0] < 0.5 && expected < 0.5)
		if isCorrect {
			correct++
		}
		t.Logf("  %v -> %.3f (expected %.0f) %s", input, out[0], expected, 
			map[bool]string{true: "✓", false: "✗"}[isCorrect])
	}
	
	if correct < 4 {
		t.Errorf("Only %d/4 patterns correct", correct)
	}
}