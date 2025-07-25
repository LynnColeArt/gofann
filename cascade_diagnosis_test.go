package gofann

import (
	"testing"
)

// TestCascadeDiagnosis helps diagnose cascade training issues
func TestCascadeDiagnosis(t *testing.T) {
	// Simple problem: learn to output 0.5
	inputs := [][]float32{{0.5}}
	outputs := [][]float32{{0.5}}
	data := CreateTrainDataArray(inputs, outputs)
	
	net := CreateCascade[float32](1, 1)
	
	// Very detailed logging
	t.Logf("Initial network: %d neurons", net.GetTotalNeurons())
	
	// Test initial output
	initialOut := net.Run([]float32{0.5})
	t.Logf("Initial output: %v", initialOut)
	
	initialMSE := net.TestData(data)
	t.Logf("Initial MSE: %f", initialMSE)
	
	// Set up detailed callback
	epochCount := 0
	net.SetCallback(func(ann *Fann[float32], epochs int, mse float32) bool {
		epochCount++
		if epochCount%10 == 0 {
			out := ann.Run([]float32{0.5})
			t.Logf("Epoch %d: MSE=%f, output=%v, neurons=%d", 
				epochCount, mse, out, ann.GetTotalNeurons())
		}
		return epochCount < 100
	})
	
	// Train with cascade
	net.CascadetrainOnData(data, 3, 1, 0.001)
	
	// Final test
	finalOut := net.Run([]float32{0.5})
	finalMSE := net.TestData(data)
	
	t.Logf("Final output: %v", finalOut)
	t.Logf("Final MSE: %f", finalMSE)
	t.Logf("Final neurons: %d", net.GetTotalNeurons())
	
	// Analyze network structure
	t.Logf("Layers: %d", len(net.layers))
	for i, layer := range net.layers {
		t.Logf("  Layer %d: neurons %d-%d", i, layer.firstNeuron, layer.lastNeuron)
	}
	
	// Check if output improved
	if finalMSE > initialMSE {
		t.Errorf("MSE got worse: %f -> %f", initialMSE, finalMSE)
	}
	
	// Check if output is reasonable
	if finalOut[0] < 0.1 || finalOut[0] > 0.9 {
		t.Errorf("Output out of reasonable range: %v", finalOut)
	}
}

// TestCascadeSimpleXOR tests cascade on simple XOR
func TestCascadeSimpleXOR(t *testing.T) {
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
	
	// Configure for quick convergence
	net.SetCascadeMaxOutEpochs(50)
	net.SetCascadeMaxCandEpochs(50)
	net.SetCascadeOutputStagnationEpochs(5)
	net.SetCascadeCandidateStagnationEpochs(5)
	
	initialMSE := net.TestData(data)
	t.Logf("XOR Initial MSE: %f", initialMSE)
	
	// Track neuron additions
	neuronHistory := []int{net.GetTotalNeurons()}
	net.SetCallback(func(ann *Fann[float32], epochs int, mse float32) bool {
		if ann.GetTotalNeurons() > neuronHistory[len(neuronHistory)-1] {
			neuronHistory = append(neuronHistory, ann.GetTotalNeurons())
			t.Logf("Added neuron, total: %d, MSE: %f", ann.GetTotalNeurons(), mse)
			
			// Test current outputs
			for i, input := range inputs {
				out := ann.Run(input)
				t.Logf("  %v -> %.3f (expected %.0f)", input, out[0], outputs[i][0])
			}
		}
		return true
	})
	
	// Train
	net.CascadetrainOnData(data, 5, 1, 0.1)
	
	finalMSE := net.TestData(data)
	t.Logf("XOR Final MSE: %f", finalMSE)
	
	// Final outputs
	t.Log("Final outputs:")
	for i, input := range inputs {
		out := net.Run(input)
		t.Logf("  %v -> %.3f (expected %.0f)", input, out[0], outputs[i][0])
	}
}