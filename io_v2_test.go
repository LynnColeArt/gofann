package gofann

import (
	"path/filepath"
	"testing"
)

func TestSaveLoadV2(t *testing.T) {
	// Create network with known weights
	net1 := CreateStandard[float32](2, 3, 1)
	
	// Set specific weights
	weights := net1.GetWeightArray()
	for i := range weights {
		weights[i] = float32(i) * 0.1
	}
	net1.SetWeights(weights)
	
	// Test outputs
	testInputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	outputs1 := make([][]float32, len(testInputs))
	for i, input := range testInputs {
		outputs1[i] = net1.Run(input)
	}
	
	// Save
	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "test_v2.net")
	
	err := net1.SaveV2(filename)
	if err != nil {
		t.Fatalf("Failed to save: %v", err)
	}
	
	// Load
	net2, err := LoadV2[float32](filename)
	if err != nil {
		t.Fatalf("Failed to load: %v", err)
	}
	
	// Compare structure
	if net1.GetTotalNeurons() != net2.GetTotalNeurons() {
		t.Errorf("Neuron count mismatch: %d vs %d", net1.GetTotalNeurons(), net2.GetTotalNeurons())
	}
	if net1.GetTotalConnections() != net2.GetTotalConnections() {
		t.Errorf("Connection count mismatch: %d vs %d", net1.GetTotalConnections(), net2.GetTotalConnections())
	}
	
	// Compare weights
	weights1 := net1.GetWeightArray()
	weights2 := net2.GetWeightArray()
	
	for i := range weights1 {
		if abs(weights1[i]-weights2[i]) > 0.0001 {
			t.Errorf("Weight[%d] mismatch: %.4f vs %.4f", i, weights1[i], weights2[i])
		}
	}
	
	// Compare outputs
	for i, input := range testInputs {
		output2 := net2.Run(input)
		if abs(outputs1[i][0]-output2[0]) > 0.0001 {
			t.Errorf("Output mismatch for %v: %.4f vs %.4f", input, outputs1[i][0], output2[0])
		}
	}
	
	t.Logf("SaveV2/LoadV2 successful - all outputs match!")
}