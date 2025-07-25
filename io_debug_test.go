package gofann

import (
	"fmt"
	"path/filepath"
	"testing"
)

func TestSaveLoadDebug(t *testing.T) {
	// Create a minimal network
	net1 := CreateStandard[float32](2, 2, 1)
	if net1 == nil {
		t.Fatal("Failed to create network")
	}
	
	// Set specific weights for debugging
	weights := net1.GetWeightArray()
	for i := range weights {
		weights[i] = float32(i) * 0.1
	}
	net1.SetWeights(weights)
	
	// Print network structure
	fmt.Println("Original network:")
	fmt.Printf("  Layers: %v\n", net1.GetLayerSizes())
	fmt.Printf("  Total neurons: %d\n", net1.GetTotalNeurons())
	fmt.Printf("  Total connections: %d\n", net1.GetTotalConnections())
	fmt.Printf("  Weights: %v\n", net1.GetWeightArray())
	
	// Test network output
	testInputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	fmt.Println("  Outputs:")
	for _, input := range testInputs {
		output := net1.Run(input)
		fmt.Printf("    %v -> %.4f\n", input, output[0])
	}
	
	// Save network
	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "debug_network.net")
	
	err := net1.Save(filename)
	if err != nil {
		t.Fatalf("Failed to save network: %v", err)
	}
	
	// Load network
	net2, err := CreateFromFile[float32](filename)
	if err != nil {
		t.Fatalf("Failed to load network: %v", err)
	}
	
	// Print loaded network
	fmt.Println("\nLoaded network:")
	fmt.Printf("  Layers: %v\n", net2.GetLayerSizes())
	fmt.Printf("  Total neurons: %d\n", net2.GetTotalNeurons())
	fmt.Printf("  Total connections: %d\n", net2.GetTotalConnections())
	fmt.Printf("  Weights: %v\n", net2.GetWeightArray())
	
	// Test loaded network output
	fmt.Println("  Outputs:")
	for _, input := range testInputs {
		output := net2.Run(input)
		fmt.Printf("    %v -> %.4f\n", input, output[0])
	}
	
	// Compare weights
	weights1 := net1.GetWeightArray()
	weights2 := net2.GetWeightArray()
	
	if len(weights1) != len(weights2) {
		t.Errorf("Weight array length mismatch: %d vs %d", len(weights1), len(weights2))
	}
	
	for i := range weights1 {
		if abs(weights1[i]-weights2[i]) > 0.0001 {
			t.Errorf("Weight[%d] mismatch: %.4f vs %.4f", i, weights1[i], weights2[i])
		}
	}
}