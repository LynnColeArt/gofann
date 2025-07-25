package gofann

import (
	"fmt"
	"testing"
)

func TestConnectionBounds(t *testing.T) {
	// Test different network configurations
	configs := [][]int{
		{2, 3, 1},      // XOR network
		{10, 5, 3},     // Larger network
		{1, 1, 1},      // Minimal network
		{100, 50, 10},  // Large network
	}
	
	for _, config := range configs {
		t.Run(fmt.Sprintf("Config_%v", config), func(t *testing.T) {
			net := CreateStandard[float32](config...)
			if net == nil {
				t.Fatal("Failed to create network")
			}
			
			// Check connection bounds
			maxConnIdx := 0
			for i, neuron := range net.neurons {
				if neuron.lastCon > maxConnIdx {
					maxConnIdx = neuron.lastCon
				}
				
				// Check each connection index
				for j := neuron.firstCon; j < neuron.lastCon; j++ {
					if j >= len(net.connections) {
						t.Errorf("Neuron %d has connection index %d >= connections array len %d",
							i, j, len(net.connections))
					}
					if j >= len(net.weights) {
						t.Errorf("Neuron %d has connection index %d >= weights array len %d",
							i, j, len(net.weights))
					}
					
					// Check source neuron index
					srcNeuron := net.connections[j]
					if srcNeuron < 0 || srcNeuron >= len(net.neurons) {
						t.Errorf("Connection %d points to invalid neuron %d (total neurons: %d)",
							j, srcNeuron, len(net.neurons))
					}
				}
			}
			
			fmt.Printf("Config %v: OK (neurons=%d, connections=%d, maxConnIdx=%d)\n",
				config, len(net.neurons), len(net.connections), maxConnIdx)
		})
	}
}

func TestSparseNetwork(t *testing.T) {
	// Test sparse networks which might have connection rate issues
	rates := []float32{0.1, 0.5, 0.9, 1.0}
	
	for _, rate := range rates {
		t.Run(fmt.Sprintf("Rate_%.1f", rate), func(t *testing.T) {
			net := CreateSparse[float32](rate, 5, 4, 3)
			if net == nil {
				t.Fatal("Failed to create sparse network")
			}
			
			fmt.Printf("Sparse network (rate=%.1f): neurons=%d, connections=%d\n",
				rate, len(net.neurons), len(net.connections))
			
			// Verify network can run
			input := make([]float32, 5)
			for i := range input {
				input[i] = float32(i) * 0.1
			}
			
			output := net.Run(input)
			if output == nil {
				t.Fatal("Run failed")
			}
			fmt.Printf("  Run successful, output: %v\n", output)
		})
	}
}