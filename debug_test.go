package gofann

import (
	"fmt"
	"testing"
)

func TestConnectionSetup(t *testing.T) {
	// Test basic network creation
	net := CreateStandard[float32](2, 3, 1)
	if net == nil {
		t.Fatal("Failed to create network")
	}
	
	fmt.Printf("Network created:\n")
	fmt.Printf("  Total neurons: %d\n", net.totalNeurons)
	fmt.Printf("  Total connections: %d\n", net.totalConnections)
	fmt.Printf("  Connections array len: %d\n", len(net.connections))
	fmt.Printf("  Weights array len: %d\n", len(net.weights))
	
	// Check each neuron's connections
	for i, neuron := range net.neurons {
		fmt.Printf("Neuron %d: firstCon=%d, lastCon=%d\n", i, neuron.firstCon, neuron.lastCon)
		if neuron.lastCon > len(net.connections) {
			t.Errorf("Neuron %d has lastCon=%d but connections array len=%d", 
				i, neuron.lastCon, len(net.connections))
		}
	}
}

func TestRunNetwork(t *testing.T) {
	net := CreateStandard[float32](2, 3, 1)
	if net == nil {
		t.Fatal("Failed to create network")
	}
	
	// Try to run the network
	input := []float32{0.5, 0.5}
	output := net.Run(input)
	if output == nil {
		t.Fatal("Run returned nil")
	}
	
	fmt.Printf("Input: %v => Output: %v\n", input, output)
}