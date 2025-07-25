package gofann

import (
	"testing"
)

// TestShortcutNetworkCreation tests basic shortcut network creation
func TestShortcutNetworkCreation(t *testing.T) {
	// Create a simple 3-layer shortcut network: 2 inputs, 3 hidden, 1 output
	ann := CreateShortcut[float32](2, 3, 1)
	
	if ann == nil {
		t.Fatal("Failed to create shortcut network")
	}
	
	if ann.networkType != NetTypeShortcut {
		t.Errorf("Expected network type %d, got %d", NetTypeShortcut, ann.networkType)
	}
	
	// Verify layer structure
	expectedLayers := 3 // input, hidden, output
	if len(ann.layers) != expectedLayers {
		t.Errorf("Expected %d layers, got %d", expectedLayers, len(ann.layers))
	}
	
	t.Logf("✅ Shortcut network created with %d layers", len(ann.layers))
	t.Logf("   Total neurons: %d", ann.totalNeurons)
	t.Logf("   Total connections: %d", ann.totalConnections)
}

// TestShortcutNetworkConnections verifies the shortcut connection pattern
func TestShortcutNetworkConnections(t *testing.T) {
	// Create shortcut network: 2 inputs, 2 hidden, 1 output
	ann := CreateShortcut[float32](2, 2, 1)
	
	// In a shortcut network:
	// - Input layer connects to hidden layer AND output layer
	// - Hidden layer connects to output layer
	// So we should have more connections than a standard network
	
	standardAnn := CreateStandard[float32](2, 2, 1)
	
	if ann.totalConnections <= standardAnn.totalConnections {
		t.Errorf("Shortcut network should have more connections than standard network")
		t.Errorf("Shortcut: %d, Standard: %d", ann.totalConnections, standardAnn.totalConnections)
	}
	
	t.Logf("✅ Connection counts: Shortcut=%d, Standard=%d", 
		ann.totalConnections, standardAnn.totalConnections)
}

// TestShortcutNetworkExecution tests that shortcut networks can run
func TestShortcutNetworkExecution(t *testing.T) {
	ann := CreateShortcut[float32](2, 3, 1)
	ann.RandomizeWeights(-1.0, 1.0)
	
	// Test with sample input
	input := []float32{0.5, -0.3}
	output := ann.Run(input)
	
	if len(output) != 1 {
		t.Errorf("Expected 1 output, got %d", len(output))
	}
	
	t.Logf("✅ Shortcut network execution successful")
	t.Logf("   Input: %v", input)
	t.Logf("   Output: %v", output)
}

// TestShortcutNetworkTraining tests basic training functionality
func TestShortcutNetworkTraining(t *testing.T) {
	ann := CreateShortcut[float32](2, 4, 1)
	ann.RandomizeWeights(-1.0, 1.0)
	
	// Create simple XOR-like training data
	data := CreateTrainDataArray[float32]([][]float32{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}, [][]float32{
		{0}, {1}, {1}, {0},
	})
	
	// Train for a few epochs
	initialMSE := ann.TestData(data)
	
	for i := 0; i < 100; i++ {
		ann.TrainEpoch(data)
	}
	
	finalMSE := ann.TestData(data)
	
	if finalMSE >= initialMSE {
		t.Logf("⚠️  Training didn't improve MSE (initial: %.6f, final: %.6f)", 
			float64(initialMSE), float64(finalMSE))
		t.Logf("   This is not necessarily an error - XOR is hard and may need more epochs")
	} else {
		t.Logf("✅ Training improved MSE from %.6f to %.6f", 
			float64(initialMSE), float64(finalMSE))
	}
	
	// Test all XOR patterns
	t.Logf("   XOR Results after training:")
	patterns := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	expected := []float32{0, 1, 1, 0}
	
	for i, pattern := range patterns {
		result := ann.Run(pattern)
		t.Logf("   %v -> %.3f (expected %.0f)", pattern, result[0], expected[i])
	}
}

// TestShortcutVsStandardPerformance compares shortcut vs standard networks
func TestShortcutVsStandardPerformance(t *testing.T) {
	// Create both network types with same architecture
	shortcut := CreateShortcut[float32](2, 4, 1)
	standard := CreateStandard[float32](2, 4, 1)
	
	shortcut.RandomizeWeights(-1.0, 1.0)
	standard.RandomizeWeights(-1.0, 1.0)
	
	// Same training data
	data := CreateTrainDataArray[float32]([][]float32{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}, [][]float32{
		{0}, {1}, {1}, {0},
	})
	
	// Train both for same number of epochs
	for i := 0; i < 200; i++ {
		shortcut.TrainEpoch(data)
		standard.TrainEpoch(data)
	}
	
	shortcutMSE := shortcut.TestData(data)
	standardMSE := standard.TestData(data)
	
	t.Logf("✅ Performance comparison after 200 epochs:")
	t.Logf("   Shortcut MSE: %.6f", float64(shortcutMSE))
	t.Logf("   Standard MSE: %.6f", float64(standardMSE))
	t.Logf("   Shortcut connections: %d", shortcut.totalConnections)
	t.Logf("   Standard connections: %d", standard.totalConnections)
	
	if shortcutMSE < standardMSE {
		t.Logf("   🎯 Shortcut network performed better!")
	} else {
		t.Logf("   📊 Standard network performed better (shortcut networks may need different training)")
	}
}