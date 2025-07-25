package gofann

import (
	"math"
	"testing"
)

// TestCascadeOutputTrainingDebug debugs the output training phase
func TestCascadeOutputTrainingDebug(t *testing.T) {
	// Simple test: output constant 0.5
	net := CreateStandard[float32](1, 1) // Simple network, no hidden layer
	data := CreateTrainDataArray([][]float32{{0.5}}, [][]float32{{0.5}})
	
	// Set training to RPROP (same as cascade uses)
	net.SetTrainingAlgorithm(TrainRPROP)
	net.SetRpropDeltaZero(0.01)
	net.SetRpropDeltaMax(5.0)
	
	// Get initial state
	initialOutput := net.Run([]float32{0.5})[0]
	initialMSE := net.TestData(data)
	initialWeight := net.weights[0] // First weight
	
	t.Logf("Initial: output=%.6f, MSE=%.6f, weight=%.6f", 
		initialOutput, initialMSE, initialWeight)
	
	// Train for a few epochs and watch what happens
	for epoch := 1; epoch <= 10; epoch++ {
		mse := net.TrainEpoch(data)
		output := net.Run([]float32{0.5})[0]
		weight := net.weights[0]
		
		t.Logf("Epoch %2d: output=%.6f, MSE=%.6f, weight=%.6f, delta=%.6f", 
			epoch, output, mse, weight, weight-initialWeight)
		
		// Check if output is collapsing
		if output < 0.01 {
			t.Logf("ERROR: Output collapsed to near zero!")
			break
		}
	}
	
	// Now test cascade output training specifically
	t.Log("\n--- Testing Cascade Output Training ---")
	
	// Reset network
	net2 := CreateCascade[float32](1, 1)
	
	// Get initial state
	initialOutput2 := net2.Run([]float32{0.5})[0]
	initialMSE2 := net2.TestData(data)
	
	t.Logf("Cascade Initial: output=%.6f, MSE=%.6f", initialOutput2, initialMSE2)
	
	// Manually run cascade output training
	net2.cascadeTrainOutput(data, 0.001)
	
	finalOutput2 := net2.Run([]float32{0.5})[0]
	finalMSE2 := net2.TestData(data)
	
	t.Logf("Cascade After Output Training: output=%.6f, MSE=%.6f", 
		finalOutput2, finalMSE2)
	
	if finalOutput2 < 0.01 {
		t.Error("Cascade output training caused output collapse!")
	}
}

// TestGradientDirection checks if gradients are being applied correctly
func TestGradientDirection(t *testing.T) {
	net := CreateStandard[float32](1, 1)
	
	// Test cases where we know the gradient direction
	testCases := []struct {
		name           string
		input          float32
		target         float32
		currentOutput  float32
		expectedChange string // "increase" or "decrease"
	}{
		{"Below target", 0.5, 0.8, 0.3, "increase"},
		{"Above target", 0.5, 0.2, 0.7, "decrease"},
		{"At target", 0.5, 0.5, 0.5, "stable"},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Set network to produce currentOutput
			// For a simple 1-1 network with sigmoid: output ≈ sigmoid(weight * input)
			// We need to find weight such that sigmoid(weight * 0.5) = currentOutput
			
			if tc.currentOutput > 0.99 {
				tc.currentOutput = 0.99
			} else if tc.currentOutput < 0.01 {
				tc.currentOutput = 0.01
			}
			
			// Inverse sigmoid to find required sum
			requiredSum := float32(math.Log(float64(tc.currentOutput) / 
				(1.0 - float64(tc.currentOutput))))
			requiredWeight := requiredSum / tc.input
			
			net.weights[0] = requiredWeight
			
			// Verify setup
			output := net.Run([]float32{tc.input})[0]
			t.Logf("Setup: weight=%.4f, output=%.4f (wanted %.4f)", 
				requiredWeight, output, tc.currentOutput)
			
			// Create training data
			data := CreateTrainDataArray(
				[][]float32{{tc.input}}, 
				[][]float32{{tc.target}},
			)
			
			// Train one epoch
			initialWeight := net.weights[0]
			net.TrainEpoch(data)
			finalWeight := net.weights[0]
			finalOutput := net.Run([]float32{tc.input})[0]
			
			weightChange := finalWeight - initialWeight
			outputChange := finalOutput - output
			
			t.Logf("After training: weight %.4f->%.4f (Δ=%.6f), output %.4f->%.4f (Δ=%.6f)",
				initialWeight, finalWeight, weightChange,
				output, finalOutput, outputChange)
			
			// Check direction
			switch tc.expectedChange {
			case "increase":
				if outputChange <= 0 {
					t.Errorf("Expected output to increase, but changed by %.6f", 
						outputChange)
				}
			case "decrease":
				if outputChange >= 0 {
					t.Errorf("Expected output to decrease, but changed by %.6f", 
						outputChange)
				}
			case "stable":
				if math.Abs(float64(outputChange)) > 0.01 {
					t.Errorf("Expected output to stay stable, but changed by %.6f", 
						outputChange)
				}
			}
		})
	}
}

// TestCascadeWithDifferentAlgorithms tests if using different algorithms helps
func TestCascadeWithDifferentAlgorithms(t *testing.T) {
	algorithms := []struct {
		name string
		algo TrainAlgorithm
		lr   float32
	}{
		{"Incremental", TrainIncremental, 0.1},
		{"Batch", TrainBatch, 0.1},
		{"RPROP", TrainRPROP, 0},
	}
	
	for _, alg := range algorithms {
		t.Run(alg.name, func(t *testing.T) {
			net := CreateCascade[float32](1, 1)
			net.SetTrainingAlgorithm(alg.algo)
			if alg.lr > 0 {
				net.SetLearningRate(alg.lr)
			}
			
			data := CreateTrainDataArray([][]float32{{0.5}}, [][]float32{{0.5}})
			
			initialOutput := net.Run([]float32{0.5})[0]
			initialMSE := net.TestData(data)
			
			// Run cascade output training
			net.cascadeTrainOutput(data, 0.001)
			
			finalOutput := net.Run([]float32{0.5})[0]
			finalMSE := net.TestData(data)
			
			t.Logf("%s: output %.4f->%.4f, MSE %.4f->%.4f",
				alg.name, initialOutput, finalOutput, initialMSE, finalMSE)
			
			if finalMSE > initialMSE {
				t.Logf("WARNING: %s made MSE worse!", alg.name)
			}
			
			if finalOutput < 0.01 {
				t.Errorf("%s caused output collapse!", alg.name)
			}
		})
	}
}