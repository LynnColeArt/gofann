package gofann

import (
	"math"
	"testing"
)

// TestCascadeEdgeCases tests cascade training with edge cases
func TestCascadeEdgeCases(t *testing.T) {
	t.Run("ZeroMaxNeurons", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		data := CreateTrainDataArray([][]float32{{0, 0}}, [][]float32{{1}})
		
		initialNeurons := net.GetTotalNeurons()
		net.CascadetrainOnData(data, 0, 1, 0.01) // maxNeurons = 0
		finalNeurons := net.GetTotalNeurons()
		
		if finalNeurons != initialNeurons {
			t.Errorf("Expected no neurons added with maxNeurons=0, but got %d -> %d",
				initialNeurons, finalNeurons)
		}
	})

	t.Run("NegativeMaxNeurons", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		data := CreateTrainDataArray([][]float32{{0, 0}}, [][]float32{{1}})
		
		// Should handle negative maxNeurons gracefully
		net.CascadetrainOnData(data, -5, 1, 0.01)
		// Should not crash
	})

	t.Run("HugeMaxNeurons", func(t *testing.T) {
		net := CreateCascade[float32](1, 1)
		data := CreateTrainDataArray([][]float32{{0}}, [][]float32{{1}})
		
		// Set very aggressive stagnation to stop early
		net.SetCascadeOutputStagnationEpochs(1)
		net.SetCascadeCandidateStagnationEpochs(1)
		net.SetCascadeMaxOutEpochs(2)
		net.SetCascadeMaxCandEpochs(2)
		net.SetCascadeMinOutEpochs(0) // Allow immediate stagnation check
		net.SetCascadeMinCandEpochs(0)
		
		// Reduce candidates to speed up test
		net.SetCascadeActivationFunctions([]ActivationFunc{Sigmoid})
		net.SetCascadeActivationSteepnesses([]float32{0.5})
		net.SetCascadeNumCandidateGroups(1)
		
		// Add callback to monitor progress
		neuronsAdded := 0
		net.SetCallback(func(ann *Fann[float32], epochs int, mse float32) bool {
			if ann.GetTotalNeurons() > 3 + neuronsAdded {
				neuronsAdded++
				t.Logf("Added neuron %d, MSE: %.4f", neuronsAdded, mse)
			}
			// Stop after 5 neurons to prevent runaway
			return neuronsAdded < 5
		})
		
		initialNeurons := net.GetTotalNeurons()
		net.CascadetrainOnData(data, 1000000, 1, 0.0001) // Huge maxNeurons
		finalNeurons := net.GetTotalNeurons()
		
		// Should stop due to saturation, not add millions
		totalAdded := finalNeurons-initialNeurons
		if totalAdded > 100 {
			t.Errorf("Expected to stop before 100 neurons, but added %d", totalAdded)
		}
		t.Logf("Stopped after adding %d neurons", totalAdded)
	})

	t.Run("ZeroNeuronsPerCascade", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		data := CreateTrainDataArray([][]float32{{0, 0}}, [][]float32{{1}})
		
		// neuronsPerCascade = 0 should be handled
		net.CascadetrainOnData(data, 5, 0, 0.01)
		// Should not crash
	})

	t.Run("EmptyTrainingData", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		data := &TrainData[float32]{
			numData:   0,
			numInput:  2,
			numOutput: 1,
			inputs:    [][]float32{},
			outputs:   [][]float32{},
		}
		
		// Should handle empty data gracefully
		net.CascadetrainOnData(data, 5, 1, 0.01)
		// Should not crash
		
		// Network should be unchanged
		if net.GetTotalNeurons() != 4 { // 2 input + 1 bias + 1 output
			t.Errorf("Network changed with empty data")
		}
	})

	t.Run("SingleTrainingSample", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		data := CreateTrainDataArray([][]float32{{0.5, 0.5}}, [][]float32{{0.7}})
		
		initialMSE := net.TestData(data)
		net.CascadetrainOnData(data, 3, 1, 0.01)
		finalMSE := net.TestData(data)
		
		// Should improve on single sample (cascade may not perfectly fit)
		if finalMSE >= initialMSE {
			t.Errorf("Failed to improve on single sample, MSE: %f -> %f", initialMSE, finalMSE)
		}
		t.Logf("Single sample MSE: %f -> %f", initialMSE, finalMSE)
	})

	t.Run("MismatchedDataDimensions", func(t *testing.T) {
		net := CreateCascade[float32](2, 1) // 2 inputs, 1 output
		
		// Wrong input size
		data := CreateTrainDataArray([][]float32{{0, 0, 0}}, [][]float32{{1}})
		net.CascadetrainOnData(data, 5, 1, 0.01)
		
		// Wrong output size
		data = CreateTrainDataArray([][]float32{{0, 0}}, [][]float32{{1, 0}})
		net.CascadetrainOnData(data, 5, 1, 0.01)
		
		// Should handle gracefully without crash
	})

	t.Run("NaNInData", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		data := CreateTrainDataArray(
			[][]float32{{float32(math.NaN()), 0.5}, {0.5, 0.5}},
			[][]float32{{1}, {0}},
		)
		
		net.CascadetrainOnData(data, 3, 1, 0.01)
		// Should not crash, though results may be undefined
	})

	t.Run("InfiniteInData", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		data := CreateTrainDataArray(
			[][]float32{{float32(math.Inf(1)), 0.5}, {0.5, 0.5}},
			[][]float32{{1}, {0}},
		)
		
		net.CascadetrainOnData(data, 3, 1, 0.01)
		// Should not crash
	})
}

// TestCascadeParameterBounds tests parameter edge cases
func TestCascadeParameterBounds(t *testing.T) {
	net := CreateCascade[float32](2, 1)
	
	tests := []struct {
		name   string
		setter func(float32)
		getter func() float32
		values []float32
	}{
		{
			name:   "OutputChangeFraction",
			setter: func(v float32) { net.SetCascadeOutputChangeFraction(v) },
			getter: func() float32 { return net.GetCascadeOutputChangeFraction() },
			values: []float32{-1, 0, 0.000001, 1, 2, 1000000},
		},
		{
			name:   "CandidateChangeFraction",
			setter: func(v float32) { net.SetCascadeCandidateChangeFraction(v) },
			getter: func() float32 { return net.GetCascadeCandidateChangeFraction() },
			values: []float32{-1, 0, 0.000001, 1, 2, 1000000},
		},
		{
			name:   "CandidateLimit",
			setter: func(v float32) { net.SetCascadeCandidateLimit(v) },
			getter: func() float32 { return net.GetCascadeCandidateLimit() },
			values: []float32{-1000, -1, 0, 0.001, 1, 1000000},
		},
	}
	
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, v := range test.values {
				test.setter(v)
				got := test.getter()
				// Some values might be clamped or converted
				t.Logf("%s: set %f, got %f", test.name, v, got)
			}
		})
	}
	
	// Test epoch parameters with negative/zero/huge values
	epochTests := []struct {
		name   string
		setter func(int)
		getter func() int
		values []int
	}{
		{
			name:   "OutputStagnationEpochs",
			setter: func(v int) { net.SetCascadeOutputStagnationEpochs(v) },
			getter: func() int { return net.GetCascadeOutputStagnationEpochs() },
			values: []int{-1, 0, 1, 1000000},
		},
		{
			name:   "MaxOutEpochs",
			setter: func(v int) { net.SetCascadeMaxOutEpochs(v) },
			getter: func() int { return net.GetCascadeMaxOutEpochs() },
			values: []int{-1, 0, 1, 1000000},
		},
	}
	
	for _, test := range epochTests {
		t.Run(test.name, func(t *testing.T) {
			for _, v := range test.values {
				test.setter(v)
				got := test.getter()
				if got != v {
					t.Errorf("%s: set %d, got %d", test.name, v, got)
				}
			}
		})
	}
}

// TestCascadeActivationFunctions tests all activation function combinations
func TestCascadeActivationFunctions(t *testing.T) {
	// Test with empty activation functions
	t.Run("EmptyActivationFunctions", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		net.SetCascadeActivationFunctions([]ActivationFunc{})
		
		data := CreateTrainDataArray([][]float32{{0, 0}}, [][]float32{{1}})
		net.CascadetrainOnData(data, 2, 1, 0.01)
		// Should handle empty activation functions
	})
	
	// Test with nil steepnesses
	t.Run("EmptySteepnesses", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		net.SetCascadeActivationSteepnesses([]float32{})
		
		data := CreateTrainDataArray([][]float32{{0, 0}}, [][]float32{{1}})
		net.CascadetrainOnData(data, 2, 1, 0.01)
		// Should handle empty steepnesses
	})
	
	// Test with extreme steepnesses
	t.Run("ExtremeSteepnesses", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		net.SetCascadeActivationSteepnesses([]float32{-1, 0, 0.00001, 100000})
		
		data := CreateTrainDataArray([][]float32{{0, 0}, {1, 1}}, [][]float32{{0}, {1}})
		net.CascadetrainOnData(data, 2, 1, 0.01)
		// Should handle extreme steepnesses
	})
}

// TestCascadeConcurrency tests cascade training under concurrent conditions
func TestCascadeConcurrency(t *testing.T) {
	t.Run("CallbackPanic", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		data := CreateTrainDataArray([][]float32{{0, 0}, {1, 1}}, [][]float32{{0}, {1}})
		
		callCount := 0
		net.SetCallback(func(ann *Fann[float32], epochs int, mse float32) bool {
			callCount++
			if callCount == 5 {
				panic("Callback panic test")
			}
			return true
		})
		
		// Should recover from panic
		defer func() {
			if r := recover(); r != nil {
				t.Logf("Recovered from panic: %v", r)
			}
		}()
		
		net.CascadetrainOnData(data, 3, 1, 0.01)
	})
	
	t.Run("CallbackReturnsFalse", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		data := CreateTrainDataArray([][]float32{{0, 0}, {1, 1}}, [][]float32{{0}, {1}})
		
		callCount := 0
		net.SetCallback(func(ann *Fann[float32], epochs int, mse float32) bool {
			callCount++
			return callCount < 3 // Stop after 3 calls
		})
		
		net.CascadetrainOnData(data, 10, 1, 0.0001)
		
		// Should stop early due to callback
		if callCount > 5 {
			t.Errorf("Callback should have stopped training, but got %d calls", callCount)
		}
	})
}

// TestCascadeMemoryStress tests cascade with memory pressure
func TestCascadeMemoryStress(t *testing.T) {
	t.Run("ManyActivationFunctions", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		
		// Set up many activation functions and steepnesses
		funcs := make([]ActivationFunc, 20)
		for i := range funcs {
			funcs[i] = ActivationFunc(i % 10)
		}
		net.SetCascadeActivationFunctions(funcs)
		
		steeps := make([]float32, 20)
		for i := range steeps {
			steeps[i] = float32(i)*0.1 + 0.1
		}
		net.SetCascadeActivationSteepnesses(steeps)
		
		// This creates 400 candidates per group
		net.SetCascadeNumCandidateGroups(2)
		
		data := CreateTrainDataArray([][]float32{{0, 0}, {1, 1}}, [][]float32{{0}, {1}})
		net.CascadetrainOnData(data, 2, 1, 0.01)
		
		// Should handle large number of candidates
	})
	
	t.Run("LargeNetwork", func(t *testing.T) {
		// Start with larger network
		net := CreateStandard[float32](10, 5, 3)
		
		// Create larger dataset
		inputs := make([][]float32, 20)
		outputs := make([][]float32, 20)
		for i := 0; i < 20; i++ {
			inputs[i] = make([]float32, 10)
			outputs[i] = make([]float32, 3)
			for j := 0; j < 10; j++ {
				inputs[i][j] = float32(i*j) / 100.0
			}
			outputs[i][i%3] = 1
		}
		data := CreateTrainDataArray(inputs, outputs)
		
		// Aggressive cascade settings
		net.SetCascadeMaxOutEpochs(5)
		net.SetCascadeMaxCandEpochs(5)
		net.SetCascadeOutputStagnationEpochs(2)
		
		initialConnections := net.GetTotalConnections()
		net.CascadetrainOnData(data, 5, 1, 0.01)
		finalConnections := net.GetTotalConnections()
		
		t.Logf("Connections grew from %d to %d", initialConnections, finalConnections)
	})
}

// TestCascadeNumericalStability tests numerical edge cases
func TestCascadeNumericalStability(t *testing.T) {
	t.Run("TinyWeights", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		
		// Force tiny weight initialization
		net.RandomizeWeights(1e-10, 1e-9)
		
		data := CreateTrainDataArray([][]float32{{0, 0}, {1, 1}}, [][]float32{{0}, {1}})
		net.CascadetrainOnData(data, 2, 1, 0.1)
		
		// Check for NaN/Inf in outputs
		output := net.Run([]float32{0.5, 0.5})
		if math.IsNaN(float64(output[0])) || math.IsInf(float64(output[0]), 0) {
			t.Errorf("Got NaN or Inf output: %v", output)
		}
	})
	
	t.Run("HugeWeights", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		
		// Force huge weight initialization
		net.RandomizeWeights(1000, 10000)
		
		data := CreateTrainDataArray([][]float32{{0.001, 0.001}}, [][]float32{{0.5}})
		net.CascadetrainOnData(data, 1, 1, 0.1)
		
		// Should still produce finite outputs
		output := net.Run([]float32{0.001, 0.001})
		if math.IsNaN(float64(output[0])) || math.IsInf(float64(output[0]), 0) {
			t.Errorf("Got NaN or Inf output with huge weights: %v", output)
		}
	})
}

// TestCascadeStateCorruption tests for state corruption
func TestCascadeStateCorruption(t *testing.T) {
	t.Run("ModifyDataDuringTraining", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		inputs := [][]float32{{0, 0}, {1, 1}}
		outputs := [][]float32{{0}, {1}}
		data := CreateTrainDataArray(inputs, outputs)
		
		modified := false
		net.SetCallback(func(ann *Fann[float32], epochs int, mse float32) bool {
			if !modified && epochs > 10 {
				// Corrupt the training data mid-training
				data.inputs[0][0] = float32(math.NaN())
				modified = true
			}
			return true
		})
		
		net.CascadetrainOnData(data, 3, 1, 0.01)
		// Should handle data corruption gracefully
	})
	
	t.Run("RepeatedCascadeTraining", func(t *testing.T) {
		net := CreateCascade[float32](2, 1)
		data := CreateTrainDataArray([][]float32{{0, 0}, {1, 1}}, [][]float32{{0}, {1}})
		
		// Train multiple times
		for i := 0; i < 5; i++ {
			prevNeurons := net.GetTotalNeurons()
			net.CascadetrainOnData(data, 1, 1, 0.01)
			currNeurons := net.GetTotalNeurons()
			
			if currNeurons < prevNeurons {
				t.Errorf("Lost neurons on iteration %d: %d -> %d", 
					i, prevNeurons, currNeurons)
			}
		}
	})
}

// TestCascadeWithDifferentNetworkTypes tests cascade on various network types
func TestCascadeWithDifferentNetworkTypes(t *testing.T) {
	t.Run("SparseNetwork", func(t *testing.T) {
		net := CreateSparse[float32](0.5, 3, 2)
		data := CreateTrainDataArray(
			[][]float32{{0, 0, 0}, {1, 1, 1}},
			[][]float32{{0, 1}, {1, 0}},
		)
		
		net.CascadetrainOnData(data, 2, 1, 0.1)
		// Should work with sparse networks
	})
	
	t.Run("ExistingHiddenLayers", func(t *testing.T) {
		// Start with network that already has hidden layers
		net := CreateStandard[float32](3, 4, 2, 1)
		data := CreateTrainDataArray(
			[][]float32{{0, 0, 0}, {1, 1, 1}},
			[][]float32{{0}, {1}},
		)
		
		initialLayers := len(net.layers)
		net.CascadetrainOnData(data, 3, 1, 0.1)
		finalLayers := len(net.layers)
		
		if finalLayers != initialLayers {
			t.Errorf("Layer count changed: %d -> %d", initialLayers, finalLayers)
		}
	})
}