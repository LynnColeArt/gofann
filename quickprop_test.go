package gofann

import (
	"testing"
)

func TestQuickprop(t *testing.T) {
	// Create XOR training data
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
	trainData := CreateTrainDataArray(inputs, outputs)
	
	// Test Quickprop vs other algorithms
	algorithms := []struct {
		name string
		algo TrainAlgorithm
	}{
		{"Incremental", TrainIncremental},
		{"Batch", TrainBatch},
		{"RPROP", TrainRPROP},
		{"Quickprop", TrainQuickprop},
	}
	
	for _, alg := range algorithms {
		t.Run(alg.name, func(t *testing.T) {
			// Create network
			net := CreateStandard[float32](2, 4, 1)
			net.SetTrainingAlgorithm(alg.algo)
			net.SetLearningRate(0.7)
			
			// For Quickprop, set parameters
			if alg.algo == TrainQuickprop {
				net.SetQuickpropDecay(-0.0001)
				net.SetQuickpropMu(1.75)
			}
			
			// Train
			initialMSE := net.TestData(trainData)
			net.TrainOnData(trainData, 2000, 0, 0.01)
			finalMSE := net.TestData(trainData)
			
			t.Logf("%s: Initial MSE=%.4f, Final MSE=%.4f", 
				alg.name, initialMSE, finalMSE)
			
			// Test outputs
			for i, input := range inputs {
				output := net.Run(input)
				expected := outputs[i][0]
				t.Logf("  %v -> %.3f (expected %.0f)", 
					input, output[0], expected)
			}
			
			// Check if it learned
			if finalMSE > 0.1 {
				t.Logf("Warning: %s did not converge well (MSE=%.4f)", 
					alg.name, finalMSE)
			}
		})
	}
}

func TestQuickpropOnMushroom(t *testing.T) {
	// Load mushroom dataset
	trainPath := "/media/lynn/big_drive/workspaces/fanmaker/fann_source/datasets/mushroom.train"
	trainData, err := ReadTrainFromFile[float32](trainPath)
	if err != nil {
		t.Skipf("Mushroom dataset not found: %v", err)
	}
	
	// Use a subset for faster testing
	subset := trainData.Subset(0, 500)
	
	// Create network
	net := CreateStandard[float32](125, 30, 2)
	net.SetTrainingAlgorithm(TrainQuickprop)
	net.SetLearningRate(0.7)
	net.SetQuickpropDecay(-0.0001)
	net.SetQuickpropMu(1.75)
	
	// Track progress
	epochMSEs := []float32{}
	net.SetCallback(func(ann *Fann[float32], epochs int, mse float32) bool {
		epochMSEs = append(epochMSEs, mse)
		return true
	})
	
	// Train
	net.TrainOnData(subset, 20, 5, 0.001)
	
	// Log progress
	t.Log("Quickprop training progress:")
	for i, mse := range epochMSEs {
		t.Logf("  Epoch %d: MSE=%.6f", (i+1)*5, mse)
	}
	
	// Test accuracy
	correct := 0
	for i := 0; i < subset.GetNumData(); i++ {
		input := subset.GetInput(i)
		output := net.Run(input)
		expected := subset.GetOutput(i)
		
		// Determine predicted class
		predictedClass := 0
		if output[1] > output[0] {
			predictedClass = 1
		}
		
		// Determine expected class
		expectedClass := 0
		if expected[1] > expected[0] {
			expectedClass = 1
		}
		
		if predictedClass == expectedClass {
			correct++
		}
	}
	
	accuracy := float32(correct) / float32(subset.GetNumData()) * 100
	t.Logf("Final accuracy: %.1f%% (%d/%d)", accuracy, correct, subset.GetNumData())
	
	if accuracy < 90 {
		t.Errorf("Quickprop accuracy too low: %.1f%%", accuracy)
	}
}