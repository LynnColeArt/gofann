package gofann

import (
	"testing"
)

func TestSarprop(t *testing.T) {
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
	
	// Compare Sarprop with other algorithms
	algorithms := []struct {
		name string
		algo TrainAlgorithm
	}{
		{"RPROP", TrainRPROP},
		{"Sarprop", TrainSarprop},
		{"Incremental", TrainIncremental},
	}
	
	for _, alg := range algorithms {
		t.Run(alg.name, func(t *testing.T) {
			// Create network
			net := CreateStandard[float32](2, 4, 1)
			net.SetTrainingAlgorithm(alg.algo)
			net.SetLearningRate(0.7)
			
			// Train
			initialMSE := net.TestData(trainData)
			epochs := 0
			net.SetCallback(func(ann *Fann[float32], ep int, mse float32) bool {
				epochs = ep
				return true
			})
			
			net.TrainOnData(trainData, 1000, 200, 0.01)
			finalMSE := net.TestData(trainData)
			
			t.Logf("%s: Initial MSE=%.4f, Final MSE=%.4f (epochs=%d)", 
				alg.name, initialMSE, finalMSE, epochs)
			
			// Test outputs
			for i, input := range inputs {
				output := net.Run(input)
				expected := outputs[i][0]
				t.Logf("  %v -> %.3f (expected %.0f)", 
					input, output[0], expected)
			}
			
			// Check convergence
			if finalMSE > 0.1 {
				t.Logf("Warning: %s did not converge well (MSE=%.4f)", 
					alg.name, finalMSE)
			}
		})
	}
}

func TestSarpropParameters(t *testing.T) {
	net := CreateStandard[float32](2, 2, 1)
	
	// Test parameter setters/getters
	net.SetSarpropWeightDecayShift(-5.0)
	if net.GetSarpropWeightDecayShift() != -5.0 {
		t.Errorf("SarpropWeightDecayShift: got %f, want -5.0", 
			net.GetSarpropWeightDecayShift())
	}
	
	net.SetSarpropStepErrorThresholdFactor(0.2)
	if net.GetSarpropStepErrorThresholdFactor() != 0.2 {
		t.Errorf("SarpropStepErrorThresholdFactor: got %f, want 0.2", 
			net.GetSarpropStepErrorThresholdFactor())
	}
	
	net.SetSarpropStepErrorShift(1.5)
	if net.GetSarpropStepErrorShift() != 1.5 {
		t.Errorf("SarpropStepErrorShift: got %f, want 1.5", 
			net.GetSarpropStepErrorShift())
	}
	
	net.SetSarpropTemperature(0.02)
	if net.GetSarpropTemperature() != 0.02 {
		t.Errorf("SarpropTemperature: got %f, want 0.02", 
			net.GetSarpropTemperature())
	}
}

func TestSarpropOnLargerDataset(t *testing.T) {
	// Load mushroom dataset
	trainPath := "/media/lynn/big_drive/workspaces/fanmaker/fann_source/datasets/mushroom.train"
	trainData, err := ReadTrainFromFile[float32](trainPath)
	if err != nil {
		t.Skipf("Mushroom dataset not found: %v", err)
	}
	
	// Use a subset for faster testing
	subset := trainData.Subset(0, 1000)
	
	// Test both RPROP and Sarprop
	algorithms := []TrainAlgorithm{TrainRPROP, TrainSarprop}
	
	for _, algo := range algorithms {
		algoName := "RPROP"
		if algo == TrainSarprop {
			algoName = "Sarprop"
		}
		
		t.Run(algoName, func(t *testing.T) {
			// Create network
			net := CreateStandard[float32](125, 30, 2)
			net.SetTrainingAlgorithm(algo)
			
			// Track MSE over epochs
			mseHistory := []float32{}
			net.SetCallback(func(ann *Fann[float32], epochs int, mse float32) bool {
				mseHistory = append(mseHistory, mse)
				return true
			})
			
			// Train
			net.TrainOnData(subset, 20, 5, 0.001)
			
			// Log progress
			t.Logf("%s MSE progress:", algoName)
			for i, mse := range mseHistory {
				t.Logf("  Epoch %d: %.6f", (i+1)*5, mse)
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
		})
	}
}