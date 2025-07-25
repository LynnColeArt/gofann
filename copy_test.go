package gofann

import (
	"testing"
)

func TestCopy(t *testing.T) {
	// Create a network
	ann := CreateStandard[float32](2, 3, 1)
	
	// Set some parameters to non-default values
	ann.SetLearningRate(0.5)
	ann.SetLearningMomentum(0.1)
	ann.SetTrainingAlgorithm(TrainQuickprop)
	ann.SetRpropDeltaMax(100)
	
	// Train it a bit to create internal state
	inputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	outputs := [][]float32{{0}, {1}, {1}, {0}}
	data := CreateTrainDataArray(inputs, outputs)
	ann.TrainEpoch(data)
	
	// Create a copy
	copy := ann.Copy()
	
	// Verify the copy is not nil
	if copy == nil {
		t.Fatal("Copy() returned nil")
	}
	
	// Verify parameters were copied
	if copy.GetLearningRate() != ann.GetLearningRate() {
		t.Errorf("Learning rate not copied: %f != %f", copy.GetLearningRate(), ann.GetLearningRate())
	}
	if copy.GetLearningMomentum() != ann.GetLearningMomentum() {
		t.Errorf("Learning momentum not copied: %f != %f", copy.GetLearningMomentum(), ann.GetLearningMomentum())
	}
	if copy.GetTrainingAlgorithm() != ann.GetTrainingAlgorithm() {
		t.Errorf("Training algorithm not copied: %v != %v", copy.GetTrainingAlgorithm(), ann.GetTrainingAlgorithm())
	}
	if copy.GetRpropDeltaMax() != ann.GetRpropDeltaMax() {
		t.Errorf("RPROP delta max not copied: %f != %f", copy.GetRpropDeltaMax(), ann.GetRpropDeltaMax())
	}
	
	// Verify structure was copied
	if copy.GetNumInput() != ann.GetNumInput() {
		t.Errorf("Number of inputs not copied: %d != %d", copy.GetNumInput(), ann.GetNumInput())
	}
	if copy.GetNumOutput() != ann.GetNumOutput() {
		t.Errorf("Number of outputs not copied: %d != %d", copy.GetNumOutput(), ann.GetNumOutput())
	}
	if copy.GetTotalNeurons() != ann.GetTotalNeurons() {
		t.Errorf("Total neurons not copied: %d != %d", copy.GetTotalNeurons(), ann.GetTotalNeurons())
	}
	if copy.GetTotalConnections() != ann.GetTotalConnections() {
		t.Errorf("Total connections not copied: %d != %d", copy.GetTotalConnections(), ann.GetTotalConnections())
	}
	
	// Verify weights were copied
	for i := 0; i < len(ann.weights); i++ {
		if copy.weights[i] != ann.weights[i] {
			t.Errorf("Weight %d not copied: %f != %f", i, copy.weights[i], ann.weights[i])
			break
		}
	}
	
	// Verify the networks produce the same output
	for _, input := range inputs {
		origOutput := ann.Run(input)
		copyOutput := copy.Run(input)
		
		if len(origOutput) != len(copyOutput) {
			t.Errorf("Output length mismatch: %d != %d", len(origOutput), len(copyOutput))
			continue
		}
		
		for i := range origOutput {
			if origOutput[i] != copyOutput[i] {
				t.Errorf("Output mismatch for input %v: %f != %f", input, origOutput[i], copyOutput[i])
			}
		}
	}
	
	// Verify modifying the copy doesn't affect the original
	copy.SetLearningRate(0.9)
	if ann.GetLearningRate() == 0.9 {
		t.Error("Modifying copy affected original")
	}
	
	// Verify training the copy doesn't affect the original
	origMSE := ann.TestData(data)
	copy.TrainEpoch(data)
	newMSE := ann.TestData(data)
	if origMSE != newMSE {
		t.Error("Training copy affected original MSE")
	}
}

func TestCopyNil(t *testing.T) {
	var ann *Fann[float32]
	copy := ann.Copy()
	if copy != nil {
		t.Error("Copy of nil network should return nil")
	}
}

func TestCopyCascade(t *testing.T) {
	// Create a cascade network
	ann := CreateCascade[float32](2, 1)
	
	// Set cascade parameters
	ann.SetCascadeOutputChangeFraction(0.02)
	ann.SetCascadeWeightMultiplier(0.5)
	
	// Create a copy
	copy := ann.Copy()
	
	// Verify cascade parameters were copied
	if copy.GetCascadeOutputChangeFraction() != ann.GetCascadeOutputChangeFraction() {
		t.Errorf("Cascade output change fraction not copied: %f != %f", 
			copy.GetCascadeOutputChangeFraction(), ann.GetCascadeOutputChangeFraction())
	}
	if copy.GetCascadeWeightMultiplier() != ann.GetCascadeWeightMultiplier() {
		t.Errorf("Cascade weight multiplier not copied: %f != %f", 
			copy.GetCascadeWeightMultiplier(), ann.GetCascadeWeightMultiplier())
	}
	
	// Verify cascade activation functions were copied
	if len(copy.cascadeActivationFunctions) != len(ann.cascadeActivationFunctions) {
		t.Errorf("Cascade activation functions not copied properly")
	}
}