package gofann

import (
	"testing"
)

func TestCascadeTraining(t *testing.T) {
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

	t.Run("BasicCascade", func(t *testing.T) {
		// Create minimal network (just input and output)
		net := CreateStandard[float32](2, 1)
		
		// Set cascade parameters
		net.SetCascadeOutputChangeFraction(0.01)
		net.SetCascadeOutputStagnationEpochs(10)
		net.SetCascadeCandidateChangeFraction(0.01)
		net.SetCascadeCandidateStagnationEpochs(10)
		net.SetCascadeMaxOutEpochs(50)
		net.SetCascadeMaxCandEpochs(50)
		net.SetCascadeMinOutEpochs(10)
		net.SetCascadeMinCandEpochs(10)
		
		// Track progress
		addedNeurons := 0
		net.SetCallback(func(ann *Fann[float32], epochs int, mse float32) bool {
			if ann.GetTotalNeurons() > 3 + addedNeurons { // 2 input + 1 output + bias
				addedNeurons = ann.GetTotalNeurons() - 3
				t.Logf("Added neuron %d, MSE: %.4f", addedNeurons, mse)
			}
			return true
		})
		
		// Train with cascade
		initialNeurons := net.GetTotalNeurons()
		net.CascadetrainOnData(trainData, 10, 1, 0.01)
		finalNeurons := net.GetTotalNeurons()
		
		t.Logf("Initial neurons: %d, Final neurons: %d", 
			initialNeurons, finalNeurons)
		
		// Test the network
		finalMSE := net.TestData(trainData)
		t.Logf("Final MSE: %.4f", finalMSE)
		
		// Verify it added neurons
		if finalNeurons <= initialNeurons {
			t.Errorf("Cascade training did not add neurons")
		}
		
		// Test outputs
		for i, input := range inputs {
			output := net.Run(input)
			expected := outputs[i][0]
			t.Logf("  %v -> %.3f (expected %.0f)", 
				input, output[0], expected)
		}
	})
	
	t.Run("CascadeParameters", func(t *testing.T) {
		net := CreateStandard[float32](2, 1)
		
		// Test all parameter setters/getters
		net.SetCascadeOutputChangeFraction(0.02)
		if net.GetCascadeOutputChangeFraction() != 0.02 {
			t.Errorf("CascadeOutputChangeFraction: got %f, want 0.02",
				net.GetCascadeOutputChangeFraction())
		}
		
		net.SetCascadeOutputStagnationEpochs(15)
		if net.GetCascadeOutputStagnationEpochs() != 15 {
			t.Errorf("CascadeOutputStagnationEpochs: got %d, want 15",
				net.GetCascadeOutputStagnationEpochs())
		}
		
		net.SetCascadeCandidateChangeFraction(0.03)
		if net.GetCascadeCandidateChangeFraction() != 0.03 {
			t.Errorf("CascadeCandidateChangeFraction: got %f, want 0.03",
				net.GetCascadeCandidateChangeFraction())
		}
		
		net.SetCascadeCandidateStagnationEpochs(20)
		if net.GetCascadeCandidateStagnationEpochs() != 20 {
			t.Errorf("CascadeCandidateStagnationEpochs: got %d, want 20",
				net.GetCascadeCandidateStagnationEpochs())
		}
		
		net.SetCascadeCandidateLimit(2000.0)
		if net.GetCascadeCandidateLimit() != 2000.0 {
			t.Errorf("CascadeCandidateLimit: got %f, want 2000.0",
				net.GetCascadeCandidateLimit())
		}
		
		net.SetCascadeMaxOutEpochs(200)
		if net.GetCascadeMaxOutEpochs() != 200 {
			t.Errorf("CascadeMaxOutEpochs: got %d, want 200",
				net.GetCascadeMaxOutEpochs())
		}
		
		net.SetCascadeMaxCandEpochs(250)
		if net.GetCascadeMaxCandEpochs() != 250 {
			t.Errorf("CascadeMaxCandEpochs: got %d, want 250",
				net.GetCascadeMaxCandEpochs())
		}
		
		net.SetCascadeMinOutEpochs(25)
		if net.GetCascadeMinOutEpochs() != 25 {
			t.Errorf("CascadeMinOutEpochs: got %d, want 25",
				net.GetCascadeMinOutEpochs())
		}
		
		net.SetCascadeMinCandEpochs(30)
		if net.GetCascadeMinCandEpochs() != 30 {
			t.Errorf("CascadeMinCandEpochs: got %d, want 30",
				net.GetCascadeMinCandEpochs())
		}
		
		// Test activation functions
		funcs := []ActivationFunc{Sigmoid, SigmoidSymmetric, Linear}
		net.SetCascadeActivationFunctions(funcs)
		gotFuncs := net.GetCascadeActivationFunctions()
		if len(gotFuncs) != len(funcs) {
			t.Errorf("CascadeActivationFunctions: got %d functions, want %d",
				len(gotFuncs), len(funcs))
		}
		
		// Test activation steepnesses
		steeps := []float32{0.1, 0.5, 1.0}
		net.SetCascadeActivationSteepnesses(steeps)
		gotSteeps := net.GetCascadeActivationSteepnesses()
		if len(gotSteeps) != len(steeps) {
			t.Errorf("CascadeActivationSteepnesses: got %d values, want %d",
				len(gotSteeps), len(steeps))
		}
		
		net.SetCascadeNumCandidateGroups(3)
		if net.GetCascadeNumCandidateGroups() != 3 {
			t.Errorf("CascadeNumCandidateGroups: got %d, want 3",
				net.GetCascadeNumCandidateGroups())
		}
	})
}

func TestCascadeOnLargerDataset(t *testing.T) {
	// Load diabetes dataset - good for testing cascade
	trainPath := "/media/lynn/big_drive/workspaces/fanmaker/fann_source/datasets/diabetes.train"
	trainData, err := ReadTrainFromFile[float32](trainPath)
	if err != nil {
		t.Skipf("Diabetes dataset not found: %v", err)
	}
	
	// Use a subset for faster testing
	subset := trainData.Subset(0, 200)
	
	// Start with minimal network
	net := CreateStandard[float32](8, 2)
	
	// Configure cascade parameters for quick testing
	net.SetCascadeMaxOutEpochs(20)
	net.SetCascadeMaxCandEpochs(20)
	net.SetCascadeMinOutEpochs(5)
	net.SetCascadeMinCandEpochs(5)
	net.SetCascadeOutputStagnationEpochs(5)
	net.SetCascadeCandidateStagnationEpochs(5)
	
	// Track neuron additions
	neuronHistory := []int{net.GetTotalNeurons()}
	mseHistory := []float32{}
	
	net.SetCallback(func(ann *Fann[float32], epochs int, mse float32) bool {
		currentNeurons := ann.GetTotalNeurons()
		if currentNeurons > neuronHistory[len(neuronHistory)-1] {
			neuronHistory = append(neuronHistory, currentNeurons)
			mseHistory = append(mseHistory, mse)
			t.Logf("Added neuron, total: %d, MSE: %.4f", 
				currentNeurons, mse)
		}
		return true
	})
	
	// Train with cascade
	initialMSE := net.TestData(subset)
	net.CascadetrainOnData(subset, 5, 1, 0.1)
	finalMSE := net.TestData(subset)
	
	t.Logf("Initial MSE: %.4f, Final MSE: %.4f", initialMSE, finalMSE)
	t.Logf("Neuron growth: %v", neuronHistory)
	
	// Check that cascade improved performance
	if finalMSE >= initialMSE {
		t.Errorf("Cascade training did not improve MSE")
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
}