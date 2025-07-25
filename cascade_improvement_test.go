package gofann

import (
	"testing"
)

// TestCascadeActuallyImproves verifies cascade training makes real improvements
func TestCascadeActuallyImproves(t *testing.T) {
	// Simple problem that requires hidden neurons: XOR
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
	data := CreateTrainDataArray(inputs, outputs)
	
	// Start with minimal network
	net := CreateCascade[float32](2, 1)
	
	// Configure for testing
	net.SetCascadeMaxOutEpochs(30)
	net.SetCascadeMaxCandEpochs(30)
	net.SetCascadeOutputStagnationEpochs(10)
	net.SetCascadeCandidateStagnationEpochs(10)
	
	// Get initial performance
	initialMSE := net.TestData(data)
	initialBitFail := net.GetBitFail()
	t.Logf("Initial: MSE=%f, BitFail=%d", initialMSE, initialBitFail)
	
	// Track MSE after each neuron addition
	mseHistory := []float32{initialMSE}
	neuronHistory := []int{net.GetTotalNeurons()}
	
	net.SetCallback(func(ann *Fann[float32], epochs int, mse float32) bool {
		currentNeurons := ann.GetTotalNeurons()
		if currentNeurons > neuronHistory[len(neuronHistory)-1] {
			// New neuron was added
			neuronHistory = append(neuronHistory, currentNeurons)
			mseHistory = append(mseHistory, mse)
			t.Logf("Added neuron %d: MSE=%f", currentNeurons, mse)
			
			// Check if MSE is improving
			if len(mseHistory) > 2 {
				prevMSE := mseHistory[len(mseHistory)-2]
				if mse >= prevMSE {
					t.Logf("WARNING: MSE not improving after adding neuron! %f -> %f", prevMSE, mse)
				}
			}
		}
		return true
	})
	
	// Train with cascade
	net.CascadetrainOnData(data, 3, 1, 0.001)
	
	// Final performance
	finalMSE := net.TestData(data)
	finalBitFail := net.GetBitFail()
	t.Logf("Final: MSE=%f, BitFail=%d, Neurons=%d", finalMSE, finalBitFail, net.GetTotalNeurons())
	
	// Test actual outputs
	t.Log("Final outputs:")
	correctCount := 0
	for i, input := range inputs {
		out := net.Run(input)
		expected := outputs[i][0]
		correct := (out[0] > 0.5 && expected > 0.5) || (out[0] < 0.5 && expected < 0.5)
		if correct {
			correctCount++
		}
		t.Logf("  %v -> %.3f (expected %.0f) %s", input, out[0], expected, map[bool]string{true: "✓", false: "✗"}[correct])
	}
	
	// Verify improvement
	if finalMSE >= initialMSE {
		t.Errorf("Cascade training did not improve MSE: %f -> %f", initialMSE, finalMSE)
	}
	
	if correctCount < 2 {
		t.Errorf("Network learned less than half the patterns correctly: %d/4", correctCount)
	}
	
	// Check that we added neurons
	if net.GetTotalNeurons() <= 4 {
		t.Errorf("No hidden neurons were added during cascade training")
	}
}

// TestCascadeOutputWeightInit tests if output weights to new neurons are initialized properly
func TestCascadeOutputWeightInit(t *testing.T) {
	// Create a simple problem where output should be 0.8
	inputs := [][]float32{{0.5}}
	outputs := [][]float32{{0.8}}
	data := CreateTrainDataArray(inputs, outputs)
	
	net := CreateCascade[float32](1, 1)
	
	// Train output layer first
	net.cascadeTrainOutput(data, 0.001)
	outputAfterTraining := net.Run([]float32{0.5})[0]
	t.Logf("Output after training: %f (target 0.8)", outputAfterTraining)
	
	// For this test, we'll just run one iteration of cascade to see what happens
	// when a neuron is added
	mseBefore := net.TestData(data)
	neuronsBefore := net.GetTotalNeurons()
	
	// Run cascade for a short time to add one neuron
	net.SetCascadeMaxOutEpochs(20)
	net.SetCascadeOutputStagnationEpochs(5)
	net.CascadetrainOnData(data, 1, 1, 0.0001)
	
	if net.GetTotalNeurons() > neuronsBefore {
		
		// Check MSE after adding candidate
		mseAfter := net.TestData(data)
		outputAfter := net.Run([]float32{0.5})[0]
		
		t.Logf("Before candidate: MSE=%f, output=%f", mseBefore, outputAfterTraining)
		t.Logf("After candidate: MSE=%f, output=%f", mseAfter, outputAfter)
		
		// The MSE should not get much worse when adding a well-trained candidate
		if mseAfter > mseBefore * 2.0 {
			t.Errorf("MSE got much worse after adding candidate: %f -> %f", mseBefore, mseAfter)
		}
	}
}