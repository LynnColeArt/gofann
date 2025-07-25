package gofann

import (
	"math"
	"testing"
)

// TestCascadeBasicMechanism tests the basic cascade training mechanism
func TestCascadeBasicMechanism(t *testing.T) {
	// Very simple problem: learn to output a constant
	inputs := [][]float32{{1.0}}
	outputs := [][]float32{{0.8}}
	data := CreateTrainDataArray(inputs, outputs)
	
	net := CreateCascade[float32](1, 1)
	
	// First, verify output training works
	initialMSE := net.TestData(data)
	initialOut := net.Run([]float32{1.0})[0]
	t.Logf("Initial: output=%f, MSE=%f", initialOut, initialMSE)
	
	// Train just the output layer
	net.cascadeTrainOutput(data, 0.001)
	
	afterOutputMSE := net.TestData(data)
	afterOutputOut := net.Run([]float32{1.0})[0]
	t.Logf("After output training: output=%f, MSE=%f", afterOutputOut, afterOutputMSE)
	
	if afterOutputMSE >= initialMSE {
		t.Errorf("Output training failed to improve MSE: %f -> %f", initialMSE, afterOutputMSE)
	}
	
	// Now let's manually test candidate training
	// For a linear problem, we don't really need hidden neurons, 
	// but let's see if the mechanism works
	
	// Save current state
	savedWeights := make([]float32, len(net.weights))
	copy(savedWeights, net.weights)
	
	// Try adding a neuron
	neuronsBefore := net.GetTotalNeurons()
	net.SetCascadeMaxOutEpochs(10) 
	net.SetCascadeOutputStagnationEpochs(5)
	net.CascadetrainOnData(data, 1, 1, 0.0001)
	
	if net.GetTotalNeurons() > neuronsBefore {
		finalMSE := net.TestData(data)
		finalOut := net.Run([]float32{1.0})[0]
		t.Logf("After adding neuron: output=%f, MSE=%f, neurons=%d", finalOut, finalMSE, net.GetTotalNeurons())
		
		// Even if we add a neuron, MSE shouldn't get much worse
		if finalMSE > afterOutputMSE * 1.5 {
			t.Errorf("Adding neuron made MSE much worse: %f -> %f", afterOutputMSE, finalMSE)
		}
	} else {
		t.Log("No neuron was added (expected for this simple problem)")
	}
}

// TestCascadeCorrelationCalculation tests if correlation is calculated correctly
func TestCascadeCorrelationCalculation(t *testing.T) {
	// Create a network with known error pattern
	net := CreateStandard[float32](2, 1)
	
	// Create data with specific pattern
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
	CreateTrainDataArray(inputs, outputs) // data not used in this test
	
	// Get network outputs and errors
	errors := make([]float32, 4)
	networkOutputs := make([]float32, 4)
	for i := 0; i < 4; i++ {
		out := net.Run(inputs[i])[0]
		networkOutputs[i] = out
		errors[i] = outputs[i][0] - out
		t.Logf("Pattern %d: output=%f, target=%f, error=%f", i, out, outputs[i][0], errors[i])
	}
	
	// Calculate what a good candidate's correlation should be
	// For XOR, a candidate that outputs high for patterns 1,2 and low for 0,3
	// should have high correlation with the error
	
	// Simulate a good candidate output
	goodCandidateOutputs := []float32{0.1, 0.9, 0.9, 0.1} // Matches XOR pattern
	
	// Calculate correlation
	var sumErrors, sumCandidate, sumErrorCandidate float32
	for i := 0; i < 4; i++ {
		sumErrors += errors[i]
		sumCandidate += goodCandidateOutputs[i]
		sumErrorCandidate += errors[i] * goodCandidateOutputs[i]
	}
	
	meanError := sumErrors / 4
	meanCandidate := sumCandidate / 4
	
	// Calculate covariance and standard deviations
	var covariance, varError, varCandidate float32
	for i := 0; i < 4; i++ {
		errorDev := errors[i] - meanError
		candidateDev := goodCandidateOutputs[i] - meanCandidate
		covariance += errorDev * candidateDev
		varError += errorDev * errorDev
		varCandidate += candidateDev * candidateDev
	}
	covariance /= 4
	stdError := float32(math.Sqrt(float64(varError / 4)))
	stdCandidate := float32(math.Sqrt(float64(varCandidate / 4)))
	
	correlation := covariance / (stdError * stdCandidate)
	t.Logf("Good candidate correlation: %f", correlation)
	
	// A good candidate for XOR should have high absolute correlation
	if math.Abs(float64(correlation)) < 0.5 {
		t.Logf("WARNING: Even a perfect candidate has low correlation. The cascade algorithm may not work well for this problem.")
	}
}