package gofann

import (
	"path/filepath"
	"testing"
)

func TestReadFANNTrainData(t *testing.T) {
	// Test loading the XOR dataset from FANN
	xorPath := "/media/lynn/big_drive/workspaces/fanmaker/fann_source/examples/xor.data"
	
	data, err := ReadTrainFromFile[float32](xorPath)
	if err != nil {
		t.Fatalf("Failed to load XOR data: %v", err)
	}
	
	// Verify data
	if data.GetNumData() != 4 {
		t.Errorf("Expected 4 samples, got %d", data.GetNumData())
	}
	if data.GetNumInput() != 2 {
		t.Errorf("Expected 2 inputs, got %d", data.GetNumInput())
	}
	if data.GetNumOutput() != 1 {
		t.Errorf("Expected 1 output, got %d", data.GetNumOutput())
	}
	
	// Check the actual XOR data
	expectedInputs := [][]float32{
		{-1, -1},
		{-1, 1},
		{1, -1},
		{1, 1},
	}
	expectedOutputs := [][]float32{
		{-1},
		{1},
		{1},
		{-1},
	}
	
	for i := 0; i < 4; i++ {
		input := data.GetInput(i)
		output := data.GetOutput(i)
		
		for j := 0; j < 2; j++ {
			if input[j] != expectedInputs[i][j] {
				t.Errorf("Input[%d][%d]: got %f, expected %f", 
					i, j, input[j], expectedInputs[i][j])
			}
		}
		
		if output[0] != expectedOutputs[i][0] {
			t.Errorf("Output[%d]: got %f, expected %f", 
				i, output[0], expectedOutputs[i][0])
		}
	}
	
	t.Log("Successfully loaded FANN XOR dataset!")
}

func TestSaveLoadTrainData(t *testing.T) {
	// Create test data
	inputs := [][]float32{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
		{0.7, 0.8, 0.9},
	}
	outputs := [][]float32{
		{0.1, 0.2},
		{0.3, 0.4},
		{0.5, 0.6},
	}
	
	td1 := CreateTrainDataArray(inputs, outputs)
	
	// Save to file
	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "test_data.train")
	
	err := td1.Save(filename)
	if err != nil {
		t.Fatalf("Failed to save training data: %v", err)
	}
	
	// Load from file
	td2, err := ReadTrainFromFile[float32](filename)
	if err != nil {
		t.Fatalf("Failed to load training data: %v", err)
	}
	
	// Compare
	if td1.GetNumData() != td2.GetNumData() {
		t.Errorf("NumData mismatch: %d vs %d", td1.GetNumData(), td2.GetNumData())
	}
	if td1.GetNumInput() != td2.GetNumInput() {
		t.Errorf("NumInput mismatch: %d vs %d", td1.GetNumInput(), td2.GetNumInput())
	}
	if td1.GetNumOutput() != td2.GetNumOutput() {
		t.Errorf("NumOutput mismatch: %d vs %d", td1.GetNumOutput(), td2.GetNumOutput())
	}
	
	// Compare data values
	for i := 0; i < td1.GetNumData(); i++ {
		input1 := td1.GetInput(i)
		input2 := td2.GetInput(i)
		
		for j := 0; j < td1.GetNumInput(); j++ {
			if abs(input1[j]-input2[j]) > 0.0001 {
				t.Errorf("Input[%d][%d] mismatch: %f vs %f", i, j, input1[j], input2[j])
			}
		}
		
		output1 := td1.GetOutput(i)
		output2 := td2.GetOutput(i)
		
		for j := 0; j < td1.GetNumOutput(); j++ {
			if abs(output1[j]-output2[j]) > 0.0001 {
				t.Errorf("Output[%d][%d] mismatch: %f vs %f", i, j, output1[j], output2[j])
			}
		}
	}
}

func TestTrainOnFANNDataset(t *testing.T) {
	// Create a network
	net := CreateStandard[float32](2, 3, 1)
	net.SetTrainingAlgorithm(TrainIncremental)
	net.SetLearningRate(0.7)
	
	// Train on XOR dataset from FANN
	xorPath := "/media/lynn/big_drive/workspaces/fanmaker/fann_source/examples/xor.data"
	
	// Convert -1/1 to 0/1 for sigmoid activation
	data, err := ReadTrainFromFile[float32](xorPath)
	if err != nil {
		t.Fatalf("Failed to load data: %v", err)
	}
	
	// Scale from [-1,1] to [0,1]
	data.ScaleInput(0, 1)
	data.ScaleOutput(0, 1)
	
	// Train
	initialMSE := net.TestData(data)
	t.Logf("Initial MSE: %f", initialMSE)
	
	net.TrainOnData(data, 5000, 1000, 0.01)
	
	finalMSE := net.TestData(data)
	t.Logf("Final MSE: %f", finalMSE)
	
	// Test the network
	for i := 0; i < data.GetNumData(); i++ {
		input := data.GetInput(i)
		output := net.Run(input)
		expected := data.GetOutput(i)
		t.Logf("Input: %v => Output: %.3f (expected: %.3f)", input, output[0], expected[0])
	}
	
	if finalMSE > 0.1 {
		t.Errorf("Network did not learn XOR properly, MSE: %f", finalMSE)
	}
}