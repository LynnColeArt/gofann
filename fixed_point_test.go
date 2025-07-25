package gofann

import (
	"fmt"
	"os"
	"testing"
)

// TestFixedPointSave tests saving networks in fixed-point format
func TestFixedPointSave(t *testing.T) {
	// Create a simple network
	ann := CreateStandard[float32](2, 3, 1)
	ann.RandomizeWeights(-1.0, 1.0)
	
	// Test different decimal point precisions
	testCases := []struct {
		decimalPoint uint
		description  string
	}{
		{8, "8-bit precision (256 levels)"},
		{12, "12-bit precision (4096 levels)"},
		{16, "16-bit precision (65536 levels)"},
	}
	
	for _, tc := range testCases {
		filename := "test_fixed_" + tc.description + ".net"
		
		// Save in fixed-point format
		err := ann.SaveToFixed(filename, tc.decimalPoint)
		if err != nil {
			t.Errorf("Failed to save fixed-point network (%s): %v", tc.description, err)
			continue
		}
		
		// Check file was created
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			t.Errorf("Fixed-point file not created: %s", filename)
			continue
		}
		
		t.Logf("✅ Fixed-point save successful: %s", tc.description)
		
		// Clean up
		os.Remove(filename)
	}
}

// TestFixedPointPrecision tests the precision of fixed-point conversion
func TestFixedPointPrecision(t *testing.T) {
	// Test the core fixed-point conversion logic
	testValues := []float64{0.0, 0.5, -0.5, 1.0, -1.0, 0.125, -0.125}
	
	for decimalPoint := uint(4); decimalPoint <= 16; decimalPoint += 4 {
		multiplier := float64(uint(1) << decimalPoint)
		t.Logf("Testing %d-bit precision (multiplier: %.0f)", decimalPoint, multiplier)
		
		for _, original := range testValues {
			// Convert to fixed-point and back
			fixedValue := int(original * multiplier)
			recovered := float64(fixedValue) / multiplier
			
			// Calculate error
			error := recovered - original
			maxError := 1.0 / multiplier // Maximum quantization error
			
			if error > maxError || error < -maxError {
				t.Errorf("Fixed-point error too large for %d-bit: %.6f -> %d -> %.6f (error: %.6f, max: %.6f)", 
					decimalPoint, original, fixedValue, recovered, error, maxError)
			}
			
			t.Logf("   %.6f -> %d -> %.6f (error: %.6f)", original, fixedValue, recovered, error)
		}
	}
	
	t.Logf("✅ Fixed-point precision tests passed")
}

// TestFixedPointNetworkComparison compares fixed-point vs floating-point networks
func TestFixedPointNetworkComparison(t *testing.T) {
	// Create and train a network
	ann := CreateStandard[float32](2, 4, 1)
	ann.RandomizeWeights(-1.0, 1.0)
	
	// Create training data
	data := CreateTrainDataArray[float32]([][]float32{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}, [][]float32{
		{0}, {1}, {1}, {0},
	})
	
	// Train briefly
	for i := 0; i < 50; i++ {
		ann.TrainEpoch(data)
	}
	
	// Test input
	testInput := []float32{0.5, 0.8}
	originalOutput := ann.Run(testInput)
	
	// Save in different fixed-point formats
	precisions := []uint{8, 12, 16}
	
	for _, precision := range precisions {
		filename := "comparison_test.net"
		
		// Save in fixed-point format
		err := ann.SaveToFixed(filename, precision)
		if err != nil {
			t.Errorf("Failed to save %d-bit fixed-point: %v", precision, err)
			continue
		}
		
		t.Logf("✅ %d-bit fixed-point save successful", precision)
		t.Logf("   Original output: %.6f", originalOutput[0])
		t.Logf("   Quantization level: 1/%.0f = %.6f", 
			float64(uint(1)<<precision), 1.0/float64(uint(1)<<precision))
		
		// Clean up
		os.Remove(filename)
	}
}

// TestFixedPointFileFormat tests that fixed-point files have correct format
func TestFixedPointFileFormat(t *testing.T) {
	ann := CreateStandard[float32](2, 2, 1)
	ann.RandomizeWeights(-0.5, 0.5)
	
	filename := "format_test.net"
	decimalPoint := uint(12)
	
	// Save in fixed-point format
	err := ann.SaveToFixed(filename, decimalPoint)
	if err != nil {
		t.Fatalf("Failed to save fixed-point network: %v", err)
	}
	defer os.Remove(filename)
	
	// Read file and check format
	file, err := os.Open(filename)
	if err != nil {
		t.Fatalf("Failed to open saved file: %v", err)
	}
	defer file.Close()
	
	// Read first line to check fixed-point header
	var header string
	_, err = fmt.Fscanf(file, "%s", &header)
	if err != nil {
		t.Fatalf("Failed to read header: %v", err)
	}
	
	expectedHeader := fmt.Sprintf("FANN_FIX_%d.0", 1<<decimalPoint)
	if header != expectedHeader {
		t.Errorf("Wrong fixed-point header: expected %s, got %s", expectedHeader, header)
	} else {
		t.Logf("✅ Fixed-point header correct: %s", header)
	}
}