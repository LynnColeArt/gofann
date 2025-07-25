package gofann

import (
	"fmt"
	"os"
	"testing"
)

func TestSaveFormat(t *testing.T) {
	// Create a minimal network
	net := CreateStandard[float32](2, 2, 1)
	
	// Save to current directory for inspection
	filename := "test_network.net"
	err := net.Save(filename)
	if err != nil {
		t.Fatalf("Failed to save: %v", err)
	}
	defer os.Remove(filename)
	
	// Read and print file content
	content, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("Failed to read file: %v", err)
	}
	
	fmt.Println("Saved file content:")
	fmt.Println(string(content))
}