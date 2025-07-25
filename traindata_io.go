package gofann

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Save saves the training data to a file in FANN format
func (td *TrainData[T]) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("can't create training data file: %w", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// Write header
	fmt.Fprintf(writer, "%d %d %d\n", td.numData, td.numInput, td.numOutput)

	// Write data
	for i := 0; i < td.numData; i++ {
		// Write inputs
		for j := 0; j < td.numInput; j++ {
			fmt.Fprintf(writer, "%g ", float64(td.inputs[i][j]))
		}
		fmt.Fprintln(writer)

		// Write outputs
		for j := 0; j < td.numOutput; j++ {
			fmt.Fprintf(writer, "%g ", float64(td.outputs[i][j]))
		}
		fmt.Fprintln(writer)
	}

	return nil
}

// SaveToFixed saves the training data in fixed-point format
func (td *TrainData[T]) SaveToFixed(filename string, decimalPoint uint) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("can't create training data file: %w", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// Write header
	fmt.Fprintf(writer, "%d %d %d\n", td.numData, td.numInput, td.numOutput)

	// Calculate multiplier for fixed point
	multiplier := float64(uint(1) << decimalPoint)

	// Write data
	for i := 0; i < td.numData; i++ {
		// Write inputs as fixed point
		for j := 0; j < td.numInput; j++ {
			fixedValue := int(float64(td.inputs[i][j]) * multiplier)
			fmt.Fprintf(writer, "%d ", fixedValue)
		}
		fmt.Fprintln(writer)

		// Write outputs as fixed point
		for j := 0; j < td.numOutput; j++ {
			fixedValue := int(float64(td.outputs[i][j]) * multiplier)
			fmt.Fprintf(writer, "%d ", fixedValue)
		}
		fmt.Fprintln(writer)
	}

	return nil
}

// ReadTrainFromFile reads training data from a FANN format file
func ReadTrainFromFile[T Numeric](filename string) (*TrainData[T], error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("can't open training data file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	// Read header
	if !scanner.Scan() {
		return nil, fmt.Errorf("can't read training data header")
	}

	header := strings.Fields(scanner.Text())
	if len(header) != 3 {
		return nil, fmt.Errorf("invalid training data header")
	}

	numData, err := strconv.Atoi(header[0])
	if err != nil {
		return nil, fmt.Errorf("invalid number of data samples: %w", err)
	}

	numInput, err := strconv.Atoi(header[1])
	if err != nil {
		return nil, fmt.Errorf("invalid number of inputs: %w", err)
	}

	numOutput, err := strconv.Atoi(header[2])
	if err != nil {
		return nil, fmt.Errorf("invalid number of outputs: %w", err)
	}

	// Create training data
	td := CreateTrainData[T](numData, numInput, numOutput)

	// Read data
	for i := 0; i < numData; i++ {
		// Read inputs
		if !scanner.Scan() {
			return nil, fmt.Errorf("unexpected end of file reading inputs for sample %d", i)
		}

		inputs := strings.Fields(scanner.Text())
		if len(inputs) != numInput {
			return nil, fmt.Errorf("wrong number of inputs for sample %d: got %d, expected %d", 
				i, len(inputs), numInput)
		}

		for j := 0; j < numInput; j++ {
			val, err := strconv.ParseFloat(inputs[j], 64)
			if err != nil {
				return nil, fmt.Errorf("invalid input value at sample %d, input %d: %w", i, j, err)
			}
			td.inputs[i][j] = T(val)
		}

		// Read outputs
		if !scanner.Scan() {
			return nil, fmt.Errorf("unexpected end of file reading outputs for sample %d", i)
		}

		outputs := strings.Fields(scanner.Text())
		if len(outputs) != numOutput {
			return nil, fmt.Errorf("wrong number of outputs for sample %d: got %d, expected %d", 
				i, len(outputs), numOutput)
		}

		for j := 0; j < numOutput; j++ {
			val, err := strconv.ParseFloat(outputs[j], 64)
			if err != nil {
				return nil, fmt.Errorf("invalid output value at sample %d, output %d: %w", i, j, err)
			}
			td.outputs[i][j] = T(val)
		}
	}

	return td, nil
}


// TestOnFile tests the network on data from a file
func (ann *Fann[T]) TestOnFile(filename string) (float32, error) {
	data, err := ReadTrainFromFile[T](filename)
	if err != nil {
		ann.setError(ErrCantOpenTD, err.Error())
		return 0, err
	}

	return ann.TestData(data), nil
}