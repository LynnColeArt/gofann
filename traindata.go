package gofann

import (
	"math"
	"math/rand"
	"runtime"
)

// TrainData represents training data for neural networks
type TrainData[T Numeric] struct {
	numData   int
	numInput  int
	numOutput int
	inputs    [][]T
	outputs   [][]T
}

// CreateTrainData creates empty training data
func CreateTrainData[T Numeric](numData, numInput, numOutput int) *TrainData[T] {
	td := &TrainData[T]{
		numData:   numData,
		numInput:  numInput,
		numOutput: numOutput,
		inputs:    make([][]T, numData),
		outputs:   make([][]T, numData),
	}
	
	for i := 0; i < numData; i++ {
		td.inputs[i] = make([]T, numInput)
		td.outputs[i] = make([]T, numOutput)
	}
	
	return td
}

// CreateTrainDataArray creates training data from input/output arrays
func CreateTrainDataArray[T Numeric](inputs, outputs [][]T) *TrainData[T] {
	if len(inputs) != len(outputs) {
		return nil
	}
	if len(inputs) == 0 {
		return nil
	}
	
	numData := len(inputs)
	numInput := len(inputs[0])
	numOutput := len(outputs[0])
	
	td := &TrainData[T]{
		numData:   numData,
		numInput:  numInput,
		numOutput: numOutput,
		inputs:    make([][]T, numData),
		outputs:   make([][]T, numData),
	}
	
	// Deep copy the data
	for i := 0; i < numData; i++ {
		td.inputs[i] = make([]T, numInput)
		copy(td.inputs[i], inputs[i])
		
		td.outputs[i] = make([]T, numOutput)
		copy(td.outputs[i], outputs[i])
	}
	
	return td
}

// GetNumData returns the number of training patterns
func (td *TrainData[T]) GetNumData() int {
	return td.numData
}

// GetNumInput returns the number of inputs
func (td *TrainData[T]) GetNumInput() int {
	return td.numInput
}

// GetNumOutput returns the number of outputs
func (td *TrainData[T]) GetNumOutput() int {
	return td.numOutput
}

// GetInput returns the input array for the given index
func (td *TrainData[T]) GetInput(index int) []T {
	if index < 0 || index >= td.numData {
		return nil
	}
	return td.inputs[index]
}

// GetOutput returns the output array for the given index
func (td *TrainData[T]) GetOutput(index int) []T {
	if index < 0 || index >= td.numData {
		return nil
	}
	return td.outputs[index]
}

// SetInput sets the input array for the given index
func (td *TrainData[T]) SetInput(index int, input []T) {
	if index >= 0 && index < td.numData && len(input) == td.numInput {
		copy(td.inputs[index], input)
	}
}

// SetOutput sets the output array for the given index
func (td *TrainData[T]) SetOutput(index int, output []T) {
	if index >= 0 && index < td.numData && len(output) == td.numOutput {
		copy(td.outputs[index], output)
	}
}

// Shuffle randomly shuffles the training data
func (td *TrainData[T]) Shuffle() {
	n := td.numData
	for i := n - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		td.inputs[i], td.inputs[j] = td.inputs[j], td.inputs[i]
		td.outputs[i], td.outputs[j] = td.outputs[j], td.outputs[i]
	}
}

// Duplicate creates a copy of the training data
func (td *TrainData[T]) Duplicate() *TrainData[T] {
	newTd := &TrainData[T]{
		numData:   td.numData,
		numInput:  td.numInput,
		numOutput: td.numOutput,
		inputs:    make([][]T, td.numData),
		outputs:   make([][]T, td.numData),
	}
	
	for i := 0; i < td.numData; i++ {
		newTd.inputs[i] = make([]T, td.numInput)
		copy(newTd.inputs[i], td.inputs[i])
		
		newTd.outputs[i] = make([]T, td.numOutput)
		copy(newTd.outputs[i], td.outputs[i])
	}
	
	return newTd
}

// Subset returns a subset of the training data
func (td *TrainData[T]) Subset(pos, length int) *TrainData[T] {
	if pos < 0 || pos >= td.numData || length <= 0 {
		return nil
	}
	
	if pos+length > td.numData {
		length = td.numData - pos
	}
	
	newTd := &TrainData[T]{
		numData:   length,
		numInput:  td.numInput,
		numOutput: td.numOutput,
		inputs:    make([][]T, length),
		outputs:   make([][]T, length),
	}
	
	for i := 0; i < length; i++ {
		newTd.inputs[i] = make([]T, td.numInput)
		copy(newTd.inputs[i], td.inputs[pos+i])
		
		newTd.outputs[i] = make([]T, td.numOutput)
		copy(newTd.outputs[i], td.outputs[pos+i])
	}
	
	return newTd
}

// Merge merges two training data sets
func (td *TrainData[T]) Merge(other *TrainData[T]) *TrainData[T] {
	if td.numInput != other.numInput || td.numOutput != other.numOutput {
		return nil
	}
	
	totalData := td.numData + other.numData
	newTd := &TrainData[T]{
		numData:   totalData,
		numInput:  td.numInput,
		numOutput: td.numOutput,
		inputs:    make([][]T, totalData),
		outputs:   make([][]T, totalData),
	}
	
	// Copy first dataset
	for i := 0; i < td.numData; i++ {
		newTd.inputs[i] = make([]T, td.numInput)
		copy(newTd.inputs[i], td.inputs[i])
		
		newTd.outputs[i] = make([]T, td.numOutput)
		copy(newTd.outputs[i], td.outputs[i])
	}
	
	// Copy second dataset
	for i := 0; i < other.numData; i++ {
		idx := td.numData + i
		newTd.inputs[idx] = make([]T, td.numInput)
		copy(newTd.inputs[idx], other.inputs[i])
		
		newTd.outputs[idx] = make([]T, td.numOutput)
		copy(newTd.outputs[idx], other.outputs[i])
	}
	
	return newTd
}

// GetMinInput returns the minimum input value
func (td *TrainData[T]) GetMinInput() T {
	if td.numData == 0 || td.numInput == 0 {
		return T(0)
	}
	
	min := td.inputs[0][0]
	for i := 0; i < td.numData; i++ {
		for j := 0; j < td.numInput; j++ {
			if td.inputs[i][j] < min {
				min = td.inputs[i][j]
			}
		}
	}
	return min
}

// GetMaxInput returns the maximum input value
func (td *TrainData[T]) GetMaxInput() T {
	if td.numData == 0 || td.numInput == 0 {
		return T(0)
	}
	
	max := td.inputs[0][0]
	for i := 0; i < td.numData; i++ {
		for j := 0; j < td.numInput; j++ {
			if td.inputs[i][j] > max {
				max = td.inputs[i][j]
			}
		}
	}
	return max
}

// GetMinOutput returns the minimum output value
func (td *TrainData[T]) GetMinOutput() T {
	if td.numData == 0 || td.numOutput == 0 {
		return T(0)
	}
	
	min := td.outputs[0][0]
	for i := 0; i < td.numData; i++ {
		for j := 0; j < td.numOutput; j++ {
			if td.outputs[i][j] < min {
				min = td.outputs[i][j]
			}
		}
	}
	return min
}

// GetMaxOutput returns the maximum output value
func (td *TrainData[T]) GetMaxOutput() T {
	if td.numData == 0 || td.numOutput == 0 {
		return T(0)
	}
	
	max := td.outputs[0][0]
	for i := 0; i < td.numData; i++ {
		for j := 0; j < td.numOutput; j++ {
			if td.outputs[i][j] > max {
				max = td.outputs[i][j]
			}
		}
	}
	return max
}

// ScaleInput scales input data to the specified range
func (td *TrainData[T]) ScaleInput(newMin, newMax T) {
	oldMin := td.GetMinInput()
	oldMax := td.GetMaxInput()
	
	// Handle case where all values are the same
	if oldMax == oldMin {
		// Set all values to the middle of the new range
		midValue := (newMin + newMax) / T(2)
		for i := 0; i < td.numData; i++ {
			for j := 0; j < td.numInput; j++ {
				td.inputs[i][j] = midValue
			}
		}
		return
	}
	
	scale := (newMax - newMin) / (oldMax - oldMin)
	
	for i := 0; i < td.numData; i++ {
		for j := 0; j < td.numInput; j++ {
			td.inputs[i][j] = newMin + (td.inputs[i][j]-oldMin)*scale
		}
	}
}

// ScaleOutput scales output data to the specified range
func (td *TrainData[T]) ScaleOutput(newMin, newMax T) {
	oldMin := td.GetMinOutput()
	oldMax := td.GetMaxOutput()
	
	// Handle case where all values are the same
	if oldMax == oldMin {
		// Set all values to the middle of the new range
		midValue := (newMin + newMax) / T(2)
		for i := 0; i < td.numData; i++ {
			for j := 0; j < td.numOutput; j++ {
				td.outputs[i][j] = midValue
			}
		}
		return
	}
	
	scale := (newMax - newMin) / (oldMax - oldMin)
	
	for i := 0; i < td.numData; i++ {
		for j := 0; j < td.numOutput; j++ {
			td.outputs[i][j] = newMin + (td.outputs[i][j]-oldMin)*scale
		}
	}
}

// Scale scales both input and output data to the specified range
func (td *TrainData[T]) Scale(newMin, newMax T) {
	td.ScaleInput(newMin, newMax)
	td.ScaleOutput(newMin, newMax)
}

// TrainOnData trains the network on the given data
func (ann *Fann[T]) TrainOnData(data *TrainData[T], maxEpochs, epochsBetweenReports int, desiredError float32) {
	if data.numInput != ann.numInput {
		ann.setError(ErrTrainDataMismatch, "training data input size mismatch")
		return
	}
	if data.numOutput != ann.numOutput {
		ann.setError(ErrTrainDataMismatch, "training data output size mismatch")
		return
	}
	
	for epoch := 1; epoch <= maxEpochs; epoch++ {
		mse := ann.TrainEpoch(data)
		
		// Yield CPU time periodically to prevent system freeze
		if epoch%100 == 0 {
			runtime.Gosched()
		}
		
		// Report progress
		if epochsBetweenReports > 0 && epoch%epochsBetweenReports == 0 {
			if ann.callback != nil {
				if !ann.callback(ann, epoch, mse) {
					return // Callback requested stop
				}
			}
		}
		
		// Check if desired error reached
		if mse <= desiredError {
			return
		}
	}
}

// Test tests the network with one input-output pair and returns the output
func (ann *Fann[T]) Test(input, desiredOutput []T) []T {
	output := ann.Run(input)
	if output != nil && desiredOutput != nil {
		ann.calculateMSE(desiredOutput)
	}
	return output
}

// TestData tests the network on all the data and returns MSE
func (ann *Fann[T]) TestData(data *TrainData[T]) float32 {
	ann.ResetMSE()
	
	for i := 0; i < data.numData; i++ {
		ann.Test(data.inputs[i], data.outputs[i])
	}
	
	return ann.GetMSE()
}

// SetCallback sets the training callback function
func (ann *Fann[T]) SetCallback(callback func(ann *Fann[T], epochs int, mse float32) bool) {
	ann.callback = callback
}

// InitWeights initializes weights using Widrow-Nguyen algorithm
func (ann *Fann[T]) InitWeights(data *TrainData[T]) {
	// Find min and max for inputs and outputs
	minInput := data.GetMinInput()
	maxInput := data.GetMaxInput()
	minOutput := data.GetMinOutput()
	maxOutput := data.GetMaxOutput()
	
	// Calculate scale factor
	inputRange := float64(maxInput - minInput)
	outputRange := float64(maxOutput - minOutput)
	
	// Calculate weight range based on network topology
	neuronsExceptInput := ann.totalNeurons - ann.numInput
	if neuronsExceptInput == 0 {
		neuronsExceptInput = 1 // Prevent division by zero
	}
	numConnPerNeuron := float64(ann.totalConnections) / float64(neuronsExceptInput)
	
	// Prevent division by zero in power calculation
	inputDivisor := float64(ann.numInput)
	if inputDivisor == 0 {
		inputDivisor = 1
	}
	weightRange := math.Pow(0.7*float64(len(ann.layers)), 1.0/inputDivisor) / 
		math.Sqrt(numConnPerNeuron)
	
	if inputRange > 0 {
		weightRange *= outputRange / inputRange
	}
	
	// Randomize weights with calculated range
	ann.RandomizeWeights(T(-weightRange), T(weightRange))
}