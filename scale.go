package gofann

import "fmt"

// ScaleInput scales the inputs before feeding them to the network
type ScaleInput[T Numeric] struct {
	min T
	max T
}

// ScaleOutput scales the outputs from the network
type ScaleOutput[T Numeric] struct {
	min T
	max T
}

// ScaleParams holds the scaling parameters for the network
type ScaleParams[T Numeric] struct {
	inputScale  []ScaleInput[T]
	outputScale []ScaleOutput[T]
	enabled     bool
}

// SetInputScaling sets the input scaling parameters
func (ann *Fann[T]) SetInputScaling(inputMin, inputMax []T) error {
	if len(inputMin) != ann.numInput || len(inputMax) != ann.numInput {
		return fmt.Errorf("input scaling arrays must have length %d", ann.numInput)
	}
	
	if ann.scaleParams == nil {
		ann.scaleParams = &ScaleParams[T]{
			inputScale:  make([]ScaleInput[T], ann.numInput),
			outputScale: make([]ScaleOutput[T], ann.numOutput),
		}
	}
	
	for i := 0; i < ann.numInput; i++ {
		ann.scaleParams.inputScale[i].min = inputMin[i]
		ann.scaleParams.inputScale[i].max = inputMax[i]
	}
	
	ann.scaleParams.enabled = true
	return nil
}

// SetOutputScaling sets the output scaling parameters
func (ann *Fann[T]) SetOutputScaling(outputMin, outputMax []T) error {
	if len(outputMin) != ann.numOutput || len(outputMax) != ann.numOutput {
		return fmt.Errorf("output scaling arrays must have length %d", ann.numOutput)
	}
	
	if ann.scaleParams == nil {
		ann.scaleParams = &ScaleParams[T]{
			inputScale:  make([]ScaleInput[T], ann.numInput),
			outputScale: make([]ScaleOutput[T], ann.numOutput),
		}
	}
	
	for i := 0; i < ann.numOutput; i++ {
		ann.scaleParams.outputScale[i].min = outputMin[i]
		ann.scaleParams.outputScale[i].max = outputMax[i]
	}
	
	ann.scaleParams.enabled = true
	return nil
}

// ClearScaling clears all scaling parameters
func (ann *Fann[T]) ClearScaling() {
	ann.scaleParams = nil
}

// ScaleInput scales input data according to the network's input scaling parameters
func (ann *Fann[T]) ScaleInput(input []T) []T {
	if ann.scaleParams == nil || !ann.scaleParams.enabled {
		return input
	}
	
	scaled := make([]T, len(input))
	for i := 0; i < len(input) && i < ann.numInput; i++ {
		scale := &ann.scaleParams.inputScale[i]
		if scale.max != scale.min {
			// Scale to [-1, 1]
			scaled[i] = T(2.0) * (input[i] - scale.min) / (scale.max - scale.min) - T(1.0)
		} else {
			scaled[i] = input[i]
		}
	}
	return scaled
}

// ScaleOutput scales output data according to the network's output scaling parameters
func (ann *Fann[T]) ScaleOutput(output []T) []T {
	if ann.scaleParams == nil || !ann.scaleParams.enabled {
		return output
	}
	
	scaled := make([]T, len(output))
	for i := 0; i < len(output) && i < ann.numOutput; i++ {
		scale := &ann.scaleParams.outputScale[i]
		if scale.max != scale.min {
			// Scale from [-1, 1] to [min, max]
			scaled[i] = (output[i] + T(1.0)) * (scale.max - scale.min) / T(2.0) + scale.min
		} else {
			scaled[i] = output[i]
		}
	}
	return scaled
}

// DescaleInput reverses input scaling
func (ann *Fann[T]) DescaleInput(scaledInput []T) []T {
	if ann.scaleParams == nil || !ann.scaleParams.enabled {
		return scaledInput
	}
	
	descaled := make([]T, len(scaledInput))
	for i := 0; i < len(scaledInput) && i < ann.numInput; i++ {
		scale := &ann.scaleParams.inputScale[i]
		if scale.max != scale.min {
			// Descale from [-1, 1]
			descaled[i] = (scaledInput[i] + T(1.0)) * (scale.max - scale.min) / T(2.0) + scale.min
		} else {
			descaled[i] = scaledInput[i]
		}
	}
	return descaled
}

// DescaleOutput reverses output scaling
func (ann *Fann[T]) DescaleOutput(scaledOutput []T) []T {
	if ann.scaleParams == nil || !ann.scaleParams.enabled {
		return scaledOutput
	}
	
	descaled := make([]T, len(scaledOutput))
	for i := 0; i < len(scaledOutput) && i < ann.numOutput; i++ {
		scale := &ann.scaleParams.outputScale[i]
		if scale.max != scale.min {
			// Descale to [-1, 1]
			descaled[i] = T(2.0) * (scaledOutput[i] - scale.min) / (scale.max - scale.min) - T(1.0)
		} else {
			descaled[i] = scaledOutput[i]
		}
	}
	return descaled
}

// RunScaled runs the network with automatic input/output scaling
func (ann *Fann[T]) RunScaled(input []T) []T {
	// Scale input
	scaledInput := ann.ScaleInput(input)
	
	// Run network
	output := ann.Run(scaledInput)
	
	// Scale output
	return ann.ScaleOutput(output)
}

// TrainScaled trains the network with a single input/output pair with scaling
func (ann *Fann[T]) TrainScaled(input []T, desiredOutput []T) {
	// Scale input and desired output
	scaledInput := ann.ScaleInput(input)
	scaledOutput := ann.DescaleOutput(desiredOutput)
	
	// Train with scaled values
	ann.Train(scaledInput, scaledOutput)
}

// TestScaled tests the network on a single input/output pair with scaling
func (ann *Fann[T]) TestScaled(input []T, desiredOutput []T) []T {
	// Scale input and desired output
	scaledInput := ann.ScaleInput(input)
	scaledOutput := ann.DescaleOutput(desiredOutput)
	
	// Test with scaled values - returns network output
	output := ann.Test(scaledInput, scaledOutput)
	
	// Scale the output back to original scale
	return ann.ScaleOutput(output)
}

// SetScalingFromData sets scaling parameters based on training data
func (ann *Fann[T]) SetScalingFromData(data *TrainData[T]) error {
	if data.numData == 0 {
		return fmt.Errorf("cannot set scaling from empty data")
	}
	
	// Find min/max for inputs
	inputMin := make([]T, ann.numInput)
	inputMax := make([]T, ann.numInput)
	outputMin := make([]T, ann.numOutput)
	outputMax := make([]T, ann.numOutput)
	
	// Initialize with first sample
	copy(inputMin, data.inputs[0])
	copy(inputMax, data.inputs[0])
	copy(outputMin, data.outputs[0])
	copy(outputMax, data.outputs[0])
	
	// Find actual min/max
	for i := 1; i < data.numData; i++ {
		for j := 0; j < ann.numInput; j++ {
			if data.inputs[i][j] < inputMin[j] {
				inputMin[j] = data.inputs[i][j]
			}
			if data.inputs[i][j] > inputMax[j] {
				inputMax[j] = data.inputs[i][j]
			}
		}
		for j := 0; j < ann.numOutput; j++ {
			if data.outputs[i][j] < outputMin[j] {
				outputMin[j] = data.outputs[i][j]
			}
			if data.outputs[i][j] > outputMax[j] {
				outputMax[j] = data.outputs[i][j]
			}
		}
	}
	
	// Set scaling parameters
	if err := ann.SetInputScaling(inputMin, inputMax); err != nil {
		return err
	}
	return ann.SetOutputScaling(outputMin, outputMax)
}