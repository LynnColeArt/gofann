package gofann

import (
	"fmt"
	"io"
	"os"
)

// PrintConnections prints the connections of the network to stdout
func (ann *Fann[T]) PrintConnections() {
	ann.FprintConnections(os.Stdout)
}

// FprintConnections prints the connections of the network to the given writer
func (ann *Fann[T]) FprintConnections(w io.Writer) {
	fmt.Fprintf(w, "Network connections:\n")
	fmt.Fprintf(w, "Total neurons: %d\n", ann.totalNeurons)
	fmt.Fprintf(w, "Total connections: %d\n", ann.totalConnections)
	fmt.Fprintf(w, "\n")
	
	// Print layer information
	for i, layer := range ann.layers {
		layerSize := layer.lastNeuron - layer.firstNeuron
		if i < len(ann.layers)-1 {
			layerSize-- // Exclude bias neuron
		}
		fmt.Fprintf(w, "Layer %d: %d neurons (indices %d-%d)\n", 
			i, layerSize, layer.firstNeuron, layer.lastNeuron-1)
	}
	fmt.Fprintf(w, "\n")
	
	// Print connections for each neuron
	for i := 0; i < ann.totalNeurons; i++ {
		neuron := &ann.neurons[i]
		numConnections := neuron.lastCon - neuron.firstCon
		
		if numConnections > 0 {
			fmt.Fprintf(w, "Neuron %d (", i)
			
			// Determine layer
			layerIdx := -1
			for l, layer := range ann.layers {
				if i >= layer.firstNeuron && i < layer.lastNeuron {
					layerIdx = l
					break
				}
			}
			
			if layerIdx >= 0 {
				fmt.Fprintf(w, "layer %d", layerIdx)
				if layerIdx < len(ann.layers)-1 && i == ann.layers[layerIdx].lastNeuron-1 {
					fmt.Fprintf(w, ", bias")
				}
			}
			
			fmt.Fprintf(w, ") has %d connections:\n", numConnections)
			
			for j := neuron.firstCon; j < neuron.lastCon; j++ {
				srcNeuron := ann.connections[j]
				weight := ann.weights[j]
				fmt.Fprintf(w, "  <- Neuron %d (weight: %f)\n", srcNeuron, weight)
			}
		}
	}
}

// PrintParameters prints the parameters of the network to stdout
func (ann *Fann[T]) PrintParameters() {
	ann.FprintParameters(os.Stdout)
}

// FprintParameters prints the parameters of the network to the given writer
func (ann *Fann[T]) FprintParameters(w io.Writer) {
	fmt.Fprintf(w, "Network parameters:\n")
	fmt.Fprintf(w, "===================\n")
	
	// Structure
	fmt.Fprintf(w, "\nStructure:\n")
	fmt.Fprintf(w, "  Network type: %s\n", getNetworkTypeName(ann.networkType))
	fmt.Fprintf(w, "  Input neurons: %d\n", ann.numInput)
	fmt.Fprintf(w, "  Output neurons: %d\n", ann.numOutput)
	fmt.Fprintf(w, "  Total neurons: %d\n", ann.totalNeurons)
	fmt.Fprintf(w, "  Total connections: %d\n", ann.totalConnections)
	fmt.Fprintf(w, "  Connection rate: %.2f\n", ann.connectionRate)
	
	// Training parameters
	fmt.Fprintf(w, "\nTraining parameters:\n")
	fmt.Fprintf(w, "  Training algorithm: %s\n", getTrainingAlgorithmName(ann.trainingAlgorithm))
	fmt.Fprintf(w, "  Learning rate: %f\n", ann.learningRate)
	fmt.Fprintf(w, "  Learning momentum: %f\n", ann.learningMomentum)
	fmt.Fprintf(w, "  Error function: %s\n", getErrorFunctionName(ann.trainErrorFunction))
	fmt.Fprintf(w, "  Bit fail limit: %f\n", ann.bitFailLimit)
	
	// Algorithm-specific parameters
	switch ann.trainingAlgorithm {
	case TrainRPROP:
		fmt.Fprintf(w, "\nRPROP parameters:\n")
		fmt.Fprintf(w, "  Increase factor: %f\n", ann.rpropIncreaseFactor)
		fmt.Fprintf(w, "  Decrease factor: %f\n", ann.rpropDecreaseFactor)
		fmt.Fprintf(w, "  Delta min: %f\n", ann.rpropDeltaMin)
		fmt.Fprintf(w, "  Delta max: %f\n", ann.rpropDeltaMax)
		fmt.Fprintf(w, "  Delta zero: %f\n", ann.rpropDeltaZero)
		
	case TrainQuickprop:
		fmt.Fprintf(w, "\nQuickprop parameters:\n")
		fmt.Fprintf(w, "  Decay: %f\n", ann.quickpropDecay)
		fmt.Fprintf(w, "  Mu: %f\n", ann.quickpropMu)
		
	case TrainSarprop:
		fmt.Fprintf(w, "\nSARPROP parameters:\n")
		fmt.Fprintf(w, "  Weight decay shift: %f\n", ann.sarpropWeightDecayShift)
		fmt.Fprintf(w, "  Step error threshold factor: %f\n", ann.sarpropStepErrorThresholdFactor)
		fmt.Fprintf(w, "  Step error shift: %f\n", ann.sarpropStepErrorShift)
		fmt.Fprintf(w, "  Temperature: %f\n", ann.sarpropTemperature)
	}
	
	// Cascade parameters
	fmt.Fprintf(w, "\nCascade parameters:\n")
	fmt.Fprintf(w, "  Output change fraction: %f\n", ann.cascadeOutputChangeFraction)
	fmt.Fprintf(w, "  Output stagnation epochs: %d\n", ann.cascadeOutputStagnationEpochs)
	fmt.Fprintf(w, "  Candidate change fraction: %f\n", ann.cascadeCandidateChangeFraction)
	fmt.Fprintf(w, "  Candidate stagnation epochs: %d\n", ann.cascadeCandidateStagnationEpochs)
	fmt.Fprintf(w, "  Max output epochs: %d\n", ann.cascadeMaxOutEpochs)
	fmt.Fprintf(w, "  Min output epochs: %d\n", ann.cascadeMinOutEpochs)
	fmt.Fprintf(w, "  Max candidate epochs: %d\n", ann.cascadeMaxCandEpochs)
	fmt.Fprintf(w, "  Min candidate epochs: %d\n", ann.cascadeMinCandEpochs)
	fmt.Fprintf(w, "  Candidate groups: %d\n", ann.cascadeNumCandidateGroups)
	fmt.Fprintf(w, "  Candidate limit: %d\n", ann.cascadeCandidateLimit)
	fmt.Fprintf(w, "  Weight multiplier: %f\n", ann.cascadeWeightMultiplier)
	
	// Current training state
	fmt.Fprintf(w, "\nCurrent state:\n")
	fmt.Fprintf(w, "  MSE: %f\n", ann.mse)
	fmt.Fprintf(w, "  Bit fail: %d\n", ann.bitFail)
}

func getNetworkTypeName(nt NetworkType) string {
	switch nt {
	case NetTypeLayer:
		return "LAYER"
	case NetTypeShortcut:
		return "SHORTCUT"
	default:
		return "UNKNOWN"
	}
}

func getTrainingAlgorithmName(ta TrainAlgorithm) string {
	switch ta {
	case TrainIncremental:
		return "INCREMENTAL"
	case TrainBatch:
		return "BATCH"
	case TrainRPROP:
		return "RPROP"
	case TrainQuickprop:
		return "QUICKPROP"
	case TrainSarprop:
		return "SARPROP"
	default:
		return "UNKNOWN"
	}
}

func getErrorFunctionName(ef ErrorFunc) string {
	switch ef {
	case ErrorLinear:
		return "LINEAR"
	case ErrorTanh:
		return "TANH"
	default:
		return "UNKNOWN"
	}
}