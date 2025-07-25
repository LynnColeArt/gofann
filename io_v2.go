package gofann

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// SaveV2 saves the network in a simplified format that's easier to debug
func (ann *Fann[T]) SaveV2(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	w := bufio.NewWriter(file)
	defer w.Flush()

	// Write header
	fmt.Fprintln(w, "GOFANN_V1")
	
	// Write network structure
	fmt.Fprintf(w, "layers=%d\n", len(ann.layers))
	for i, layer := range ann.layers {
		size := layer.lastNeuron - layer.firstNeuron
		if i < len(ann.layers)-1 {
			size-- // Don't count bias
		}
		fmt.Fprintf(w, "layer_%d_size=%d\n", i, size)
	}
	
	// Write parameters
	fmt.Fprintf(w, "num_input=%d\n", ann.numInput)
	fmt.Fprintf(w, "num_output=%d\n", ann.numOutput)
	fmt.Fprintf(w, "total_neurons=%d\n", ann.totalNeurons)
	fmt.Fprintf(w, "total_connections=%d\n", ann.totalConnections)
	fmt.Fprintf(w, "learning_rate=%f\n", ann.learningRate)
	fmt.Fprintf(w, "learning_momentum=%f\n", ann.learningMomentum)
	fmt.Fprintf(w, "training_algorithm=%d\n", ann.trainingAlgorithm)
	fmt.Fprintf(w, "network_type=%d\n", ann.networkType)
	fmt.Fprintf(w, "connection_rate=%f\n", ann.connectionRate)
	
	// Write neurons
	fmt.Fprintln(w, "NEURONS")
	for i, neuron := range ann.neurons {
		fmt.Fprintf(w, "%d %d %d %d %f\n", 
			i, neuron.firstCon, neuron.lastCon, 
			neuron.activationFunction, float32(neuron.activationSteepness))
	}
	
	// Write connections
	fmt.Fprintln(w, "CONNECTIONS")
	for i, conn := range ann.connections {
		fmt.Fprintf(w, "%d %d\n", i, conn)
	}
	
	// Write weights
	fmt.Fprintln(w, "WEIGHTS")
	for i, weight := range ann.weights {
		fmt.Fprintf(w, "%d %f\n", i, float32(weight))
	}
	
	return nil
}

// LoadV2 loads a network from simplified format
func LoadV2[T Numeric](filename string) (*Fann[T], error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	
	// Read header
	if !scanner.Scan() || scanner.Text() != "GOFANN_V1" {
		return nil, fmt.Errorf("invalid file format")
	}
	
	// Read parameters
	params := make(map[string]string)
	var layerSizes []int
	
	for scanner.Scan() {
		line := scanner.Text()
		if line == "NEURONS" {
			break
		}
		
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := parts[0]
			value := parts[1]
			params[key] = value
			
			// Extract layer sizes
			if strings.HasPrefix(key, "layer_") && strings.HasSuffix(key, "_size") {
				size, _ := strconv.Atoi(value)
				layerSizes = append(layerSizes, size)
			}
		}
	}
	
	// Create network
	netType := NetTypeLayer
	if val, ok := params["network_type"]; ok {
		t, _ := strconv.Atoi(val)
		netType = NetworkType(t)
	}
	
	connRate := float32(1.0)
	if val, ok := params["connection_rate"]; ok {
		r, _ := strconv.ParseFloat(val, 32)
		connRate = float32(r)
	}
	
	ann := createNetwork[T](netType, connRate, layerSizes)
	
	// Set parameters
	if val, ok := params["learning_rate"]; ok {
		lr, _ := strconv.ParseFloat(val, 32)
		ann.learningRate = float32(lr)
	}
	if val, ok := params["learning_momentum"]; ok {
		lm, _ := strconv.ParseFloat(val, 32)
		ann.learningMomentum = float32(lm)
	}
	if val, ok := params["training_algorithm"]; ok {
		ta, _ := strconv.Atoi(val)
		ann.trainingAlgorithm = TrainAlgorithm(ta)
	}
	
	// Read neurons
	for scanner.Scan() {
		line := scanner.Text()
		if line == "CONNECTIONS" {
			break
		}
		
		fields := strings.Fields(line)
		if len(fields) >= 5 {
			idx, _ := strconv.Atoi(fields[0])
			firstCon, _ := strconv.Atoi(fields[1])
			lastCon, _ := strconv.Atoi(fields[2])
			actFunc, _ := strconv.Atoi(fields[3])
			actSteep, _ := strconv.ParseFloat(fields[4], 64)
			
			if idx < len(ann.neurons) {
				ann.neurons[idx].firstCon = firstCon
				ann.neurons[idx].lastCon = lastCon
				ann.neurons[idx].activationFunction = ActivationFunc(actFunc)
				ann.neurons[idx].activationSteepness = T(actSteep)
			}
		}
	}
	
	// Read connections
	for scanner.Scan() {
		line := scanner.Text()
		if line == "WEIGHTS" {
			break
		}
		
		fields := strings.Fields(line)
		if len(fields) >= 2 {
			idx, _ := strconv.Atoi(fields[0])
			conn, _ := strconv.Atoi(fields[1])
			
			if idx < len(ann.connections) {
				ann.connections[idx] = conn
			}
		}
	}
	
	// Read weights
	for scanner.Scan() {
		line := scanner.Text()
		
		fields := strings.Fields(line)
		if len(fields) >= 2 {
			idx, _ := strconv.Atoi(fields[0])
			weight, _ := strconv.ParseFloat(fields[1], 64)
			
			if idx < len(ann.weights) {
				ann.weights[idx] = T(weight)
			}
		}
	}
	
	return ann, nil
}