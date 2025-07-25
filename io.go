package gofann

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// FANN file format constants
const (
	fannFileVersion = "FANN_FLO_2.1"
	fixedPointDecimals = 6
)

// Save saves the network to a file in FANN format
func (ann *Fann[T]) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		ann.setError(ErrCantOpenWriter, err.Error())
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// Write header
	fmt.Fprintln(writer, fannFileVersion)
	fmt.Fprintf(writer, "num_layers=%d\n", len(ann.layers))
	fmt.Fprintf(writer, "learning_rate=%f\n", ann.learningRate)
	fmt.Fprintf(writer, "connection_rate=%f\n", ann.connectionRate)
	fmt.Fprintf(writer, "network_type=%d\n", ann.networkType)
	fmt.Fprintf(writer, "learning_momentum=%f\n", ann.learningMomentum)
	fmt.Fprintf(writer, "training_algorithm=%d\n", ann.trainingAlgorithm)
	fmt.Fprintf(writer, "train_error_function=%d\n", ann.trainErrorFunction)
	fmt.Fprintf(writer, "train_stop_function=%d\n", ann.trainStopFunction)
	fmt.Fprintf(writer, "cascade_output_change_fraction=%f\n", ann.cascadeOutputChangeFraction)
	fmt.Fprintf(writer, "quickprop_decay=%f\n", ann.quickpropDecay)
	fmt.Fprintf(writer, "quickprop_mu=%f\n", ann.quickpropMu)
	fmt.Fprintf(writer, "rprop_increase_factor=%f\n", ann.rpropIncreaseFactor)
	fmt.Fprintf(writer, "rprop_decrease_factor=%f\n", ann.rpropDecreaseFactor)
	fmt.Fprintf(writer, "rprop_delta_min=%f\n", ann.rpropDeltaMin)
	fmt.Fprintf(writer, "rprop_delta_max=%f\n", ann.rpropDeltaMax)
	fmt.Fprintf(writer, "rprop_delta_zero=%f\n", ann.rpropDeltaZero)
	fmt.Fprintf(writer, "cascade_output_stagnation_epochs=%d\n", ann.cascadeOutputStagnationEpochs)
	fmt.Fprintf(writer, "cascade_candidate_change_fraction=%f\n", ann.cascadeCandidateChangeFraction)
	fmt.Fprintf(writer, "cascade_candidate_stagnation_epochs=%d\n", ann.cascadeCandidateStagnationEpochs)
	fmt.Fprintf(writer, "cascade_max_out_epochs=%d\n", ann.cascadeMaxOutEpochs)
	fmt.Fprintf(writer, "cascade_min_out_epochs=%d\n", ann.cascadeMinOutEpochs)
	fmt.Fprintf(writer, "cascade_max_cand_epochs=%d\n", ann.cascadeMaxCandEpochs)
	fmt.Fprintf(writer, "cascade_min_cand_epochs=%d\n", ann.cascadeMinCandEpochs)
	fmt.Fprintf(writer, "cascade_num_candidate_groups=%d\n", ann.cascadeNumCandidateGroups)
	
	// Bit fail limit
	fmt.Fprintf(writer, "bit_fail_limit=%f\n", float32(ann.bitFailLimit))
	fmt.Fprintf(writer, "cascade_candidate_limit=%f\n", float32(ann.cascadeCandidateLimit))
	fmt.Fprintf(writer, "cascade_weight_multiplier=%f\n", float32(ann.cascadeWeightMultiplier))
	
	// Cascade activation functions
	fmt.Fprintf(writer, "cascade_activation_functions_count=%d\n", len(ann.cascadeActivationFunctions))
	fmt.Fprintf(writer, "cascade_activation_functions=")
	for i, fn := range ann.cascadeActivationFunctions {
		if i > 0 {
			fmt.Fprint(writer, " ")
		}
		fmt.Fprintf(writer, "%d", fn)
	}
	fmt.Fprintln(writer)
	
	// Cascade activation steepnesses
	fmt.Fprintf(writer, "cascade_activation_steepnesses_count=%d\n", len(ann.cascadeActivationSteepnesses))
	fmt.Fprintf(writer, "cascade_activation_steepnesses=")
	for i, steep := range ann.cascadeActivationSteepnesses {
		if i > 0 {
			fmt.Fprint(writer, " ")
		}
		fmt.Fprintf(writer, "%f", float32(steep))
	}
	fmt.Fprintln(writer)
	
	// Layer sizes
	fmt.Fprint(writer, "layer_sizes=")
	for i, layer := range ann.layers {
		if i > 0 {
			fmt.Fprint(writer, " ")
		}
		size := layer.lastNeuron - layer.firstNeuron
		if i < len(ann.layers)-1 {
			size-- // Don't count bias neuron
		}
		fmt.Fprintf(writer, "%d", size)
	}
	fmt.Fprintln(writer)
	
	// Scale parameters (not implemented yet)
	fmt.Fprintln(writer, "scale_included=0")
	
	// Write neurons
	fmt.Fprintln(writer, "neurons (num_inputs, activation_function, activation_steepness)=")
	for i, layer := range ann.layers {
		for j := layer.firstNeuron; j < layer.lastNeuron; j++ {
			neuron := &ann.neurons[j]
			
			// Skip bias neurons in hidden layers
			if i < len(ann.layers)-1 && j == layer.lastNeuron-1 {
				// Bias neuron
				fmt.Fprintf(writer, "(%d, 0, 0.000000) ", 1)
			} else {
				numInputs := neuron.lastCon - neuron.firstCon
				fmt.Fprintf(writer, "(%d, %d, %f) ", 
					numInputs, 
					neuron.activationFunction,
					float32(neuron.activationSteepness))
			}
		}
		fmt.Fprintln(writer)
	}
	
	// Write connections
	fmt.Fprintln(writer, "connections (connected_to_neuron, weight)=")
	for i := 1; i < len(ann.layers); i++ {
		layer := ann.layers[i]
		
		for j := layer.firstNeuron; j < layer.lastNeuron; j++ {
			if i < len(ann.layers)-1 && j == layer.lastNeuron-1 {
				// Bias neuron - write dummy connection
				fmt.Fprintln(writer, "(0, 0.000000) ")
			} else {
				fmt.Fprintln(writer)
				neuron := &ann.neurons[j]
				for k := neuron.firstCon; k < neuron.lastCon; k++ {
					fmt.Fprintf(writer, "(%d, %f) ", 
						ann.connections[k], 
						float32(ann.weights[k]))
				}
			}
		}
	}
	
	return nil
}

// SaveToFixed saves the network in fixed-point format with specified decimal precision
func (ann *Fann[T]) SaveToFixed(filename string, decimalPoint uint) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("can't create fixed-point config file: %w", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// Calculate multiplier for fixed point conversion
	multiplier := float64(uint(1) << decimalPoint)

	// Write header - same format as regular FANN files but indicate fixed-point
	fmt.Fprintf(writer, "FANN_FIX_%d.0\n", 1<<decimalPoint) // Fixed-point indicator
	fmt.Fprintf(writer, "num_layers=%d\n", len(ann.layers))
	fmt.Fprintf(writer, "learning_rate=%d\n", int(float64(ann.learningRate)*multiplier))
	fmt.Fprintf(writer, "connection_rate=%d\n", int(float64(ann.connectionRate)*multiplier))
	fmt.Fprintf(writer, "network_type=%d\n", int(ann.networkType))
	fmt.Fprintf(writer, "learning_momentum=%d\n", int(float64(ann.learningMomentum)*multiplier))
	fmt.Fprintf(writer, "training_algorithm=%d\n", int(ann.trainingAlgorithm))
	fmt.Fprintf(writer, "train_error_function=%d\n", int(ann.trainErrorFunction))
	fmt.Fprintf(writer, "train_stop_function=%d\n", int(ann.trainStopFunction))
	fmt.Fprintf(writer, "cascade_output_change_fraction=%d\n", int(float64(ann.cascadeOutputChangeFraction)*multiplier))
	fmt.Fprintf(writer, "quickprop_decay=%d\n", int(float64(ann.quickpropDecay)*multiplier))
	fmt.Fprintf(writer, "quickprop_mu=%d\n", int(float64(ann.quickpropMu)*multiplier))
	fmt.Fprintf(writer, "rprop_increase_factor=%d\n", int(float64(ann.rpropIncreaseFactor)*multiplier))
	fmt.Fprintf(writer, "rprop_decrease_factor=%d\n", int(float64(ann.rpropDecreaseFactor)*multiplier))
	fmt.Fprintf(writer, "rprop_delta_min=%d\n", int(float64(ann.rpropDeltaMin)*multiplier))
	fmt.Fprintf(writer, "rprop_delta_max=%d\n", int(float64(ann.rpropDeltaMax)*multiplier))
	fmt.Fprintf(writer, "rprop_delta_zero=%d\n", int(float64(ann.rpropDeltaZero)*multiplier))
	fmt.Fprintf(writer, "cascade_output_stagnation_epochs=%d\n", ann.cascadeOutputStagnationEpochs)
	fmt.Fprintf(writer, "cascade_candidate_change_fraction=%d\n", int(float64(ann.cascadeCandidateChangeFraction)*multiplier))
	fmt.Fprintf(writer, "cascade_candidate_stagnation_epochs=%d\n", ann.cascadeCandidateStagnationEpochs)
	fmt.Fprintf(writer, "cascade_max_out_epochs=%d\n", ann.cascadeMaxOutEpochs)
	fmt.Fprintf(writer, "cascade_min_out_epochs=%d\n", ann.cascadeMinOutEpochs)
	fmt.Fprintf(writer, "cascade_max_cand_epochs=%d\n", ann.cascadeMaxCandEpochs)
	fmt.Fprintf(writer, "cascade_min_cand_epochs=%d\n", ann.cascadeMinCandEpochs)
	fmt.Fprintf(writer, "cascade_num_candidate_groups=%d\n", ann.cascadeNumCandidateGroups)
	fmt.Fprintf(writer, "bit_fail_limit=%d\n", int(float64(ann.bitFailLimit)*multiplier))
	fmt.Fprintf(writer, "cascade_candidate_limit=%d\n", int(float64(ann.cascadeCandidateLimit)*multiplier))
	fmt.Fprintf(writer, "cascade_weight_multiplier=%d\n", int(float64(ann.cascadeWeightMultiplier)*multiplier))
	fmt.Fprintf(writer, "cascade_activation_functions_count=%d\n", len(ann.cascadeActivationFunctions))
	
	// Write activation functions
	for _, af := range ann.cascadeActivationFunctions {
		fmt.Fprintf(writer, "%d ", int(af))
	}
	fmt.Fprintln(writer)
	
	fmt.Fprintf(writer, "cascade_activation_steepnesses_count=%d\n", len(ann.cascadeActivationSteepnesses))
	
	// Write activation steepnesses as fixed-point
	for _, steepness := range ann.cascadeActivationSteepnesses {
		fmt.Fprintf(writer, "%d ", int(float64(steepness)*multiplier))
	}
	fmt.Fprintln(writer)

	// Write layer info
	fmt.Fprintf(writer, "layer_sizes:")
	for _, layer := range ann.layers {
		fmt.Fprintf(writer, "%d ", layer.lastNeuron-layer.firstNeuron)
	}
	fmt.Fprintln(writer)

	fmt.Fprintf(writer, "scale_included=0\n") // Fixed-point doesn't use scaling
	fmt.Fprintf(writer, "neurons (num_inputs, activation_function, activation_steepness):\n")

	// Write neuron info with fixed-point steepness
	for i := 0; i < ann.totalNeurons; i++ {
		neuron := ann.neurons[i]
		numConnections := neuron.lastCon - neuron.firstCon
		steepnessFixed := int(float64(neuron.activationSteepness) * multiplier)
		fmt.Fprintf(writer, "(%d, %d, %d)\n", numConnections, int(neuron.activationFunction), steepnessFixed)
	}

	fmt.Fprintf(writer, "connections (connected_to_neuron, weight):\n")
	
	// Write connections with fixed-point weights
	for i := 0; i < ann.totalNeurons; i++ {
		neuron := ann.neurons[i]
		for connIdx := neuron.firstCon; connIdx < neuron.lastCon; connIdx++ {
			sourceNeuron := ann.connections[connIdx]
			weightFixed := int(float64(ann.weights[connIdx]) * multiplier)
			fmt.Fprintf(writer, "(%d, %d)\n", sourceNeuron, weightFixed)
		}
	}

	return nil
}

// CreateFromFile loads a neural network from a file
func CreateFromFile[T Numeric](filename string) (*Fann[T], error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("can't open config file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	
	// Read header
	if !scanner.Scan() {
		return nil, fmt.Errorf("can't read config file header")
	}
	version := scanner.Text()
	if !strings.HasPrefix(version, "FANN") {
		return nil, fmt.Errorf("wrong config file format")
	}
	
	// Parse configuration
	config := make(map[string]string)
	var layerSizes []int
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		
		if strings.HasPrefix(line, "neurons") {
			break // Start of neuron section
		}
		
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])
			config[key] = value
			
			if key == "layer_sizes" {
				// Parse layer sizes
				sizes := strings.Fields(value)
				for _, size := range sizes {
					n, err := strconv.Atoi(size)
					if err != nil {
						return nil, fmt.Errorf("invalid layer size: %s", size)
					}
					layerSizes = append(layerSizes, n)
				}
			}
		}
	}
	
	if len(layerSizes) < 2 {
		return nil, fmt.Errorf("invalid layer configuration")
	}
	
	// Get network type
	netType := NetTypeLayer
	if val, ok := config["network_type"]; ok {
		t, _ := strconv.Atoi(val)
		netType = NetworkType(t)
	}
	
	// Get connection rate
	connRate := float32(1.0)
	if val, ok := config["connection_rate"]; ok {
		r, _ := strconv.ParseFloat(val, 32)
		connRate = float32(r)
	}
	
	// Create network - but we'll reconstruct connections manually
	ann := &Fann[T]{
		networkType:       netType,
		connectionRate:    connRate,
		numInput:          layerSizes[0],
		numOutput:         layerSizes[len(layerSizes)-1],
		learningRate:      0.7,
		learningMomentum:  0.0,
		trainingAlgorithm: TrainRPROP,
		trainErrorFunction: ErrorTanh,
		bitFailLimit:      T(0.35),
		rpropIncreaseFactor: 1.2,
		rpropDecreaseFactor: 0.5,
		rpropDeltaMin:      0.000001,
		rpropDeltaMax:      50.0,
		rpropDeltaZero:     0.0125,
	}
	
	// Calculate total neurons (including bias neurons)
	totalNeurons := 0
	for i, size := range layerSizes {
		totalNeurons += size
		if i != len(layerSizes)-1 { // Add bias neuron for all but output layer
			totalNeurons++
		}
	}
	ann.totalNeurons = totalNeurons
	ann.neurons = make([]neuron[T], totalNeurons)
	
	// Set up layers
	ann.layers = make([]layer[T], len(layerSizes))
	neuronIdx := 0
	for i, size := range layerSizes {
		ann.layers[i].firstNeuron = neuronIdx
		neuronIdx += size
		if i != len(layerSizes)-1 { // Add bias neuron
			neuronIdx++
		}
		ann.layers[i].lastNeuron = neuronIdx
	}
	
	// Initialize neurons with defaults
	for i := range ann.neurons {
		ann.neurons[i].activationFunction = Sigmoid
		ann.neurons[i].activationSteepness = T(0.5)
	}
	
	// Set parameters
	if val, ok := config["learning_rate"]; ok {
		lr, _ := strconv.ParseFloat(val, 32)
		ann.learningRate = float32(lr)
	}
	if val, ok := config["learning_momentum"]; ok {
		lm, _ := strconv.ParseFloat(val, 32)
		ann.learningMomentum = float32(lm)
	}
	if val, ok := config["training_algorithm"]; ok {
		ta, _ := strconv.Atoi(val)
		ann.trainingAlgorithm = TrainAlgorithm(ta)
	}
	if val, ok := config["train_error_function"]; ok {
		tef, _ := strconv.Atoi(val)
		ann.trainErrorFunction = ErrorFunc(tef)
	}
	if val, ok := config["bit_fail_limit"]; ok {
		bfl, _ := strconv.ParseFloat(val, 64)
		ann.bitFailLimit = T(bfl)
	}
	
	// RPROP parameters
	if val, ok := config["rprop_increase_factor"]; ok {
		rif, _ := strconv.ParseFloat(val, 32)
		ann.rpropIncreaseFactor = float32(rif)
	}
	if val, ok := config["rprop_decrease_factor"]; ok {
		rdf, _ := strconv.ParseFloat(val, 32)
		ann.rpropDecreaseFactor = float32(rdf)
	}
	if val, ok := config["rprop_delta_min"]; ok {
		rdmin, _ := strconv.ParseFloat(val, 32)
		ann.rpropDeltaMin = float32(rdmin)
	}
	if val, ok := config["rprop_delta_max"]; ok {
		rdmax, _ := strconv.ParseFloat(val, 32)
		ann.rpropDeltaMax = float32(rdmax)
	}
	if val, ok := config["rprop_delta_zero"]; ok {
		rdz, _ := strconv.ParseFloat(val, 32)
		ann.rpropDeltaZero = float32(rdz)
	}
	
	// Quickprop parameters
	if val, ok := config["quickprop_decay"]; ok {
		qd, _ := strconv.ParseFloat(val, 32)
		ann.quickpropDecay = float32(qd)
	}
	if val, ok := config["quickprop_mu"]; ok {
		qm, _ := strconv.ParseFloat(val, 32)
		ann.quickpropMu = float32(qm)
	}
	
	// Cascade parameters
	if val, ok := config["cascade_output_change_fraction"]; ok {
		cocf, _ := strconv.ParseFloat(val, 32)
		ann.cascadeOutputChangeFraction = float32(cocf)
	}
	if val, ok := config["cascade_output_stagnation_epochs"]; ok {
		cose, _ := strconv.Atoi(val)
		ann.cascadeOutputStagnationEpochs = cose
	}
	if val, ok := config["cascade_candidate_change_fraction"]; ok {
		cccf, _ := strconv.ParseFloat(val, 32)
		ann.cascadeCandidateChangeFraction = float32(cccf)
	}
	if val, ok := config["cascade_candidate_stagnation_epochs"]; ok {
		ccse, _ := strconv.Atoi(val)
		ann.cascadeCandidateStagnationEpochs = ccse
	}
	if val, ok := config["cascade_max_out_epochs"]; ok {
		cmoe, _ := strconv.Atoi(val)
		ann.cascadeMaxOutEpochs = cmoe
	}
	if val, ok := config["cascade_min_out_epochs"]; ok {
		cmioe, _ := strconv.Atoi(val)
		ann.cascadeMinOutEpochs = cmioe
	}
	if val, ok := config["cascade_max_cand_epochs"]; ok {
		cmce, _ := strconv.Atoi(val)
		ann.cascadeMaxCandEpochs = cmce
	}
	if val, ok := config["cascade_min_cand_epochs"]; ok {
		cmice, _ := strconv.Atoi(val)
		ann.cascadeMinCandEpochs = cmice
	}
	if val, ok := config["cascade_num_candidate_groups"]; ok {
		cncg, _ := strconv.Atoi(val)
		ann.cascadeNumCandidateGroups = cncg
	}
	if val, ok := config["cascade_candidate_limit"]; ok {
		ccl, _ := strconv.ParseFloat(val, 64)
		ann.cascadeCandidateLimit = int(ccl)
	}
	if val, ok := config["cascade_weight_multiplier"]; ok {
		cwm, _ := strconv.ParseFloat(val, 64)
		ann.cascadeWeightMultiplier = T(cwm)
	}
	
	// First pass: count connections and read neurons
	totalConnections := 0
	neuronIdx = 0
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "connections") {
			break // Start of connections section
		}
		
		// Parse neuron entries: (num_inputs, activation_function, activation_steepness)
		neurons := strings.Split(line, ") ")
		for _, neuronStr := range neurons {
			neuronStr = strings.TrimSpace(neuronStr)
			if neuronStr == "" {
				continue
			}
			
			// Remove parentheses
			neuronStr = strings.Trim(neuronStr, "()")
			parts := strings.Split(neuronStr, ", ")
			if len(parts) >= 3 && neuronIdx < len(ann.neurons) {
				// Parse num_inputs for connection count
				numInputs, _ := strconv.Atoi(parts[0])
				
				// Set connection indices
				ann.neurons[neuronIdx].firstCon = totalConnections
				totalConnections += numInputs
				ann.neurons[neuronIdx].lastCon = totalConnections
				
				// Parse activation function
				af, _ := strconv.Atoi(parts[1])
				ann.neurons[neuronIdx].activationFunction = ActivationFunc(af)
				
				// Parse activation steepness
				as, _ := strconv.ParseFloat(parts[2], 64)
				ann.neurons[neuronIdx].activationSteepness = T(as)
				
				neuronIdx++
			}
		}
	}
	
	// Allocate connection and weight arrays
	ann.totalConnections = totalConnections
	ann.connections = make([]int, totalConnections)
	ann.weights = make([]T, totalConnections)
	
	// Read connections - simpler approach
	connIdx := 0
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		
		// Parse connection entries: (connected_to_neuron, weight)
		connections := strings.Split(line, ") ")
		for _, connStr := range connections {
			connStr = strings.TrimSpace(connStr)
			if connStr == "" {
				continue
			}
			
			// Remove parentheses
			connStr = strings.Trim(connStr, "()")
			parts := strings.Split(connStr, ", ")
			if len(parts) >= 2 && connIdx < len(ann.connections) {
				// Parse source neuron
				src, _ := strconv.Atoi(parts[0])
				ann.connections[connIdx] = src
				
				// Parse weight
				weight, _ := strconv.ParseFloat(parts[1], 64)
				ann.weights[connIdx] = T(weight)
				
				connIdx++
			}
		}
	}
	
	return ann, nil
}

// Network configuration getters


// GetLayerSizes returns the size of each layer
func (ann *Fann[T]) GetLayerSizes() []int {
	sizes := make([]int, len(ann.layers))
	for i, layer := range ann.layers {
		size := layer.lastNeuron - layer.firstNeuron
		if i < len(ann.layers)-1 {
			size-- // Don't count bias neuron
		}
		sizes[i] = size
	}
	return sizes
}


// GetWeightArray returns a copy of all weights
func (ann *Fann[T]) GetWeightArray() []T {
	weights := make([]T, len(ann.weights))
	copy(weights, ann.weights)
	return weights
}


