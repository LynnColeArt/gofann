package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/lynncoleart/gofann"
)

// CLIErrorPattern represents a single error pattern from our corpus
type CLIErrorPattern struct {
	ID              string                 `json:"id"`
	ErrorMessage    string                 `json:"error_message"`
	ErrorType       string                 `json:"error_type"`
	RootCauses      map[string]float32     `json:"root_causes"`
	UserIntents     []string               `json:"user_intent_patterns"`
	Solutions       []Solution             `json:"solutions"`
}

// Solution represents a solution approach
type Solution struct {
	ID          string   `json:"id"`
	Description string   `json:"description"`
	Commands    []string `json:"commands"`
	SuccessRate float32  `json:"success_rate"`
}

// ErrorCorpus represents the full error dataset
type ErrorCorpus struct {
	ErrorCategory string            `json:"error_category"`
	Patterns      []CLIErrorPattern `json:"patterns"`
}

// FeatureExtractor converts error messages to FANN inputs
type FeatureExtractor struct {
	keywords []string
}

// NewFeatureExtractor creates a feature extractor with common error keywords
func NewFeatureExtractor() *FeatureExtractor {
	return &FeatureExtractor{
		keywords: []string{
			"error", "failed", "cannot", "not found", "denied", "conflict",
			"nil", "null", "undefined", "missing", "invalid", "panic",
			"merge", "rebase", "detached", "HEAD", "branch", "commit",
			"module", "import", "package", "dependency", "install",
		},
	}
}

// Extract converts an error message to feature vector
func (fe *FeatureExtractor) Extract(errorMsg string) []float32 {
	lower := strings.ToLower(errorMsg)
	features := make([]float32, len(fe.keywords))
	
	// Keyword presence features
	for i, keyword := range fe.keywords {
		if strings.Contains(lower, keyword) {
			features[i] = 1.0
		}
	}
	
	// Additional features
	features = append(features,
		float32(len(errorMsg))/1000.0,           // Message length (normalized)
		float32(strings.Count(errorMsg, "\n"))/10.0, // Line count (normalized)
	)
	
	return features
}

// CLIExpertTrainer trains experts for different CLI tools
type CLIExpertTrainer struct {
	gitExpert    *gofann.ReflectiveExpert[float32]
	npmExpert    *gofann.ReflectiveExpert[float32]
	goExpert     *gofann.ReflectiveExpert[float32]
	pythonExpert *gofann.ReflectiveExpert[float32]
	router       *gofann.MoERouter[float32]
	extractor    *FeatureExtractor
}

// NewCLIExpertTrainer creates the expert system
func NewCLIExpertTrainer() *CLIExpertTrainer {
	extractor := NewFeatureExtractor()
	inputSize := len(extractor.keywords) + 2 // keywords + additional features
	
	// Create specialized experts
	gitExpert := gofann.NewReflectiveExpert[float32](
		"GitMaster", "git", 
		[]int{inputSize, 20, 10, 4}, // 4 outputs: error type classification
	)
	
	npmExpert := gofann.NewReflectiveExpert[float32](
		"NPMWizard", "npm",
		[]int{inputSize, 20, 10, 4},
	)
	
	goExpert := gofann.NewReflectiveExpert[float32](
		"GoGuru", "go",
		[]int{inputSize, 20, 10, 4},
	)
	
	pythonExpert := gofann.NewReflectiveExpert[float32](
		"PythonPro", "python",
		[]int{inputSize, 20, 10, 4},
	)
	
	// Create MoE router
	experts := []*gofann.ReflectiveExpert[float32]{
		gitExpert, npmExpert, goExpert, pythonExpert,
	}
	router := gofann.NewMoERouter(experts)
	
	return &CLIExpertTrainer{
		gitExpert:    gitExpert,
		npmExpert:    npmExpert,
		goExpert:     goExpert,
		pythonExpert: pythonExpert,
		router:       router,
		extractor:    extractor,
	}
}

// LoadCorpus loads error patterns from JSON
func LoadCorpus(path string) (*ErrorCorpus, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	
	var corpus ErrorCorpus
	err = json.Unmarshal(data, &corpus)
	return &corpus, err
}

// TrainExpert trains a single expert on its domain data
func (trainer *CLIExpertTrainer) TrainExpert(expert *gofann.ReflectiveExpert[float32], corpusPath string) error {
	corpus, err := LoadCorpus(corpusPath)
	if err != nil {
		return fmt.Errorf("loading corpus: %w", err)
	}
	
	// Create training data
	trainData := gofann.CreateTrainData[float32](
		len(corpus.Patterns),
		len(trainer.extractor.keywords)+2,
		4, // 4 error type outputs
	)
	
	// Convert patterns to training examples
	for i, pattern := range corpus.Patterns {
		input := trainer.extractor.Extract(pattern.ErrorMessage)
		
		// Classification based on error type
		output := make([]float32, 4)
		
		// Use error_type field if available
		errorType := pattern.ErrorType
		if errorType == "" {
			// Fallback to ID-based classification
			switch {
			case strings.Contains(pattern.ID, "detached") || strings.Contains(pattern.ID, "state"):
				errorType = "state"
			case strings.Contains(pattern.ID, "conflict"):
				errorType = "conflict"
			case strings.Contains(pattern.ID, "not_found") || strings.Contains(pattern.ID, "missing"):
				errorType = "missing"
			default:
				errorType = "other"
			}
		}
		
		// Set output based on error type
		switch errorType {
		case "state":
			output[0] = 1.0
		case "conflict":
			output[1] = 1.0
		case "missing":
			output[2] = 1.0
		default:
			output[3] = 1.0
		}
		
		trainData.SetInput(i, input)
		trainData.SetOutput(i, output)
	}
	
	// Train with reflection
	fmt.Printf("Training %s expert on %s...\n", expert.Name(), corpus.ErrorCategory)
	fmt.Printf("  Patterns: %d\n", len(corpus.Patterns))
	metrics := expert.Trainer().TrainWithReflection(trainData)
	fmt.Printf("Final accuracy: %.2f%%\n\n", metrics.Accuracy*100)
	
	return nil
}

// TrainAllExperts trains all CLI experts
func (trainer *CLIExpertTrainer) TrainAllExperts(corpusDir string) error {
	// Train each expert on comprehensive error files
	expertPaths := map[*gofann.ReflectiveExpert[float32]]string{
		trainer.gitExpert:    filepath.Join(corpusDir, "comprehensive_git_errors.json"),
		trainer.npmExpert:    filepath.Join(corpusDir, "comprehensive_npm_errors.json"),
		trainer.goExpert:     filepath.Join(corpusDir, "comprehensive_go_errors.json"),
		trainer.pythonExpert: filepath.Join(corpusDir, "comprehensive_python_errors.json"),
	}
	
	for expert, file := range expertPaths {
		if err := trainer.TrainExpert(expert, file); err != nil {
			log.Printf("Warning: Failed to train %s expert: %v", expert.Name(), err)
		}
	}
	
	// After training, update the router's neural networks
	fmt.Println("🎯 Updating MoE router networks...")
	trainer.router.TrainRoutingNetworks()
	
	return nil
}

// Diagnose analyzes an error message using the expert system
func (trainer *CLIExpertTrainer) Diagnose(errorMsg string) {
	fmt.Printf("\n🔍 Analyzing error:\n%s\n", errorMsg)
	fmt.Println(strings.Repeat("-", 60))
	
	// Extract features
	features := trainer.extractor.Extract(errorMsg)
	
	// Route through MoE system
	output, decision := trainer.router.Route(features)
	
	// Interpret results
	fmt.Printf("🎯 Expert Analysis:\n")
	fmt.Printf("   Confidence: %.2f%%\n", decision.Confidence()*100)
	fmt.Printf("   Active Experts: %v\n", decision.SelectedExperts())
	
	// Classify error type
	errorTypes := []string{"State Error", "Conflict", "Missing Dependency", "Other"}
	maxIdx := 0
	maxVal := output[0]
	for i, val := range output[1:] {
		if val > maxVal {
			maxVal = val
			maxIdx = i + 1
		}
	}
	
	fmt.Printf("   Error Type: %s (%.2f%% confidence)\n", 
		errorTypes[maxIdx], maxVal*100)
	
	// Provide recommendations based on type
	fmt.Printf("\n💡 Recommended Actions:\n")
	switch maxIdx {
	case 0: // State error
		fmt.Println("   1. Check current branch/state with 'git status'")
		fmt.Println("   2. Return to branch with 'git checkout main'")
		fmt.Println("   3. Save work with 'git stash' if needed")
	case 1: // Conflict
		fmt.Println("   1. View conflicts with 'git status'")
		fmt.Println("   2. Edit files to resolve conflicts")
		fmt.Println("   3. Mark resolved with 'git add' and continue")
	case 2: // Missing dependency
		fmt.Println("   1. Install missing package")
		fmt.Println("   2. Check you're in correct environment")
		fmt.Println("   3. Verify package.json/requirements.txt")
	default:
		fmt.Println("   1. Check error details carefully")
		fmt.Println("   2. Search for specific error message")
		fmt.Println("   3. Verify environment and permissions")
	}
}

func main() {
	fmt.Println("🚀 CLI Assistant Expert Training System")
	fmt.Println("=====================================")
	
	// Create trainer
	trainer := NewCLIExpertTrainer()
	
	// Train all experts
	corpusDir := "/media/lynn/big_drive/workspaces/fanmaker/cli-training-corpus"
	fmt.Println("\n📚 Training experts on CLI error patterns...")
	if err := trainer.TrainAllExperts(corpusDir); err != nil {
		log.Fatalf("Training failed: %v", err)
	}
	
	// Test the system
	fmt.Println("\n🧪 Testing the expert system...")
	
	testErrors := []string{
		"You are in 'detached HEAD' state. You can look around, make experimental changes",
		"CONFLICT (content): Merge conflict in app.js",
		"ModuleNotFoundError: No module named 'requests'",
		"panic: runtime error: invalid memory address or nil pointer dereference",
	}
	
	for _, errorMsg := range testErrors {
		trainer.Diagnose(errorMsg)
		fmt.Println()
	}
	
	fmt.Println("✅ CLI Assistant ready! The experts have been trained.")
}