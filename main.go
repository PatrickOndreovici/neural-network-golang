package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

var classIndex = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

var classNames = []string{"Iris-setosa", "Iris-versicolor", "Iris-virginica"}

func loadIrisCSV(path string) ([][]float64, []int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("could not open file: %v", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("could not read file: %v", err)
	}

	var data [][]float64
	var labels []int

	for _, row := range rows {
		if len(row) != 5 {
			return nil, nil, fmt.Errorf("expected 5 columns, got %d", len(row))
		}
		features := make([]float64, 4)
		for j := 0; j < 4; j++ {
			v, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return nil, nil, fmt.Errorf("expected numeric value, got %q", row[j])
			}
			features[j] = v
		}
		className := row[4]
		idx, ok := classIndex[className]
		if !ok {
			return nil, nil, fmt.Errorf("unknown class name: %q", className)
		}
		data = append(data, features)
		labels = append(labels, idx)
	}

	return data, labels, nil
}

func buildMatrices(data [][]float64, labels []int, numClasses int) (*mat.Dense, *mat.Dense) {
	n := len(data)
	features := len(data[0])

	xFlat := make([]float64, features*n)
	for j := 0; j < n; j++ {
		for i := 0; i < features; i++ {
			xFlat[i*n+j] = data[j][i]
		}
	}
	x := mat.NewDense(features, n, xFlat)

	yFlat := make([]float64, numClasses*n)
	for j, lbl := range labels {
		yFlat[lbl*n+j] = 1.0
	}
	y := mat.NewDense(numClasses, n, yFlat)

	return x, y
}

func argmax(vals []float64) int {
	best := 0
	for i, v := range vals {
		if v > vals[best] {
			best = i
		}
	}
	return best
}

func main() {
	const numClasses = 3

	data, labels, err := loadIrisCSV("iris.csv")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	xTrain, yTrain := buildMatrices(data, labels, numClasses)

	config := neuralNetConfig{
		inputNeurons:  4,
		hiddenNeurons: 10,
		outputNeurons: numClasses,
		numEpochs:     5000,
		learningRate:  0.05,
	}

	nn := newNetwork(config)

	fmt.Println("=== Neural Network - Iris Dataset ===")
	fmt.Printf("Examples       : %d\n", len(data))
	fmt.Printf("Features       : %d\n", config.inputNeurons)
	fmt.Printf("Hidden neurons : %d\n", config.hiddenNeurons)
	fmt.Printf("Classes        : %d\n", config.outputNeurons)
	fmt.Printf("Epochs         : %d\n", config.numEpochs)
	fmt.Printf("Learning rate  : %.3f\n\n", config.learningRate)

	fmt.Println("Training...")
	nn.train(xTrain, yTrain)
	fmt.Println("Training complete!\n")

	predictions := nn.predict(xTrain)
	_, numSamples := predictions.Dims()

	correct := 0
	confMatrix := make([][]int, numClasses)
	for i := range confMatrix {
		confMatrix[i] = make([]int, numClasses)
	}

	for j := 0; j < numSamples; j++ {
		col := make([]float64, numClasses)
		for i := 0; i < numClasses; i++ {
			col[i] = predictions.At(i, j)
		}
		predicted := argmax(col)
		actual := labels[j]
		confMatrix[actual][predicted]++
		if predicted == actual {
			correct++
		}
	}

	accuracy := float64(correct) / float64(numSamples) * 100.0
	fmt.Printf("Accuracy: %d/%d (%.1f%%)\n\n", correct, numSamples, accuracy)

	fmt.Println("Confusion matrix (rows=actual, cols=predicted):")
	fmt.Printf("%-20s", "")
	for _, name := range classNames {
		fmt.Printf("%-20s", name)
	}
	fmt.Println()
	for i, row := range confMatrix {
		fmt.Printf("%-20s", classNames[i])
		for _, v := range row {
			fmt.Printf("%-20d", v)
		}
		fmt.Println()
	}

	fmt.Println("\n--- Individual examples (first 5 per class) ---")
	shown := map[int]int{}
	for j := 0; j < numSamples; j++ {
		actual := labels[j]
		if shown[actual] >= 5 {
			continue
		}
		col := make([]float64, numClasses)
		for i := 0; i < numClasses; i++ {
			col[i] = predictions.At(i, j)
		}
		predicted := argmax(col)
		status := "✓"
		if predicted != actual {
			status = "✗"
		}
		fmt.Printf("[%s] #%3d | Actual: %-18s | Predicted: %-18s | Scores: [%.3f, %.3f, %.3f]\n",
			status, j,
			classNames[actual], classNames[predicted],
			col[0], col[1], col[2],
		)
		shown[actual]++
	}
}
