package main

import (
	"encoding/csv"
	"fmt"
	"math"
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
		return nil, nil, fmt.Errorf("We couldn't open the file: %v", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("We couldn't read the file: %v", err)
	}

	var data [][]float64
	var labels []int

	for _, row := range rows {
		if len(row) != 5 {
			return nil, nil, fmt.Errorf("Expected 5 columns, got %d", len(row))
		}
		features := make([]float64, 4)
		for j := 0; j < 4; j++ {
			v, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return nil, nil, fmt.Errorf("Expected numeric value, got %q", row[j])
			}
			features[j] = v
		}
		className := row[4]
		idx, ok := classIndex[className]
		if !ok {
			return nil, nil, fmt.Errorf("Unknown class name: %q", className)
		}
		data = append(data, features)
		labels = append(labels, idx)
	}

	return data, labels, nil
}

func normalize(data [][]float64) [][]float64 {
	n := len(data)
	if n == 0 {
		return data
	}
	cols := len(data[0])
	mins := make([]float64, cols)
	maxs := make([]float64, cols)
	for j := 0; j < cols; j++ {
		mins[j] = math.MaxFloat64
		maxs[j] = -math.MaxFloat64
	}
	for _, row := range data {
		for j, v := range row {
			if v < mins[j] {
				mins[j] = v
			}
			if v > maxs[j] {
				maxs[j] = v
			}
		}
	}
	result := make([][]float64, n)
	for i, row := range data {
		result[i] = make([]float64, cols)
		for j, v := range row {
			r := maxs[j] - mins[j]
			if r == 0 {
				result[i][j] = 0
			} else {
				result[i][j] = (v - mins[j]) / r
			}
		}
	}
	return result
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

	fmt.Println("=== Neural network - Iris dataset ===")
	fmt.Printf("Examples          : %d\n", len(data))
	fmt.Printf("Features         : %d\n", config.inputNeurons)
	fmt.Printf("Hidden neurons  : %d\n", config.hiddenNeurons)
	fmt.Printf("Classes            : %d\n", config.outputNeurons)
	fmt.Printf("Epoch            : %d\n", config.numEpochs)
	fmt.Printf("Learning rate : %.3f\n\n", config.learningRate)

	fmt.Println("Training...")
	nn.train(xTrain, yTrain)
	fmt.Println("Training done!\n")

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
	fmt.Printf("Acuratețe: %d/%d (%.1f%%)\n\n", correct, numSamples, accuracy)

	fmt.Println("Matrice de confuzie (rând=actual, coloană=prezis):")
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

	fmt.Println("\n--- Exemple individuale (primele 5 din fiecare clasă) ---")
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
		fmt.Printf("[%s] #%3d | Actual: %-18s | Prezis: %-18s | Scoruri: [%.3f, %.3f, %.3f]\n",
			status, j,
			classNames[actual], classNames[predicted],
			col[0], col[1], col[2],
		)
		shown[actual]++
	}
}
