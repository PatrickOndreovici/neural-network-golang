package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	x := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})

	y := mat.NewDense(4, 1, []float64{
		0,
		1,
		1,
		0,
	})

	config := neuralNetConfig{
		inputNeurons:  2,
		hiddenNeurons: 3,
		outputNeurons: 1,
		numEpochs:     1000,
		learningRate:  0.1,
	}

	nn := newNetwork(config)

	xT := mat.DenseCopyOf(x.T())
	yT := mat.DenseCopyOf(y.T())

	nn.train(xT, yT)

	result := nn.predict(xT)
	fmt.Println("Predictions:")
	fmt.Printf("%v\n", mat.Formatted(result))
}
