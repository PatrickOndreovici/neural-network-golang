package main

import "gonum.org/v1/gonum/mat"

type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}
