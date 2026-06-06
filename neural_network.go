package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func (nn *neuralNet) initWeights() {
	nn.wHidden = mat.NewDense(nn.config.hiddenNeurons, nn.config.inputNeurons, nil)
	nn.bHidden = mat.NewDense(nn.config.hiddenNeurons, 1, nil)
	nn.wOut = mat.NewDense(nn.config.outputNeurons, nn.config.hiddenNeurons, nil)
	nn.bOut = mat.NewDense(nn.config.outputNeurons, 1, nil)

	n, m := nn.wHidden.Dims()

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			nn.wHidden.Set(i, j, rand.Float64()*2-1)
		}
	}

	for i := 0; i < n; i++ {
		nn.bHidden.Set(i, 0, rand.Float64()*2-1)
	}

	n, m = nn.wOut.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			nn.wOut.Set(i, j, rand.Float64()*2-1)
		}
	}

	for i := 0; i < n; i++ {
		nn.bOut.Set(i, 0, rand.Float64()*2-1)
	}

}

func (nn *neuralNet) forward(x *mat.Dense) *mat.Dense {

	var z1 mat.Dense
	z1.Mul(nn.wHidden, x)
	z1.Add(&z1, nn.bHidden)

	h := mat.DenseCopyOf(&z1)

	applySigmoid := func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}
	h.Apply(applySigmoid, h)

	var z2 mat.Dense
	z2.Mul(nn.wOut, h)
	z2.Add(&z2, nn.bOut)

	y := mat.DenseCopyOf(&z2)
	y.Apply(applySigmoid, y)

	return y
}

func (nn *neuralNet) backward(x, y, yHat *mat.Dense) {
	var z1 mat.Dense

	z1.Mul(nn.wHidden, x)
	z1.Add(&z1, nn.bHidden)

	var sigmoid_z1 mat.Dense

	sigmoid_z1.Apply(func(i, j int, v float64) float64 {
		return sigmoid(v)
	}, &z1)

	var derivated_sigmoid_z1 mat.Dense

	derivated_sigmoid_z1.Apply(func(i, j int, v float64) float64 {
		return sigmoidPrime(v)
	}, &z1)

	var z2 mat.Dense

	z2.Mul(nn.wOut, &sigmoid_z1)
	z2.Add(&z2, nn.bOut)

	var sigmoid_z2 mat.Dense

	sigmoid_z2.Apply(func(i, j int, v float64) float64 {
		return sigmoid(v)
	}, &z2)

	var derivated_sigmoid_z2 mat.Dense

	derivated_sigmoid_z2.Apply(func(i, j int, v float64) float64 {
		return sigmoidPrime(v)
	}, &z2)

	// dL_dz2 = (yHat - y) ⊙ σ'(z2)
	var dL_dz2 mat.Dense
	dL_dz2.Sub(&sigmoid_z2, y)
	dL_dz2.MulElem(&dL_dz2, &derivated_sigmoid_z2)

	// dL_dW2 = dL_dz2 · hᵀ
	var dL_dW2 mat.Dense
	dL_dW2.Mul(&dL_dz2, sigmoid_z1.T())

	// dL_dB2 = dL_dz2
	dL_dB2 := mat.DenseCopyOf(&dL_dz2)

	// dL_dh = W2ᵀ · dL_dz2
	var dL_dh mat.Dense
	dL_dh.Mul(nn.wOut.T(), &dL_dz2)

	// dL_dz1 = dL_dh ⊙ σ'(z1)
	var dL_dz1 mat.Dense
	dL_dz1.MulElem(&dL_dh, &derivated_sigmoid_z1)

	// dL_dW1 = dL_dz1 · Xᵀ
	var dL_dW1 mat.Dense
	dL_dW1.Mul(&dL_dz1, x.T())

	// dL_dB1 = dL_dz1
	dL_dB1 := mat.DenseCopyOf(&dL_dz1)

	nn.wOut.Apply(func(i, j int, v float64) float64 { return v - nn.config.learningRate*dL_dW2.At(i, j) }, nn.wOut)
	nn.bOut.Apply(func(i, j int, v float64) float64 { return v - nn.config.learningRate*dL_dB2.At(i, j) }, nn.bOut)
	nn.wHidden.Apply(func(i, j int, v float64) float64 { return v - nn.config.learningRate*dL_dW1.At(i, j) }, nn.wHidden)
	nn.bHidden.Apply(func(i, j int, v float64) float64 { return v - nn.config.learningRate*dL_dB1.At(i, j) }, nn.bHidden)

}

func (nn *neuralNet) train(x, y *mat.Dense) {
	nn.initWeights()

	for epoch := 0; epoch < nn.config.numEpochs; epoch++ {
		yHat := nn.forward(x)
		nn.backward(x, y, yHat)
	}
}

func (nn *neuralNet) predict(x *mat.Dense) *mat.Dense {
	return nn.forward(x)
}
