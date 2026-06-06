package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func addBias(z, b *mat.Dense) *mat.Dense {
	rows, cols := z.Dims()
	result := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, z.At(i, j)+b.At(i, 0))
		}
	}
	return result
}

func sumCols(m *mat.Dense) *mat.Dense {
	rows, cols := m.Dims()
	result := mat.NewDense(rows, 1, nil)
	for i := 0; i < rows; i++ {
		s := 0.0
		for j := 0; j < cols; j++ {
			s += m.At(i, j)
		}
		result.Set(i, 0, s)
	}
	return result
}

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

func (nn *neuralNet) forwardFull(x *mat.Dense) (z1, h, z2, yHat *mat.Dense) {
	var z1mat mat.Dense
	z1mat.Mul(nn.wHidden, x)
	z1 = addBias(&z1mat, nn.bHidden)

	h = mat.DenseCopyOf(z1)
	h.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, h)

	var z2mat mat.Dense
	z2mat.Mul(nn.wOut, h)
	z2 = addBias(&z2mat, nn.bOut)

	yHat = mat.DenseCopyOf(z2)
	yHat.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, yHat)

	return
}

func (nn *neuralNet) backward(x, y *mat.Dense) {
	z1, h, z2, yHat := nn.forwardFull(x)

	var dSigZ2 mat.Dense
	dSigZ2.Apply(func(_, _ int, v float64) float64 { return sigmoidPrime(v) }, z2)

	var dSigZ1 mat.Dense
	dSigZ1.Apply(func(_, _ int, v float64) float64 { return sigmoidPrime(v) }, z1)

	var dL_dz2 mat.Dense
	dL_dz2.Sub(yHat, y)
	dL_dz2.MulElem(&dL_dz2, &dSigZ2)

	var dL_dW2 mat.Dense
	dL_dW2.Mul(&dL_dz2, h.T())

	dL_dB2 := sumCols(&dL_dz2)

	var dL_dh mat.Dense
	dL_dh.Mul(nn.wOut.T(), &dL_dz2)

	var dL_dz1 mat.Dense
	dL_dz1.MulElem(&dL_dh, &dSigZ1)

	var dL_dW1 mat.Dense
	dL_dW1.Mul(&dL_dz1, x.T())

	dL_dB1 := sumCols(&dL_dz1)

	lr := nn.config.learningRate
	nn.wOut.Apply(func(i, j int, v float64) float64 { return v - lr*dL_dW2.At(i, j) }, nn.wOut)
	nn.bOut.Apply(func(i, j int, v float64) float64 { return v - lr*dL_dB2.At(i, 0) }, nn.bOut)
	nn.wHidden.Apply(func(i, j int, v float64) float64 { return v - lr*dL_dW1.At(i, j) }, nn.wHidden)
	nn.bHidden.Apply(func(i, j int, v float64) float64 { return v - lr*dL_dB1.At(i, 0) }, nn.bHidden)
}

func (nn *neuralNet) train(x, y *mat.Dense) {
	nn.initWeights()
	for epoch := 0; epoch < nn.config.numEpochs; epoch++ {
		nn.backward(x, y)
	}
}

func (nn *neuralNet) predict(x *mat.Dense) *mat.Dense {
	_, _, _, yHat := nn.forwardFull(x)
	return yHat
}
