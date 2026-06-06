# Iris Neural Network from Scratch (Go)

This project implements a neural network built completely from scratch in Go (Golang) to classify the Iris dataset.

The goal of this project is to understand how neural networks work internally without using any machine learning libraries.

---

## Description

A simple feedforward neural network was implemented to classify Iris flowers into three categories:

- Setosa
- Versicolor
- Virginica

---

## Dataset

The project uses the classic Iris dataset, which contains:

- 150 samples
- 4 features:
    - sepal length
    - sepal width
    - petal length
    - petal width
- 3 output classes

---

## What was implemented

The neural network was built entirely from scratch and includes:

- forward propagation
- backpropagation
- gradient descent
- weight and bias updates
- activation functions
- loss computation

---

## Training

- supervised learning approach
- train/test split
- feature standardization (normalization)
- optimization using gradient descent

---

## Results

The model achieves approximately:

- Accuracy: 98% (147 out of 150 correct predictions)

For the Iris dataset, this is considered a very good result.

---

## How to run

```bash
git clone https://github.com/username/iris-nn-go.git
cd iris-nn-go
go run main.go