package net


import (
	"math"
	"math/rand/v2"
)


type Net struct {
	Weights [][][]float64
	Biases [][]float64
	Outputs [][]float64
	Deltas [][]float64

	NeuronsNumber []int
}


func (net *Net)createNeurons() {
	le := len(net.NeuronsNumber)

	for l := 0; l < le; l += 1 {
		net.Outputs = append(net.Outputs, []float64{})
		net.Deltas = append(net.Deltas, []float64{})
		for i := 0; i < net.NeuronsNumber[l]; i += 1 {
			net.Outputs[l] = append(net.Outputs[l], 0);
			net.Deltas[l] = append(net.Deltas[l], 0);
		}
	}

	for l := 1; l < le; l += 1 {
		net.Biases = append(net.Biases, []float64{})
		for i := 0; i < net.NeuronsNumber[l]; i += 1 {
			net.Biases[l - 1] = append(net.Biases[l - 1], rand.Float64() * 2 - 1);
		}
	}
}


func (net *Net)createLinks() {
	le := len(net.NeuronsNumber)

	for l := 0; l < le - 1; l += 1 {
		net.Weights = append(net.Weights, [][]float64{})
		for from := 0; from < net.NeuronsNumber[l]; from += 1 {
			net.Weights[l] = append(net.Weights[l], []float64{})
			for to := 0; to < net.NeuronsNumber[l + 1]; to += 1 {
				net.Weights[l][from] = append(net.Weights[l][from], rand.Float64() * 2 - 1)
			}
		}
	}
}


func (net *Net)resetNeurons(inputData []float64) {
	le := len(net.NeuronsNumber)

	for i := 0; i < net.NeuronsNumber[0]; i += 1 {
		net.Outputs[0][i] = inputData[i]
	}

	for l := 1; l < le; l += 1 {
		for i := 0; i < net.NeuronsNumber[l]; i += 1 {
			net.Outputs[l][i] = 0
		}
	}
}


func (net *Net)Forward(inputData []float64) []float64 {
	net.resetNeurons(inputData)

	le := len(net.NeuronsNumber)

	for l := 0; l < le - 1; l += 1 {
		for to := 0; to < net.NeuronsNumber[l + 1]; to += 1 {
			sum := net.Biases[l][to]

			for from := 0; from < net.NeuronsNumber[l]; from += 1 {
				sum += net.Outputs[l][from] * net.Weights[l][from][to]
			}

			net.Outputs[l + 1][to] = sigmoid(sum)
		}
	}

	res := []float64{}

	for i := 0; i < len(net.Outputs[le - 1]); i += 1 {
		res = append(res, net.Outputs[le - 1][i])
	}

	return res
}


func (net *Net)Backward(target []float64, learningRate float64) {
	le := len(net.NeuronsNumber)

	// 1. Вычисляем дельты для выходного слоя
	for i := 0; i < net.NeuronsNumber[le-1]; i++ {
		output := net.Outputs[le-1][i]
		net.Deltas[le-1][i] = (output - target[i]) * output * (1 - output) // sigmoid'
	}

	// 2. Вычисляем дельты для скрытых слоёв (обратный проход)
	for l := le - 2; l >= 0; l-- {
		for i := 0; i < net.NeuronsNumber[l]; i++ {
			sum := 0.0
			for j := 0; j < net.NeuronsNumber[l+1]; j++ {
				sum += net.Weights[l][i][j] * net.Deltas[l+1][j]
			}
			output := net.Outputs[l][i]
			net.Deltas[l][i] = sum * output * (1 - output) // sigmoid'
		}
	}

	// 3. Обновляем веса и bias
	for l := 0; l < le-1; l++ {
		for i := 0; i < net.NeuronsNumber[l]; i++ {
			for j := 0; j < net.NeuronsNumber[l+1]; j++ {
				net.Weights[l][i][j] -= learningRate * net.Outputs[l][i] * net.Deltas[l+1][j]
			}
		}
		for j := 0; j < net.NeuronsNumber[l+1]; j++ {
			net.Biases[l][j] -= learningRate * net.Deltas[l+1][j]
		}
	}
}


func CreateNet(neuronsNumber []int) *Net {
	net := &Net{
		Weights: [][][]float64{},
		Biases: [][]float64{},
		Outputs: [][]float64{},
		Deltas: [][]float64{},

		NeuronsNumber: neuronsNumber,
	}

	net.createNeurons()
	net.createLinks()

	return net
}


func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
