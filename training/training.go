package training

import (
	"fmt"
	"slices"
	"math/rand/v2"

	"github.com/min-ok/net"
)


func Training(layers []int, trainingData [][]float64, trainingAnswers [][]float64, testingData [][]float64, testingAnswers [][]float64, expectedResult float64, batchSize int, learningRate float64) *net.Net {
	n := net.CreateNet(layers)

	s := 0.0

	step := 0

	var dataLen = len(trainingData)

	for s < expectedResult {
		for i := 0; i < batchSize; i += 1 {

			r := rand.IntN(dataLen)
			data, expected := trainingData[r], trainingAnswers[r]

			n.Forward(data)
			n.Backward(expected, learningRate)
		}

		s = testing(n, testingData, testingAnswers)
		fmt.Printf("%d%% correct\n", int(s * 100))
		step += 1
	}

	fmt.Printf("%d steps", step)
	return n
}


func testing(n *net.Net, tastingData [][]float64, tastingAnswers [][]float64) float64 {
	right := 0.0

	var dataLen = len(tastingData)

	for i := 0; i < dataLen; i += 1 {
		data, expected := tastingData[i], tastingAnswers[i]

		res := n.Forward(data)


		maxValueRes := slices.Max(res)
		maxIndexRes := slices.IndexFunc(res, func(v float64) bool {
			return v == maxValueRes
		})

		maxValueExpected := slices.Max(expected)
		maxIndexExpected := slices.IndexFunc(expected, func(v float64) bool {
	        return v == maxValueExpected
	    })

		if maxIndexRes == maxIndexExpected {
			right += 1
		}
	}

	return right / float64(dataLen)
}
