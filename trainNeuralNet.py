import pickle

from time import gmtime
import numpy as np
from collections import namedtuple

from NeuralNet import NeuralNet

# learningRate: effects step size of the dendron  updating its weight
# momentumRate: effects how many cycles neuron will continue undating in a specific direction
NetConfig = namedtuple("NetConfig", ["learningRate", "momentumRate", "stepSize", "weightSize"])


def saveNeuralNet(net, fileName=net):
	fileName = fileName + strftime("%Y_%m_%d_%H_%M", gmtime())
	file = open(fileName, 'wb')
	pickle.dump(net, file)
	file.close()

def readNeuralNet(fileName):
	file = open(fileName, 'r')
	net = pickle.load(file)
	file.close()
	return net

# experimenting with whether having some preprocessed inputs quickens convergence
# sequential: every time there is an error, restart from start of training data
# currently hard coded to use gradient decent. Shall add more options
def trainNeuralNetSequential(inputList, target, layers, config):
	if layers[-1] != len(target[0]):
		raise ValueError('missmatched lengths of output and target')
	if layers[0] != len(inputList[0]):
		raise ValueError('missmatched lengths of input and layers')
					
	net = NeuralNet(layers, config)

	for i in range(len(inputList)):
		net.inputImages(inputList[i])
		net.forwardPropagation()
		net.backwardPropagation(target[i])

		error = net.getError(target[i])
		# TODO: this check is so wrong
		if error > 10:
			i = -1

	saveNeuralNet(net)

# non-sequential: iterate over entire training set, restart until satisfactory
def trainNeuralNetNonSequential(inputList, target, layers, config):
	if layers[-1] != len(target[0]):
		raise ValueError('missmatched lengths of output and target')
	if layers[0] != len(inputList[0]):
		raise ValueError('missmatched lengths of input and layers')
	net = NeuralNet(layers, config)

	while True:
		maxError = 0
		for i in range(len(inputList)):
			net.inputImages(inputList[i])
			net.forwardPropagation()
			net.backwardPropagation(target[i])

			error = net.getError(target[i])
			if error > maxError:
				maxError = error
		if maxError < 10:
			break

	saveNeuralNet(net)

# To use with separate test images
ErrorSumTrainedNeuralNet(net, inputList, target):
	error = 0
	for i in range(len(inputList)):
		net.inputImages(inputList[i])
		net.forwardPropagation()
		error += net.getError(target[i])
	return error