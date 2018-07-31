import math
import numpy as np

from neuron import Neuron


#TODO: implement decreasing step sizes as training continues
class NeuralNet:
	def __init__(self, neuronsCount, netconfig):
		self.layers = []
		for neuronCount in neuronsCount:
			layer = []
			for i in range(neuronCount):
				if not i:
					layer.append(Neuron(config))
				else:
					layer.append(Neuron(config, self.layers[i-1]))
			layer.append(Neuron) #bias neuron, same size as weight matrix
			layer[neuronCount].matrix = np.ones((5,5))
			self.layers.append(layer)

	def inputImages(self, images):
		if len(self.layers[0]) != len(images):
			raise ValueError('missmatched lengths in neuralNetwork and input')
		#TODO: add check for expected size of matrix, depending on stepsize+layers
		#  so final output is 1x1

		for i in range(len(images)):
			self.layers[0][i].matrix = images[i]

	def getError(self, targets):
		if len(self.layers[-1]) != len(targets):
			raise ValueError('missmatched lengths in neuralNetwork and target')

		error = 0
		for i in range(len(targets)):
			error += targets[i] - sum(self.layers[-1][i].matrix)
		return error

	def forwardPropagation(self):
		for layer in self.layers[1:]:
			for neuron in layer:
				neuron.forwardPropagation()

	def backPropagation(self, targets):
		if len(self.layers[-1]) != len(targets):
			raise ValueError('missmatched lengths in neuralNetwork and target')
		#TODO: target matrix not the expected dimensions (is output always gonna be 1x1?)

		for i in range(len(targets)):
			self.layers[-1][i].error = targets[i] - self.layers[-1][i].matrix
		for layer in self.layers[::-1]:
			for neuron in layer:
				neuron.backPropagation()
