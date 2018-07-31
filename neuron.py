import numpy as np
import math


class Connection:
	def __init__(self, appendedNeuron, weightSize):
		#TODO: adjustable matrix size
		self.weight = np.random.random((weightSize, weightSize))
		self.dWeight = np.zeros((weightSize, weightSize))
		self.appendedNeuron = appendedNeuron

# Very Advanced Rectified Linear Unit Activation function: no negative numbers allowed
def reLU(self, mat):
	return np.vectorize(lambda x: 0 if x < 0 else x)(mat)
#TODO: config selects activation function


class Neuron:
	# dendrons: list of Connections going forward in the neural net (away from initial image)
	# matrix: current extrapolation from image
	# error: update amount for matrix
	def __init__(self, config, neurons=[]):
		self.dendrons = []
		self.matrix = None
		self.error = None
		self.config = config

		for neuron in neurons:
			connection = Connection(neuron, self.config.weightSize)
			self.dendrons.append(connection)

	# joins all of the matrices from the previous layer, then calculates the new content
	def forwardPropagation():
		numDendrons = len(self.dendrons)
		if numDendrons != 0:
			[inputWidth, inputHeight] = self.dendrons[0].appendedNeuron.matrix.shape

			newWidth = floor((inputWidth - self.config.weightSize)/ self.config.stepSize)
			newHeight = floor((inputHeight - self.config.weightSize)/ self.config.stepSize)

			newMatrix = np.zeros((newWidth, newHeight))

			for num in range(numDendrons):
				weight = self.dendrons[num].weight
				matrix = self.dendrons[num].appendedNeuron.matrix

				for h in range(0, self.config.stepSize, newHeight):
					for w in range(0, self.config.stepSize, newWidth):
						segment = matrix[h : h + self.config.weightSize, w : w + self.config.weightSize]
						segmentValue = np.sum(segment * weight)
						newMatrix[h, w] = newMatrix[h, w] + segmentValue / numDendrons

			self.matrix = reLU(newMatrix)
			self.error = np.zeros(newMatrix.size)

	# adjusting weights and errors
	def backPropagation():
		# gradient: the direction + amount to alter the weight of a connection
		gradient = self.error * self.matrix
		[currentWidth, currentHeight] = self.matrix.size

		for dendron in self.dendrons:
			[weightWidth, weightHeight] = dendron.weight.size
			dError = np.zeros(dendron.error.size)
			for h in range(currentHeight):
				for w in range(currentWidth):
					dendron.dWeight = (self.config.learningRate *
						(dendron.appendedNeuron.matrix[h : h + weightHeight, w : w + weightWidth] * gradient[h][w]) + 
						self.config.momentumRate * dendron.dWeight)
					dError[h : h + weightHeight, w : w + weightWidth] += dendron.weight*gradient[h][w]

			dendron.weight += dendron.dWeight
			dendron.appendedNeuron.error += dError
