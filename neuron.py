import numpy as np
import math


class Connection:
	def __init__(self, appendedNeuron):
		#TODO: adjustable matrix size
		self.weight = np.random.random((5, 5))
		self.dWeight = np.zeros((5, 5))
		self.appendedNeuron = appendedNeuron


# learningRate: effects step size of the dendron  updating its weight
# momentumRate: effects how many cycles neuron will continue undating in a specific direction
class Neuron:
	learningRate = 0.001
	momentumRate = 0.01

	# dendrons: list of connections going forward in the neural net (away from initial image)
	# matrix: current extrapolation from image
	# error: update amount for matrix
	# gradient: the direction + amount to alter the weight of a connection
	def __init__(self, neurons=None):
		self.dendrons = []
		self.matrix = []
		self.error = []
	
		self.gradient = 0.0

		if neurons != None
			for neuron in neurons:
				connection = Connection(neuron)
				self.dendrons.append(connection)


	def getContent(self):
		return self.matrix

	def setContent(self, image):
		self.matrix = image

	def setError(self, error):
		self.error = error

	def getError(self):
		return self.error

	def addError(self, error):
		self.error = self.error + error

	# rectified linear unit activation function: everything below 0 is truncated
	def reLU(self, x):
		return x[x < 0] = 0

	# joins all of the matrices from the previous layer, then calculates the new content
	def forwardPropagation:
		numDendrons = len(self.dendrons)
		if numDendrons != 0:
			[inputWidth, inputHeight] = self.dendrons[0].appendedNeuron.getContent().shape
			[weightWidth, weightHeight] = self.dendrons[num].weight.size
			newWidth = floor(inputWidth / weightWidth)
			newHeight = floor(inputHeight / weightHeight)

			newMatrix = np.zeros((newWidth, newHeight))

			for num in range(numDendrons):
				weight = self.dendrons[num].weight
				matrix = self.dendrons[num].appendedNeuron.getContent()

				#TODO: adjustable step size
				for h in range(0, weightHeight, newHeight):
					for w in range(0, weightWidth, newWidth):
						segment = matrix[h : h + weightHeight, w : w + weightWidth]
						segmentValue = np.sum(segment * weight)
						newMatrix[h, w] = newMatrix[h, w] + segmentValue / numDendrons

			self.setContent(self.reLU(self.newMatrix))
			self.setError(np.zeros(newMatrix.size))

	# adjusting weights and errors
	def backPropagation:
		self.gradient = self.error * self.matrix
		[curW, curH] = self.matrix.size

		for dendron in self.dendrons:
			[weightWidth, weightHeight] = dendron.weight.size
			dError = np.zeros(dendron.getError.size)
			for h in range(curH):
				for w in range(curW):
					dendron.dWeight = Neuron.learningRate * 
						(dendron.appendedNeuron.output[h : h + weightHeight, w : w + weightWidth] * self.gradient[h, w]) + 
						Neuron.momentumRate * dendron.dWeight
					dError[h : h + weightHeight, w : w + weightWidth] += dendron.weight*self.gradient[h, w]

            dendron.weight += dendron.dWeight
            dendron.appendedNeuron.addError(dError)