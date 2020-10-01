import numpy as np


class FullyConnected:

	def __init__(self, inputSize):
		
		#Init weigts and biases to rand nums
		self.weights = np.random.randn(inputSize, 2) / inputSize #2 for smile, not-smile
		self.biases = np.zeros(2)

	def forward(self, input):
		
		#Save input shape for backprop and flatten
		self.inputshape = input.shape
		self.input = input.flatten()	#Save for backprop calculus
		
		#Adjust input using weights and biases
		self.result = np.dot(self.input, self.weights) + self.biases
		
		#Softmax activation function
		self.numerator = np.exp(self.result)
		self.denominator = np.sum(self.numerator, axis = 0)	#Sums the results along the column
		
		#print("Numerator  : ", self.numerator)
		#print("Denominator: ", self.denominator)
		
		return self.numerator/self.denominator
		
		
	def backward(self, dLoss_dOut, correct, learnRate):
		
		#Calculate gradient of output (SoftMax function) with respect to result 
		sum = self.denominator
		e_Res = self.numerator
		dOut_dRes = -e_Res[correct] * e_Res / (sum * sum)
		dOut_dRes[correct] = e_Res[correct] * (sum - e_Res[correct]) / (sum * sum)
		#print("dOut_dRes", dOut_dRes)
		
		#Gradient of result with respect to weights, biases and input
		#Using result = w*input + biases
		dRes_dw = self.input
		dRes_db = 1
		dRes_din = self.weights
		
		#Gradient of loss with respect to result
		dLoss_dRes = dLoss_dOut * dOut_dRes
		
		#Gradient of loss with respect to weights, bises and input
		dLoss_dw = dRes_dw[np.newaxis].T @ dLoss_dRes[np.newaxis]
		dLoss_db = dLoss_dRes * dRes_db
		#print("weights Delta:", dLoss_dw)
		dLoss_din = dRes_din @ dLoss_dRes			#Matrix multiplication
		
		#Update weights and biases
		self.weights -= (learnRate) * dLoss_dw
		self.biases -= (learnRate) * dLoss_db
		
		#Return loss with respect to the input in original shape to backpropagate
		return dLoss_din.reshape(self.inputshape)	
		
