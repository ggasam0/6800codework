"""
Class for a classification algorithm.
"""

import numpy as np

class Classifier:

	def __init__(self, classifier_type, **kwargs):
		"""
		Initializer. Classifier_type should be a string which refers
		to the specific algorithm the current classifier is using.
		Use keyword arguments to store parameters
		specific to the algorithm being used. E.g. if you were 
		making a neural net with 30 input nodes, hidden layer with
		10 units, and 3 output nodes your initalization might look
		something like this:

		neural_net = Classifier(weights = [], num_input=30, num_hidden=10, num_output=3)

		Here I have the weight matrices being stored in a list called weights (initially empty).
		"""
		self.classifier_type = classifier_type
		self.params = kwargs
		"""
		The kwargs you inputted just becomes a dictionary, so we can save
		that dictionary to be used in other methods.
		"""


	def train(self, training_data):
		"""
		Data should be nx(m+1) numpy matrix where n is the 
		number of examples and m is the number of features
		(recall that the first element of the vector is the label).

		I recommend implementing the specific algorithms in a
		seperate module and then determining which method to call
		based on classifier_type. E.g. if you had a module called
		neural_nets:

		if self.classifier_type == 'neural_net':
			import neural_nets
			neural_nets.train_neural_net(self.params, training_data)

		Note that your training algorithms should be modifying the parameters
		so make sure that your methods are actually modifying self.params

		You should print the accuracy, precision, and recall on the training data.
		"""

	def predict(self, data):
		"""
		Predict class of a single data vector
		Data should be 1x(m+1) numpy matrix where m is the number of features
		(recall that the first element of the vector is the label).

		I recommend implementing the specific algorithms in a
		seperate module and then determining which method to call
		based on classifier_type.

		This method should return the predicted label.
		"""

	def test(self, test_data):
		"""
		Data should be nx(m+1) numpy matrix where n is the 
		number of examples and m is the number of features
		(recall that the first element of the vector is the label).

		You should print the accuracy, precision, and recall on the test data.
		"""
