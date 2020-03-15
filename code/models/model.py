import numpy as np

class Model:
	"""

	Attributes
	----------
	options: options to initialize the model
	df: training dataframe
	responses: name of the column of the responses
	features: list of the column names of the features
	train_accuracy: the accuracy upon training
	test_accuracy: the accuracy of the model on the test set
	model: the actual model

	Methods
	-------
	train: Trains the model and sets the training accuracy
	test: Tests the model on a new df and sets the test accuracy
	"""

	def __init__(self, **kwargs):
		self.options = kwargs
		self.df = None
		self.response = None
		self.features = None
		self.train_accuracy = None
		self.test_accuracy = None
		self.model = None
		self.scaler = None
		self.X_train = None
		self.y_train = None
		self.y_train_pred = None
		self._prepare_model()

	def _prepare_model(self):
		pass

	def add_train_data(self, df, response, features, **kwargs):
		pass

	def train(self):
		pass

	def _set_train_accuracy(self):
		self.train_accuracy = np.sqrt(np.mean((self.y_train - self.y_train_pred)**2))

	def _set_test_accuracy(self):
		self.test_accuracy = np.sqrt(np.mean((self.y_test - self.y_test_pred)**2))

	def test(self, test_df):
		pass

	def log(self, path, **kwargs):
		pass