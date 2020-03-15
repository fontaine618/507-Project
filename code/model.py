class Model:

	def __init__(self, **kwargs):
		self.options = kwargs
		self._prepare_model(**kwargs)
		self.df = None
		self.response = None
		self.features = None
		self.train_accuracy = None
		self.test_accuracy = None

	def _prepare_model(self, **kwargs):
		pass

	def add_train_data(self, df, response, features):
		self.df = df
		self.response = response
		self.features = features

	def train(self):
		pass

	def test(self, test_df):
		pass