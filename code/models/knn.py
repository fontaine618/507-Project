from .model import Model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


class KNN(Model):

	def _prepare_model(self):
		if "n_neighbors" not in self.options:
			self.options["n_neighbors"] = 5
		self.model = KNeighborsRegressor(self.options["n_neighbors"])

	def add_train_data(self, df, response, features):
		self.df = df
		self.response = response
		self.features = features
		self.scaler = StandardScaler()
		self.X_train = self.df[self.features]
		self.scaler.fit(self.X_train)
		self.X_train = self.scaler.transform(self.X_train)
		self.y_train = self.df[self.response]

	def train(self):
		self.model.fit(self.X_train, self.y_train)
		self.y_train_pred = self.model.predict(self.X_train)
		self._set_train_accuracy()

	def test(self, test_df):
		self.X_test = test_df[self.features]
		self.X_test = self.scaler.transform(self.X_test)
		self.y_test = test_df[self.response]
		self.y_test_pred = self.model.predict(self.X_test)
		self._set_test_accuracy()

	def log(self, path, description):
		with open(path, "a") as log_file:
			log_file.write(
				"{}\t{}\t{}\n".format(
					self.train_accuracy,
					self.test_accuracy,
					description
				)
			)