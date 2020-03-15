from .model import Model
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
	mean_absolute_error, mean_squared_error
from datetime import datetime

class KNN(Model):

	def _prepare_model(self):
		if "n_neighbors" not in self.options:
			self.options["n_neighbors"] = 5
		if "type" not in self.options:
			self.options["type"] = "Classifier"
		if self.options["type"] == "Classifier":
			self.model = KNeighborsClassifier(self.options["n_neighbors"])
		else:
			self.options["type"] = "Regressor"
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
		if self.options["type"] == "Classifier":
			y_train_pred_class = self.model.predict(self.X_train)
			y_train_pred = y_train_pred_class
		else:
			y_train_pred = self.model.predict(self.X_train)
			y_train_pred_class = y_train_pred.round().clip(1, 5).astype(int)
		# compute metrics
		acc = accuracy_score(self.y_train, y_train_pred_class)
		prec = precision_score(self.y_train, y_train_pred_class, average="weighted")
		rec = recall_score(self.y_train, y_train_pred_class, average="weighted")
		mae = mean_absolute_error(self.y_train, y_train_pred)
		mse = mean_squared_error(self.y_train, y_train_pred)
		self.metrics.update({
			"train_accuracy": acc,
			"train_precision": prec,
			"train_recall": rec,
			"train_mae": mae,
			"train_mse": mse
		})

	def test(self, test_df):
		X_test = test_df[self.features]
		X_test = self.scaler.transform(X_test)
		y_test = test_df[self.response]
		if self.options["type"] == "Classifier":
			y_test_pred_class = self.model.predict(X_test)
			y_test_pred = y_test_pred_class
		else:
			y_test_pred = self.model.predict(X_test)
			y_test_pred_class = y_test_pred.round().clip(1, 5).astype(int)
		# compute metrics
		acc = accuracy_score(y_test, y_test_pred_class)
		prec = precision_score(y_test, y_test_pred_class, average="weighted", zero_division=0)
		rec = recall_score(y_test, y_test_pred_class, average="weighted", zero_division=0)
		mae = mean_absolute_error(y_test, y_test_pred)
		mse = mean_squared_error(y_test, y_test_pred)
		self.metrics.update({
			"test_accuracy": acc,
			"test_precision": prec,
			"test_recall": rec,
			"test_mae": mae,
			"test_mse": mse
		})

	def log(self, path="models/log/knn.tsv"):
		with open(path, "a") as log_file:
			entries = [
				str(datetime.now()),
				"KNN ({})".format(", ".join(["{}: {}".format(k,v) for k, v in self.options.items()])),
			]
			entries.extend([str(val) for val in self.metrics.values()])
			log_file.write(
				"\n" + "\t".join(entries)
			)