from .model import Model
import numpy as np
from scipy.sparse.linalg import svds
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    mean_absolute_error, mean_squared_error
from datetime import datetime



class SVD(Model):

    def _prepare_model(self):
        if "embed_dim" not in self.options:
            self.options['embed_dim'] = 20
        if "iter_nums" not in self.options:
            self.options['iter_nums'] = 20


    def add_train_data(self, df, response = "rating", features = ["user_id", "movie_id"]):
        self.df = df
        self.response = response
        self.features = features
        feature1 = features[0]
        feature2 = features[1]

        train_df = df.pivot(index=feature1, columns=feature2, values=response).fillna(0)
        train_matrix = train_df.values
        pred_matrix = train_matrix
        mask = (train_matrix == 0)

        for _ in range(self.options['iter_nums']):
            ## using visible data repeatedly
            pred_matrix = pred_matrix * mask + train_matrix

            ## project onto low-rank matrices sub-space
            U, sigma, Vt = svds(pred_matrix, k=self.options['embed_dim'])
            pred_matrix = U @ np.diag(sigma) @ Vt

        self.pred_df = pd.DataFrame(pred_matrix, index=train_df.index, columns=train_df.columns)

    def train(self):
        ## create a column of df contains rating predictions

        self.df['pred_rating'] = '0'

        for i in range(self.df.shape[0]):
            self.df.at[self.df.index[i], 'pred_rating'] = self.pred_df.at[self.df.at[self.df.index[i], 'user_id'], self.df.at[self.df.index[i], 'movie_id']]


        y_train = self.df[self.response]
        y_train_pred = self.df['pred_rating']

        #y_train_pred_class = y_train_pred.round().clip(1, 5).astype(int)
        y_train_pred_class = y_train_pred.apply(round)

        # compute metrics
        acc = accuracy_score(y_train, y_train_pred_class)
        prec = precision_score(y_train, y_train_pred_class, average="weighted", zero_division=0)
        rec = recall_score(y_train, y_train_pred_class, average="weighted", zero_division=0)
        mae = mean_absolute_error(y_train, y_train_pred)
        mse = mean_squared_error(y_train, y_train_pred)
        class_mse = mean_squared_error(y_train, y_train_pred_class)
        self.metrics.update({
            "train_accuracy": acc,
            "train_precision": prec,
            "train_recall": rec,
            "train_mae": mae,
            "train_mse": mse,
            "train_class_mse" : class_mse
        })

    def test(self, test_df):
        y_test = test_df[self.response]
        test_df['pred_rating'] = '0'

        for i in range(test_df.shape[0]):
            if (test_df.at[test_df.index[i], "user_id"] in self.pred_df.index) and (test_df.at[test_df.index[i], "movie_id"] in self.pred_df.columns):
                test_df.at[test_df.index[i], "pred_rating"] = self.pred_df.at[
                    test_df.at[test_df.index[i], "user_id"], test_df.at[test_df.index[i], "movie_id"]]
            else:
                test_df.at[test_df.index[i], "pred_rating"] = 3

        y_test_pred = test_df['pred_rating']
        y_test_pred_class = y_test_pred.apply(round)

        # compute metrics
        acc = accuracy_score(y_test, y_test_pred_class)
        prec = precision_score(y_test, y_test_pred_class, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_test_pred_class, average="weighted", zero_division=0)
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        class_mse = mean_squared_error(y_test, y_test_pred_class)
        self.metrics.update({
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_mae": mae,
            "test_mse": mse,
            "test_class_mse": class_mse
        })

    def log(self, path="models/log/svd.tsv"):
        with open(path, "a") as log_file:
            entries = [
                str(datetime.now()),
                "SVD ({})".format(", ".join(["{}: {}".format(k, v) for k, v in self.options.items()])),
            ]
            entries.extend([str(val) for val in self.metrics.values()])
            log_file.write(
                "\n" + "\t".join(entries)
            )

    def cv(self):
        metrics = pd.DataFrame(columns=[
            "cv_accuracy",
            "cv_precision",
            "cv_recall",
            "cv_mae",
            "cv_mse",
            "cv_class_mse"
        ])
        # with Pool(5) as pool:
        #	cv_metrics = pool.map(self._do_one_fold, range(1, 6))
        cv_metrics = [self._do_one_fold(i) for i in range(1, 6)]
        for i, res in enumerate(cv_metrics):
            metrics.loc[i] = list(res.values())
        self.metrics.update(metrics.mean().transpose().to_dict())

    def _do_one_fold(self, i):
        fit = SVD(**self.options)
        fit.add_train_data(self.df[self.df["fold_id"] != i], self.response, self.features)
        # fit.train()
        fit.test(self.df[self.df["fold_id"] == i])
        cv_metrics = {
            "cv_accuracy": fit.metrics['test_accuracy'],
            "cv_precision": fit.metrics['test_precision'],
            "cv_recall": fit.metrics['test_recall'],
            "cv_mae": fit.metrics['test_mae'],
            "cv_mse": fit.metrics['test_mse'],
            "cv_class_mse": fit.metrics['test_class_mse']
        }
        return cv_metrics