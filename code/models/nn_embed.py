from .model import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    mean_absolute_error, mean_squared_error
from datetime import datetime
import pandas as pd
import torch
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class RatingsDataset(Dataset):
    """Custom Dataset for loading Ratings"""

    def __init__(self, df, response, features):
        self.x = df[features].to_numpy()
        self.y = df[response].to_numpy().reshape((-1, 1))
        self.movie_id = df["movie_id"].to_numpy()
        self.user_id = df["user_id"].to_numpy()

    def __getitem__(self, item):
        return self.x[item, :], self.y[item, :], self.user_id[item], self.movie_id[item]

    def __len__(self):
        return len(self.y)


class MLPEmbed(torch.nn.Module):
    """Custom neural network class with embeddings"""

    def __init__(
            self,
            num_features,
            layers,
            type,
            n_embed_users=50,
            n_embed_movies=100,
            n_classes=5,
            n_users=943 + 1,
            n_movies=1681 + 1,
            use_features=True
    ):
        super(MLPEmbed, self).__init__()
        # embeddings
        self.n_movies = n_movies
        self.n_users = n_users
        self.n_embed_movies = n_embed_movies
        self.n_embed_users = n_embed_users
        self.user_embedding = torch.nn.Embedding(self.n_users, self.n_embed_users)
        self.movie_embedding = torch.nn.Embedding(self.n_movies, self.n_embed_movies)
        # MLP
        self.num_features = num_features
        self.type = type
        self.n_classes = n_classes
        self.n_layers = len(layers)
        self.use_features = use_features
        if self.use_features:
            self.n_input = [self.num_features + self.n_embed_users + self.n_embed_movies] + [n for n, _ in layers]
        else:
            self.n_input = [self.n_embed_users + self.n_embed_movies] + [n for n, _ in layers]
        self.n_output = [n for n, _ in layers] + [1 if self.type.startswith("Regressor") else self.n_classes]
        self.functions = [f for _, f in layers] + ["output"]
        self.layers = torch.nn.ModuleList()
        for n_in, n_out in zip(self.n_input, self.n_output):
            self.layers.append(torch.nn.Linear(n_in, n_out))

    def forward(self, x, user_ids, movie_ids):
        # Embeddings
        u = self.user_embedding(user_ids)
        m = self.movie_embedding(movie_ids)
        # MLP
        if self.use_features:
            out = torch.cat([u, m, x], dim=1)
        else:
            out = torch.cat([u, m], dim=1)
        for layer, f in zip(self.layers, self.functions):
            out = layer(out)
            if f == "relu":
                out = torch.relu(out)
            elif f == "sigmoid":
                out = torch.sigmoid(out)
            elif f == "relu6":
                out = F.relu6(out)
            elif f == "output":
                if self.type == "Classifier":
                    out = F.softmax(out, 1)
                elif self.type == "RegressorSigmoid":
                    out = torch.sigmoid(out) * 10. - 2.
                elif self.type == "RegressorRelu6":
                    out = F.relu6(out)
                else:
                    # Regressor and Ordinal
                    pass

        return out


class NNEmbed(Model):

    def _prepare_model(self):
        if "seed" not in self.options:
            self.options["seed"] = 1
        if "device" not in self.options:
            self.options["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if "batch_size" not in self.options:
            self.options["batch_size"] = 512
        if "lr" not in self.options:
            self.options["lr"] = 0.01
        if "num_epochs" not in self.options:
            self.options["num_epochs"] = 50
        if "type" not in self.options:
            self.options["type"] = "Regressor"
        if "n_classes" not in self.options:
            self.options["n_classes"] = 5
        if "use_features" not in self.options:
            self.options["use_features"] = True
        if "n_embed_users" not in self.options:
            self.options["n_embed_users"] = 50
        if "n_embed_movies" not in self.options:
            self.options["n_embed_movies"] = 100
        if "n_users" not in self.options:
            self.options["n_users"] = 943 + 1
        if "n_movies" not in self.options:
            self.options["n_movies"] = 1681 + 1
        if "weight_decay" not in self.options:
            self.options["weight_decay"] = None
        if "layers" not in self.options:
            self.options["layers"] = [(1000, "relu"), (100, "relu")]

    def add_train_data(self, df, response, features):
        self.df = df
        self.response = response
        self.features = features
        self.train_dataset = RatingsDataset(self.df, self.response, self.features)
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.options["batch_size"],
            shuffle=False,
            num_workers=4
        )
        # Create the NN model
        torch.manual_seed(self.options["seed"])
        model = MLPEmbed(
            self.train_dataset.x.shape[1],
            self.options["layers"],
            self.options["type"],
            n_embed_users=self.options["n_embed_users"],
            n_embed_movies=self.options["n_embed_movies"],
            use_features=self.options["use_features"]
        )
        self.model = model.to(self.options["device"])
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.options["lr"],
            weight_decay=self.options["weight_decay"]
        )

    def _compute_epoch_loss(self, data_loader):
        DEVICE = self.options["device"]
        curr_loss, num_examples = 0., 0
        with torch.no_grad():
            for features, targets, users, movies in data_loader:
                features = features.to(DEVICE).float()
                targets = targets.to(DEVICE).float()
                users = users.to(DEVICE).long()
                movies = movies.to(DEVICE).long()
                preds = self.model(features, users, movies)
                num_examples += targets.size(0)
                if self.options["type"].startswith("Regressor"):
                    curr_loss += F.mse_loss(preds, targets, reduction='sum')
                elif self.options["type"] == "Classifier":
                    targets_01 = torch.cuda.LongTensor(targets.size(0), self.model.n_classes).zero_()
                    targets_01 = targets_01.scatter_(1, targets.long()-1, 1)
                    curr_loss += F.binary_cross_entropy(preds.float(), targets_01.float(), reduction='sum')
                elif self.options["type"] == "Ordinal":
                    targets_01 = torch.cuda.LongTensor(targets.size(0), self.model.n_classes).zero_()
                    for k in range(self.model.n_classes):
                        targets_01 = targets_01.scatter_(
                            1, torch.max(targets.long()-1-k, torch.zeros_like(targets.long())), 1
                        )
                    curr_loss += F.binary_cross_entropy_with_logits(preds.float(), targets_01.float(), reduction='sum')
                else:
                    raise NotImplementedError("type {} not available".format(self.options["type"]))

            curr_loss = curr_loss / num_examples
            return curr_loss

    def _compute_metrics(self, data_loader):
        DEVICE = self.options["device"]
        if self.options["type"].startswith("Regressor"):
            preds = torch.tensor([[]], device=DEVICE).view((0, 1)).float()
            ys = torch.tensor([[]], device=DEVICE).view((0, 1)).float()
            with torch.no_grad():
                for features, targets, users, movies in data_loader:
                    features = features.to(DEVICE).float()
                    targets = targets.to(DEVICE).float()
                    users = users.to(DEVICE).long()
                    movies = movies.to(DEVICE).long()
                    ys = torch.cat([ys, targets], dim=0)
                    preds = torch.cat([preds, self.model(features, users, movies)], dim=0)
            preds = preds.cpu().numpy()
            ys = ys.cpu().numpy()
            preds_class = preds.round().clip(1, 5).astype(int)
        elif self.options["type"] == "Classifier":
            preds = torch.tensor([[]], device=DEVICE).view((0, self.model.n_classes)).float()
            ys = torch.tensor([[]], device=DEVICE).view((0, 1)).float()
            with torch.no_grad():
                for features, targets, users, movies in data_loader:
                    features = features.to(DEVICE).float()
                    targets = targets.to(DEVICE).float()
                    users = users.to(DEVICE).long()
                    movies = movies.to(DEVICE).long()
                    ys = torch.cat([ys, targets], dim=0)
                    preds = torch.cat([preds, self.model(features, users, movies)], dim=0)
            preds_class = preds.cpu().numpy().argmax(axis=1).reshape((-1, 1)) + 1
            preds = preds_class
            ys = ys.cpu().numpy()
        elif self.options["type"] == "Ordinal":
            preds = torch.tensor([[]], device=DEVICE).view((0, self.model.n_classes)).float()
            ys = torch.tensor([[]], device=DEVICE).view((0, 1)).float()
            with torch.no_grad():
                for features, targets, users, movies in data_loader:
                    features = features.to(DEVICE).float()
                    targets = targets.to(DEVICE).float()
                    users = users.to(DEVICE).long()
                    movies = movies.to(DEVICE).long()
                    ys = torch.cat([ys, targets], dim=0)
                    preds = torch.cat([preds, self.model(features, users, movies)], dim=0)
            preds_class = preds.cpu().numpy() > 0.0
            preds_class = preds_class.sum(axis=1).clip(1, 5).reshape((-1, 1))
            preds = preds_class
            ys = ys.cpu().numpy()
        else:
            raise NotImplementedError("type {} not available".format(self.options["type"]))
        acc = accuracy_score(ys, preds_class)
        prec = precision_score(ys, preds_class, average="weighted")
        rec = recall_score(ys, preds_class, average="weighted")
        mae = mean_absolute_error(ys, preds)
        mse = mean_squared_error(ys, preds)
        return acc, prec, rec, mae, mse

    def train(self):
        # Train model
        DEVICE = self.options["device"]
        start_time = time.time()
        minibatch_cost = []
        epoch_cost = []
        for epoch in range(self.options["num_epochs"]):
            self.model.train()
            for batch_idx, (features, targets, users, movies) in enumerate(self.train_loader):
                features = features.to(DEVICE).float()
                targets = targets.to(DEVICE).float()
                users = users.to(DEVICE).long()
                movies = movies.to(DEVICE).long()

                # FORWARD AND BACK PROP
                preds = self.model(features, users, movies)
                if self.model.type.startswith("Regressor"):
                    cost = F.mse_loss(preds, targets)
                elif self.model.type == "Classifier":
                    targets_01 = torch.cuda.LongTensor(targets.size(0), self.model.n_classes).zero_()
                    targets_01 = targets_01.scatter_(1, targets.long() - 1, 1)
                    cost = F.binary_cross_entropy(preds.float(), targets_01.float())
                elif self.model.type == "Ordinal":
                    targets_01 = torch.cuda.LongTensor(targets.size(0), self.model.n_classes).zero_()
                    for k in range(self.model.n_classes):
                        targets_01 = targets_01.scatter_(
                            1,  torch.max(targets.long() - 1 - k, torch.zeros_like(targets).long()), 1
                        )
                    cost = F.binary_cross_entropy_with_logits(preds.float(), targets_01.float())
                else:
                    raise NotImplementedError("type {} not available".format(self.options["type"]))
                self.optimizer.zero_grad()
                cost.backward()
                minibatch_cost.append(cost)


                # UPDATE MODEL PARAMETERS
                self.optimizer.step()

                # LOGGING
                if not batch_idx % 50:
                    print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                          % (epoch + 1, self.options["num_epochs"], batch_idx,
                             len(self.train_loader), cost))

            self.model.eval()
            cost = self._compute_epoch_loss(self.train_loader)
            epoch_cost.append(cost)

            print('Epoch: %03d/%03d Train Cost: %.4f' % (epoch + 1, self.options["num_epochs"], cost))
            print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
        print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
        # Evaluate
        acc, prec, rec, mae, mse = self._compute_metrics(self.train_loader)
        self.metrics.update({
            "train_accuracy": acc,
            "train_precision": prec,
            "train_recall": rec,
            "train_mae": mae,
            "train_mse": mse
        })

    def test(self, test_df):
        test_dataset = RatingsDataset(test_df, self.response, self.features)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.options["batch_size"],
            shuffle=False,
            num_workers=4
        )
        acc, prec, rec, mae, mse = self._compute_metrics(test_loader)
        self.metrics.update({
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_mae": mae,
            "test_mse": mse
        })

    def log(self, path="models/log/nn.tsv"):
        with open(path, "a") as log_file:
            entries = [
                str(datetime.now()),
                "NNEmbed ({})".format(", ".join(["{}: {}".format(k, v) for k, v in self.options.items()])),
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
            "cv_mse"
        ])
        cv_metrics = [self._do_one_fold(i) for i in range(1, 6)]
        for i, res in enumerate(cv_metrics):
            metrics.loc[i] = list(res.values())
        self.metrics.update(metrics.mean().transpose().to_dict())

    def _do_one_fold(self, i):
        fit = NNEmbed(**self.options)
        fit.add_train_data(self.df[self.df["fold_id"] != i], self.response, self.features)
        fit.train()
        fit.test(self.df[self.df["fold_id"] == i])
        cv_metrics = {
            "cv_accuracy": fit.metrics['test_accuracy'],
            "cv_precision": fit.metrics['test_precision'],
            "cv_recall": fit.metrics['test_recall'],
            "cv_mae": fit.metrics['test_mae'],
            "cv_mse": fit.metrics['test_mse']
        }
        return cv_metrics
