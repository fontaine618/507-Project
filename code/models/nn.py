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

    def __getitem__(self, item):
        return self.x[item, :], self.y[item, :]

    def __len__(self):
        return len(self.y)


class MLP(torch.nn.Module):
    """Custom neural network class"""

    def __init__(
            self,
            num_features,
            layers,
            type,
            n_classes=5
    ):
        super(MLP, self).__init__()
        self.type = type
        self.n_classes = n_classes
        self.n_layers = len(layers)
        self.n_input = [num_features] + [n for n, _ in layers]
        self.n_output = [n for n, _ in layers] + [1 if self.type.startswith("Regressor") else self.n_classes]
        self.functions = [f for _, f in layers] + ["output"]
        self.layers = torch.nn.ModuleList()
        for n_in, n_out in zip(self.n_input, self.n_output):
            self.layers.append(torch.nn.Linear(n_in, n_out))

    def forward(self, x):
        out = x
        for layer, f in zip(self.layers, self.functions):
            out = layer(out)
            if f == "relu":
                out = torch.relu(out)
            elif f == "sigmoid":
                out = torch.sigmoid(out)
            elif f == "output":
                if self.type == "Classifier":
                    out = F.softmax(out, 1)
                elif self.type == "RegressorSigmoid":
                    out = torch.sigmoid(out) * 10. - 2.
                else:
                    # Regressor and Ordinal
                    pass

        return out


class NN(Model):

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
        model = MLP(
            self.train_dataset.x.shape[1],
            self.options["layers"],
            self.options["type"]
        )
        self.model = model.to(self.options["device"])
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.options["lr"])

    def _compute_epoch_loss(self, data_loader):
        curr_loss, num_examples = 0., 0
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.options["device"]).float()
                targets = targets.to(self.options["device"]).float()
                preds = self.model(features.float())
                num_examples += targets.size(0)
                if self.options["type"] in ["Regressor", "RegressorSigmoid"]:
                    curr_loss += F.mse_loss(preds, targets, reduction='sum')
                elif self.options["type"] == "Classifier":
                    targets_01 = torch.cuda.FloatTensor(targets.size(0), self.model.n_classes).zero_()
                    targets_01 = targets_01.scatter_(1, targets.long()-1, 1)
                    curr_loss += F.binary_cross_entropy(preds, targets_01, reduction='sum')
                elif self.options["type"] == "Ordinal":
                    targets_01 = torch.cuda.FloatTensor(targets.size(0), self.model.n_classes).zero_()
                    for k in range(self.model.n_classes):
                        targets_01 = targets_01.scatter_(
                            1, torch.max(targets.long()-1-k, torch.zeros_like(targets.long())), 1
                        )
                    curr_loss += F.binary_cross_entropy_with_logits(preds, targets_01, reduction='sum')
                else:
                    raise NotImplementedError("type {} not available".format(self.options["type"]))

            curr_loss = curr_loss / num_examples
            return curr_loss

    def _compute_metrics(self, data_loader):
        DEVICE = self.options["device"]
        if self.options["type"] in ["Regressor", "RegressorSigmoid"]:
            preds = torch.tensor([[]], device=DEVICE).view((0, 1)).float()
            ys = torch.tensor([[]], device=DEVICE).view((0, 1)).float()
            with torch.no_grad():
                for features, targets in data_loader:
                    features = features.to(DEVICE).float()
                    targets = targets.to(DEVICE).float()
                    ys = torch.cat([ys, targets], dim=0)
                    preds = torch.cat([preds, self.model(features)], dim=0)
            preds = preds.cpu().numpy()
            ys = ys.cpu().numpy()
            preds_class = preds.round().clip(1, 5).astype(int)
        elif self.options["type"] == "Classifier":
            preds = torch.tensor([[]], device=DEVICE).view((0, self.model.n_classes)).float()
            ys = torch.tensor([[]], device=DEVICE).view((0, 1)).float()
            with torch.no_grad():
                for features, targets in data_loader:
                    features = features.to(DEVICE).float()
                    targets = targets.to(DEVICE).float()
                    ys = torch.cat([ys, targets], dim=0)
                    preds = torch.cat([preds, self.model(features)], dim=0)
            preds_class = preds.cpu().numpy().argmax(axis=1).reshape((-1, 1)) + 1
            preds = preds_class
            ys = ys.cpu().numpy()
        elif self.options["type"] == "Ordinal":
            preds = torch.tensor([[]], device=DEVICE).view((0, self.model.n_classes)).float()
            ys = torch.tensor([[]], device=DEVICE).view((0, 1)).float()
            with torch.no_grad():
                for features, targets in data_loader:
                    features = features.to(DEVICE).float()
                    targets = targets.to(DEVICE).float()
                    ys = torch.cat([ys, targets], dim=0)
                    preds = torch.cat([preds, self.model(features)], dim=0)
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
            for batch_idx, (features, targets) in enumerate(self.train_loader):
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)

                # FORWARD AND BACK PROP
                preds = self.model(features.float())
                if self.model.type in ["Regressor", "RegressorSigmoid"]:
                    cost = F.mse_loss(preds, targets.float())
                elif self.model.type == "Classifier":
                    targets_01 = torch.cuda.FloatTensor(targets.size(0), self.model.n_classes).zero_()
                    targets_01 = targets_01.scatter_(1, targets.long() - 1, 1)
                    cost = F.binary_cross_entropy(preds, targets_01)
                elif self.model.type == "Ordinal":
                    targets_01 = torch.cuda.FloatTensor(targets.size(0), self.model.n_classes).zero_()
                    for k in range(self.model.n_classes):
                        targets_01 = targets_01.scatter_(
                            1,  torch.max(targets.long() - 1 - k, torch.zeros_like(targets)), 1
                        )
                    cost = F.binary_cross_entropy_with_logits(preds, targets_01)
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
                "NN ({})".format(", ".join(["{}: {}".format(k, v) for k, v in self.options.items()])),
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
        fit = NN(**self.options)
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
