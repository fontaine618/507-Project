import torch
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    mean_absolute_error, mean_squared_error
import data

train, test, features = data.load_train_test_and_feature_list()

RANDOM_SEED = 123
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 512
response = "rating"
lr = 0.01
NUM_EPOCHS = 50
type = "Regressor"
n_classes = 5
layers = [(1000, "relu"), (100, "relu")]


class RatingsDataset(Dataset):
    """Custom Dataset for loading Ratings"""

    def __init__(self, df, response, features):
        self.x = df[features].to_numpy()
        self.y = df[response].to_numpy().reshape((-1, 1))

    def __getitem__(self, item):
        return self.x[item, :], self.y[item, :]

    def __len__(self):
        return len(self.y)


train_dataset = RatingsDataset(train[train["fold_id"] != 5], response, features)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

valid_dataset = RatingsDataset(train[train["fold_id"] == 5], response, features)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

test_dataset = RatingsDataset(test, response, features)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

torch.manual_seed(0)

# test everything works
num_epochs = 2
for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        print('Epoch:', epoch + 1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        print('break minibatch for-loop')
        break


class MLP(torch.nn.Module):

    def __init__(
            self,
            num_features,
            layers,
            type=type,
            n_classes=5
    ):
        super(MLP, self).__init__()
        self.type = type
        self.n_classes = n_classes
        self.n_layers = len(layers)
        self.n_input = [num_features] + [n for n, _ in layers]
        self.n_output = [n for n, _ in layers] + [1 if type == "Regressor" else self.n_classes]
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
                else:
                    pass

        return out


torch.manual_seed(RANDOM_SEED)
model = MLP(
    train_dataset.x.shape[1],
    layers,
    type
)
model = model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


# TODO branch on type, cross_entropy
def compute_epoch_loss(model, data_loader):
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).float()
            preds = model(features.float())
            num_examples += targets.size(0)
            curr_loss += F.mse_loss(preds, targets, reduction='sum')

        curr_loss = curr_loss / num_examples
        return curr_loss


start_time = time.time()
minibatch_cost = []
epoch_cost = []

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(DEVICE).float()
        targets = targets.to(DEVICE).float()

        # FORWARD AND BACK PROP
        preds = model(features.float())
        cost = F.mse_loss(preds, targets)
        optimizer.zero_grad()
        cost.backward()
        minibatch_cost.append(cost)

        # UPDATE MODEL PARAMETERS
        optimizer.step()

        # LOGGING
        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                  % (epoch + 1, NUM_EPOCHS, batch_idx,
                     len(train_loader), cost))

    model.eval()
    cost = compute_epoch_loss(model, train_loader)
    cost_valid = compute_epoch_loss(model, valid_loader)
    epoch_cost.append(cost)

    print('Epoch: %03d/%03d Train Cost: %.4f Valid Cost: %.4f' % (
        epoch + 1, NUM_EPOCHS, cost, cost_valid))
    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

plt.plot(range(len(minibatch_cost)), minibatch_cost)
plt.ylabel('MSE')
plt.xlabel('Minibatch')
plt.show()

plt.plot(range(len(epoch_cost)), epoch_cost)
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.show()

print('Test loss: %.4f' % compute_epoch_loss(model, test_loader))


def compute_metrics(model, data_loader):
    if type == "Regressor":
        preds = torch.tensor([[]], device=DEVICE).view((0, 1)).float()
        ys = torch.tensor([[]], device=DEVICE).view((0, 1)).float()
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(DEVICE).float()
                targets = targets.to(DEVICE).float()
                ys = torch.cat([ys, targets], dim=0)
                preds = torch.cat([preds, model(features)], dim=0)
        preds = preds.cpu().numpy()
        ys = ys.cpu().numpy()
    else:
        preds = torch.tensor([[]], device=DEVICE).view((0, n_classes)).float()
        ys = torch.tensor([[]], device=DEVICE).view((0, 1)).float()
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(DEVICE).float()
                targets = targets.to(DEVICE).float()
                ys = torch.cat([ys, targets], dim=0)
                preds = torch.cat([preds, model(features)], dim=0)
        preds_class = preds.cpu().numpy().argmax(axis=1).reshape((-1, 1)) + 1
        preds = preds_class
        ys = ys.cpu().numpy()
    preds_class = preds.round().clip(1, 5).astype(int)
    acc = accuracy_score(ys, preds_class)
    prec = precision_score(ys, preds_class, average="weighted", zero_division=0)
    rec = recall_score(ys, preds_class, average="weighted", zero_division=0)
    mae = mean_absolute_error(ys, preds)
    mse = mean_squared_error(ys, preds)
    return acc, prec, rec, mae, mse


compute_metrics(model, test_loader)
