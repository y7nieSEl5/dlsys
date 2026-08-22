import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    residual_fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    return nn.Sequential(nn.Residual(residual_fn), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    layers = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        layers.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for X, y in dataloader:
        logits = model(X)
        loss = nn.SoftmaxLoss()(logits, y)
        total_loss += loss.numpy() * X.shape[0]
        total_correct += (logits.numpy().argmax(axis=1) == y.numpy()).sum()
        total_samples += X.shape[0]
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return 1 - accuracy, avg_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(image_filename=data_dir + "/train-images-idx3-ubyte.gz", label_filename=data_dir + "/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(image_filename=data_dir + "/t10k-images-idx3-ubyte.gz", label_filename=data_dir + "/t10k-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = MLPResNet(784, hidden_dim=hidden_dim, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for e in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)
        test_error, test_loss = epoch(test_dataloader, model)
    return (train_error, train_loss, test_error, test_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
