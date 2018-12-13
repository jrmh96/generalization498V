import torch
import numpy as np
import torch.nn as nn
from numpy import linalg as LA
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 320)
        # self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(320, 10)
        # self.fc3 = nn.Linear(128, 10)
        self.fc1_bn = nn.LayerNorm(320)
        # self.fc2_bn = nn.LayerNorm(128)
        self.d = 1

    def forward(self, x):
        x = x.view(-1, 28*28)
        if self.d == 1:
            print(x.size())
            self.d+=1
        x = F.relu(self.fc1_bn(self.fc1(x)))
        # x = F.relu(self.fc2_bn(self.fc2(x)))
        return self.fc2(x) # last layer activation function is identity y=x

    def evaluateZ(self, x):
        x = self.forward(x)
        z = x.data.numpy()
        G = np.dot(z.T, z)  # z is 1xn vector
        values, vectors = LA.eig(G)
        return (np.sum(values,axis=0))

def encode(targets, batch_size=None, num_classes=10):
        """"
        targets - is a pytorch tensor of dim 1xBatchSize
        returns - a matrix containing 1-hot vectors for the targets
        """
        targets_np = targets.data.numpy()
        if batch_size == None:
            batch_size = targets.shape[0]

        targets_np = targets.data.numpy()

        if batch_size == None:
            batch_size = targets.shape[0]

        ranged = range(0, batch_size)
        # cords = zip(ranged, targets_np)

        empty = torch.zeros((batch_size, num_classes))

        empty[ranged, targets_np] = 1

        return empty

def dfs(params, w):
    dfsHelper(params, 0, 0, 0, 1, w)

def dfsHelper(params, depth, row, col, wk, w):

    if(depth == len(params)):
        w.append(wk)
        return

    if(row == len(params[depth])):
        return

    if(col == len(params[depth])):
        dfsHelper(params, depth, row+1, 0, wk, w)

    dfsHelper(params, depth+1, row, col, wk*params[depth][row][col], w)


if __name__ == '__main__':
    batch_size = 32
    dy = 10
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    mnist_val = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    validation_loader = torch.utils.data.DataLoader(
        mnist_val,
        batch_size=1, shuffle=False)

    interval = 500

    model = MLP()
    loss_fn = torch.nn.MSELoss(reduction='sum')

    for epoch in range(0,1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            target = encode(target)

            optimizer = optim.Adam(model.parameters(), lr=0.0005)

            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % interval == 0:
                model.eval()
                print("Loss at batch {}, epoch {}: {}".format(batch_idx, epoch, loss.item()))

            # print(output)
        model.eval()

        val_loss, correct = 0, 0

        for data, target in validation_loader:  # Validation error

            output = model(data)
            pred = output.data.max(1)[1]

            encodedTarget = encode(target)
            val_loss = loss_fn(output, encodedTarget)

            if pred == target.item():
                correct += pred.eq(target.data).data.sum().item()

            accuracy = 100. * correct / len(validation_loader.dataset)

        parameters = []
        for param in model.parameters():
            print(type(param.data), param.size()) # get Weight matrices, note that forward pass uses W^T x
            lstParam = list(param.size())
            if(len(lstParam) == 2):
                # print(param.size())
                parameters.append(torch.transpose(param.data, 0, 1).numpy()) # parameters are in order by network structure

        # compute the weight path vector
        wbar = []
        dfs(parameters, wbar)

        print(len(wbar))


