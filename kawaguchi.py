import torch
import random
import numpy as np
import torch.nn as nn
from numpy import linalg as LA
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
        self.fc1_bn = nn.LayerNorm(10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        #x = self.fc1_bn(x)
        #x = F.relu(x)
        output = x
        return output

    def evaluateZ(self, x):
        x = self.forward(x)


        z = x.data.numpy()
        G = np.dot(z.T,z)
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
    cords = zip(ranged, targets_np)

    empty = torch.zeros((batch_size,num_classes))

    empty[ranged, targets_np] = 1

    return empty
if __name__ == '__main__':
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=False)



    interval = 10000

    model = MLP()
    loss_fn = torch.nn.MSELoss(reduction='sum')

    for i in range(0,10):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            target = encode(target)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            output = model(data)
            loss = loss_fn(output,target)
            loss.backward()
            optimizer.step()
            if batch_idx % interval == 0:
                model.eval()
                print("Eigen: %f" % model.evaluateZ(data))
                print("Loss: %f" % float(loss.item()))

        model.eval()
        val_loss, correct = 0, 0
        for data, target in validation_loader:
            output = model(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            if pred == target.item():
                correct += pred.eq(target.data).data.sum().item()

        val_loss /= len(validation_loader)
        accuracy = 100. * correct / len(validation_loader.dataset)
        print(accuracy)
