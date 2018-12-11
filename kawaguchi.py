import torch
import random
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
        self.fc1 = nn.Linear(28*28, 512)
        # self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.fc1_bn = nn.LayerNorm(512)
        self.fc2_bn = nn.LayerNorm(128)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        # x = self.fc3(x)
        return self.fc3(x) # last layer activation function is identity y=x

    def evaluateZ(self, x): # right now, x is the activation on the last layer
        # print(x.shape)
        x = self.forward(x)
        z = x.data.numpy()
        # z = x.detach().numpy()
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
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                          # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=False)

    interval = 500

    model = MLP()
    loss_fn = torch.nn.MSELoss(reduction='sum')
    p = 0

    for epoch in range(0,3):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # if batch_idx > 5:
              #  break
            model.train()
            target = encode(target)

            optimizer = optim.Adam(model.parameters(), lr=0.0005)

            # use this one to get bad test errors:
            # optimizer = optim.Adam(model.parameters(), lr = 0.1)

            output = model(data)
            loss = loss_fn(output,target)
            loss.backward()
            optimizer.step()
            if batch_idx % interval == 0:
                # print("Output from model: {}".format(output))
                # print("Output from target: {}".format(target))
                model.eval()
                print("Loss at batch {}, epoch {}: {}".format(batch_idx, epoch, loss.item()))

        # print(output)
        model.eval()

        val_loss, correct = 0, 0

        testLoss = []  # Training eigen values vs. loss
        testEigenVals = []

        for data, target in validation_loader: # Validation error

            output = model(data)

            pred = output.data.max(1)[1]  # get the index of the max probability

            eigen = model.evaluateZ(data) # sum of eigen values of the output vector zT x z

            # print("Eigen: %f" % eigen)
            # print("Loss: %f" % float(loss.item()))

            if pred == target.item():
                correct += pred.eq(target.data).data.sum().item()
            # val_loss = loss_fn(output,target)
            encodedTarget = encode(target)
            val_loss = loss_fn(output,encodedTarget)

            testLoss.append(val_loss.item())
            testEigenVals.append(eigen)

        # val_loss /= len(validation_loader)

        accuracy = 100. * correct / len(validation_loader.dataset)

        # plot loss (x) vs. G eigenvalue sums (y):
        print(max(testLoss))
        print(max(testEigenVals)) # why are there so many fewer when high loss?
        plt.scatter(testLoss, testEigenVals)
        plt.title('Eigenvalues vs. Loss on test set at end of epoch {}'.format(epoch))
        plt.xlabel('Test Loss')
        plt.ylabel('Eigenvalues of h.T*h') # h = activation vector of last layer
        plt.show()
        plt.gcf().clear()
        print("\n\n********EPOCH********* {} done".format(epoch))
        print("Accuracy: %f %%\n\n" % accuracy)

    parameters = []

    '''
    for param in model.parameters():
        if(len(param.size()) == 2):
            print(type(param.data), param.size())
            parameters.append(param.data)
    '''

plt.close('all')

