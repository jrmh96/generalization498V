import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.fc1 = nn.Linear(28 * 28, 50)
    self.fc1_drop = nn.Dropout(0.2)
    self.fc2 = nn.Linear(50, 50)
    self.fc2_drop = nn.Dropout(0.2)
    self.fc3 = nn.Linear(50, 10)

  def forward(self, x):
    x = x.view(-1, 28 * 28)
    x = F.relu(self.fc1(x))
    x = self.fc1_drop(x)
    x = F.relu(self.fc2(x))
    x = self.fc2_drop(x)
    x = self.fc3(x)
    return F.log_softmax(x, dim=1)

batch_size = 32
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

model = Network()

# input data shape: (1, 28, 28)

for i in range(0, 2): # 11 epochs
  optimizer = optim.Adam(model.parameters(), lr=0.0001)
  model.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    # print(list(target.size()))
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()

  model.eval()
  val_loss, correct = 0, 0
  accuracy_vect = [0] * 10
  total_vect = [0] * 10

  for data, target in validation_loader:
    output = model(data)

    val_loss += F.nll_loss(output, target).item()
    pred = output.data.max(1)[1]  # get the index of the max log-probability
    if pred == target.item():
      accuracy_vect[pred] += 1
    correct += pred.eq(target.data).cpu().sum()
    total_vect[target] += 1

  val_loss /= len(validation_loader)
  accuracy = 100. * correct / len(validation_loader.dataset)

  accuracy_vect = torch.FloatTensor(accuracy_vect)
  total_vect = torch.FloatTensor(total_vect)

  percentage_acc_vect = accuracy_vect / total_vect
  print("Current training accuracy: {}" .format(percentage_acc_vect))

print("Final Accuracy: {}".format(percentage_acc_vect)) # Generalization error after one training episode

# Compute Kawaguchi's eq. on the final learned net

# set up v, wBar[], z[], eig[], u[], G

summary(model, (1, 28, 28))

dy = 10

i = 0

for param in model.parameters():
  if i > 20:
    break
  print(type(param.data), param.size())
  i+=1

# z=w_i

'''
ksum = 0
for k in range(1, dy+1):
  wbar = wBar[k-1]
  wbarNorm = torch.norm(wbar)
  vNorm = torch.norm(v)
  eigSum = 0
  for j in range(0, eig.size):
    eigSum+=eig[j]*cos(angle(u[j], wbar))**2
  ksum += 2*vNorm*wbarNorm*cos(angle(wbar, v))
  ksum += (wbarNorm**2) * eigSum

'''
# Compare Final Acc. to keq
