import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F


import torchvision.transforms as transforms
from torch import nn
from torch import optim


train_data = torchvision.datasets.CIFAR100('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

test_data = torchvision.datasets.CIFAR100('../data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

batch_size = 100
test_batch_size = 100

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


losses = []
accuracies = []

class FcNetwork(nn.Module):
    def __init__(self):
        super(FcNetwork, self).__init__()

        #Convolutions
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.cnn3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Spatial pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

        #BatchNormalisation
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        #Dropout
        self.drop = nn.Dropout(0.3)
        self.dropFC = nn.Dropout(0.5)

        #Fully-Connected Layer
        self.fc = nn.Linear(256*4*4, 100)

    def forward(self, image):

        #Premiere couche de convolution
        x = self.cnn1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #Deuxieme couche de convolution
        x = self.drop(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #Troisieme couche de convolution
        x = self.drop(x)
        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #Fully-Connected layer
        x = self.dropFC(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x

model = FcNetwork()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0005)
loss_fn = nn.NLLLoss()

def train(epoch):

    model.train()
    correct = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        #data, target = Variable(data), Variable(target)
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        train_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    train_loss /= 50000
    print('\n' + 'Train' + ' set {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch, train_loss, correct, 50000,100. * correct / 50000))


def test(loader, name, epoch):

    model.eval()
    test_loss = 0
    correct = 0

    for data, target in loader:

        #data, target = Variable(data, volatile=True), Variable(target)
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = model(data)
        test_loss += loss_fn(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    print('\n' + name + ' set {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,test_loss, correct, len(loader.dataset),100. * correct / len(loader.dataset)))
    losses.append(test_loss)
    accuracies.append(100. * correct / len(loader.dataset))

epochs = 10

for epoch in range(1, epochs + 1):
    train(epoch)
    test(test_loader, 'Test', epoch)

test(test_loader, 'Test', 1)

if __name__ == '__main__':
    print()

fig1, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(losses,'g-')
ax2.plot(accuracies,'b-')

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color='g')
ax2.set_ylabel("Accuracy", color = 'b')
plt.show()

# References
# http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7486599 (Very Deep Convolutional Neural Network Based Image Classification Using Small Training Sample Size)
# https://arxiv.org/pdf/1412.6806.pdf (Striving for simplicity : the all convolutional net)
# http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130 (Classification des differentes performances sur cifar10)
# http://cs231n.github.io/convolutional-networks/ (cours de Stanford sur les CNN)
# https://arxiv.org/pdf/1411.1792.pdf (How  transferable  are  features  in  deep neural networks?)
# https://machinelearningmastery.com/transfer-learning-for-deep-learning/ (A gentle introduction to transfer learning)
# http://ruder.io/transfer-learning/

#torch.save(model.state_dict(), 'mytraining2.pt')
#model.save_state_dict('mytraining.pt')
