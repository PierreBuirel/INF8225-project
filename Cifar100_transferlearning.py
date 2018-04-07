import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import torch.utils.data as data_utils

# train_data = torchvision.datasets.CIFAR100('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
# test_data = torchvision.datasets.CIFAR100('../data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
#
# chosen_labels=np.array([40,41,42,43,44,55,56,57,58,59,60,61,62,63,64,75,76,77,78,79,80,81,82,83,84,90,91,92,93,94,95,96,97,98,99])
#
# dataset=train_data
# dim_train=0
# labels_train = [dataset[0][1]]
# images_train = [dataset[0][0].numpy()]
# for indice_batch, data_temp in enumerate(dataset):
#     print(indice_batch)
#     if (data_temp[1] in chosen_labels):
#         labels_train = np.append(labels_train,data_temp[1])
#         images_train = np.append(images_train,[data_temp[0].numpy()],axis=0)
#         dim_train=dim_train+1
# np.save('images_train.npy', images_train)
# np.save('labels_train.npy', labels_train)
#
# dataset=test_data
# dim_test=0
# labels_test = [dataset[0][1]]
# images_test = [dataset[0][0].numpy()]
# for indice_batch, data_temp in enumerate(dataset):
#     print(indice_batch)
#     if (data_temp[1] in chosen_labels):
#         labels_test = np.append(labels_test,data_temp[1])
#         images_test = np.append(images_test,[data_temp[0].numpy()],axis=0)
#         dim_test=dim_test+1
# np.save('images_test.npy', images_test)
# np.save('labels_test.npy', labels_test)

images_train = np.load('images_train.npy')
labels_train = np.load('labels_train.npy')
labels_train = torch.from_numpy(labels_train)
images_train = torch.from_numpy(images_train)
customdataset_train=data_utils.TensorDataset(images_train,labels_train)

images_test = np.load('images_test.npy')
labels_test = np.load('labels_test.npy')
labels_test = torch.from_numpy(labels_test)
images_test = torch.from_numpy(images_test)
customdataset_test=data_utils.TensorDataset(images_test,labels_test)

batch_size = 100
test_batch_size = 100

train_loader = torch.utils.data.DataLoader(customdataset_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(customdataset_test, batch_size=batch_size, shuffle=True)

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
        self.fc = nn.Linear(256*4*4, 10)

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
model.load_state_dict(torch.load('mytraining.pt'))
model.fc = nn.Linear(256*4*4, 100)
model.cuda()

# ct = 0
# for child in model.children():
#     model.children.__sizeof__()
#     ct += 1
#     if ct < 4:
#         for param in child.parameters():
#             param.requires_grad = False
#             import pdb ; pdb.set_trace()


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001,weight_decay=0.0005)
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

    train_loss /= 17500
    print('\n' + 'Train' + ' set {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch, train_loss, correct, 17500, 100. * correct / 17500))


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

epochs = 100

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
# http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130 (Classification des differentes performances sur cifar10)
# http://cs231n.github.io/convolutional-networks/ (cours de Stanford sur les CNN)
# https://machinelearningmastery.com/transfer-learning-for-deep-learning/ (A gentle introduction to transfer learning)
# http://ruder.io/transfer-learning/

#Articles
# http://cs231n.stanford.edu/reports/2016/pdfs/001_Report.pdf
# https://arxiv.org/pdf/1411.1792.pdf (How  transferable  are  features  in  deep neural networks?)
# http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7486599 (Batchnorm / Dropout)


#torch.save(model.state_dict(), 'mytraining2.pt')
#model.save_state_dict('mytraining.pt')

#import pdb ; pdb.set_trace()
