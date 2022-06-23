import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu:0")
print(device)

tag_pose_vecs = np.load("../data/trainPose.npy")
force_vecs = np.load('../data/trainForce.npy')
maxForce = [0,0,0,0,0,0]
maxPose = [0,0,0,0,0,0]

for i in range(6):
    maxPose[i] = max(tag_pose_vecs[:,i]) if abs(max(tag_pose_vecs[:,i]))>abs(min(tag_pose_vecs[:,i])) else abs(min(tag_pose_vecs[:,i]))
    maxForce[i] = max(force_vecs[:,i]) if abs(max(force_vecs[:,i]))>abs(min(force_vecs[:,i])) else abs(min(force_vecs[:,i]))

for i in range(6):
    for j in range(len(force_vecs)):
        tag_pose_vecs[j][i] /= maxPose[i]
        force_vecs[j][i] /= maxForce[i]

x = torch.from_numpy(tag_pose_vecs[:, :]).float()
y = torch.from_numpy(force_vecs[:, :]).float()
x, y = Variable(x), Variable(y)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(6, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 6),
        )

    def forward(self, x):
        y_pred = self.layers(x)
        return y_pred


net = Net()
net.to(device)
learningRate = 1
pthName = "../model/scalerDataModel.pth"
for i in range(5):
    if i > 0:
        pthName = "../model/scalerDataModel" + str(i-1) + ".pth"
        checkpoint = torch.load(pthName)
        net.load_state_dict(checkpoint)

    learningRate *= 0.1
    optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    BATCH_SIZE = 4096*8
    EPOCH = 10000

    time0 = time.time()
    torch_dataset = Data.TensorDataset(x, y)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, )

    losses = []
    for epoch in range(EPOCH):
        running_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            b_x = Variable(batch_x).to(device)
            b_y = Variable(batch_y).to(device)
            prediction = net(b_x)  # input x and predict based on x
            loss = loss_func(prediction, b_y)  # must be (1. nn output, 2. target)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            running_loss += loss.item()
            losses.append(loss.item())

            l = running_loss / 1
            print("Epoch: %s step: %s, loss = " % (epoch, step), l)
            running_loss = 0.0
    time1 = time.time()
    print("time:" , time1-time0)
    pthName = "../model/scalerDataModel" + str(i) +".pth"
    torch.save(net.state_dict(), pthName )
    string = "../model/scalerDatalosses" + str(i) +".npy"
    np.save(string, losses)

    # plt.plot(losses)
    # plt.show()