import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device("cuda:0")
# device = torch.device("cpu:0")
print(device)

# tag_pose_vecs = np.load("/home/dong/PycharmProjects/train/old/train_pose.npy")
# force_vecs = np.load('/home/dong/PycharmProjects/train/old/train_force.npy')
tag_pose_vecs = np.load("trainPose.npy")
force_vecs = np.load('trainForce.npy')
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
checkpoint = torch.load("10004.pth")
net.load_state_dict(checkpoint)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

BATCH_SIZE = 256
# EPOCH = 10
# EPOCH = 1000
time0 = time.time()

for EPOCH in [10000]:
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
    torch.save(net.state_dict(), "10005.pth" )
    string = "losses" + str(EPOCH) +".npy"
    np.save(string, losses)

    # plt.plot(losses)
    # plt.show()