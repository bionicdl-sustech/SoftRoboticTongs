import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import numpy as np
import matplotlib.pyplot as plt
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
net.load_state_dict(torch.load('model/rawDataModel.pth'))
net.eval()
net.to(device)

tag_pose_vecs = np.load("data/testPose.npy")
force_vecs = np.load('data/testForce.npy')

x = torch.from_numpy(tag_pose_vecs[:, :]).float()
y = torch.from_numpy(force_vecs[:, :]).float()
x, y = Variable(x), Variable(y)
time0 = time.time()
prediction = net(x.to(device))
y_pred = prediction.detach().cpu().numpy()
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}


#change the i in 0-5 for plotting fx fy fz mx my mz
i = 2


# Fx
if i == 0:
    plt.figure(figsize=(11.5, 11.5), dpi=115)
    plt.plot([-1.8, 0.8], [-1.8, 0.8],linestyle='--',color="black")
    plt.scatter(y[:,i:i+1], y_pred[:,i:i+1])
    plt.xlim(-2, 1)
    plt.ylim(-2, 1)
    plt.xlabel("Fx Ground Truth(N)",font)
    plt.ylabel("Fx Prediction(N)",font)
    plt.tick_params(labelsize=20)
    plt.show()

# Fy
if i == 1:
    plt.figure(figsize=(11.5, 11.5), dpi=115)
    plt.plot([-18, 1], [-18, 1],linestyle='--',color="black")
    plt.scatter(y[:,i:i+1], y_pred[:,i:i+1])
    plt.xlim(-20, 3)
    plt.ylim(-20, 3)
    plt.xlabel("Fy Ground Truth(N)",font)
    plt.ylabel("Fy Prediction(N)",font)
    plt.tick_params(labelsize=20)
    plt.show()

# Fz
if i == 2:
    plt.figure(figsize=(11.5, 11.5), dpi=115)
    plt.plot([-9, 3], [-9, 3],linestyle='--',color="black")
    plt.scatter(y[:,i:i+1], y_pred[:,i:i+1])
    plt.xlim(-10, 4)
    plt.ylim(-10, 4)
    plt.xlabel("Fz Ground Truth(N)",font)
    plt.ylabel("Fz Prediction(N)",font)
    plt.tick_params(labelsize=20)
    plt.show()

# mx
if i == 3:
    plt.figure(figsize=(11.5, 11.5), dpi=115)
    plt.plot([-0.3, 1.8], [-0.3, 1.8],linestyle='--',color="black")
    plt.scatter(y[:,i:i+1], y_pred[:,i:i+1])
    plt.xlim(-0.5, 2)
    plt.ylim(-0.5, 2)
    plt.xlabel("Mx Ground Truth(Nm)",font)
    plt.ylabel("Mx Prediction(Nm)",font)
    plt.tick_params(labelsize=20)
    plt.show()

# my
if i == 4:
    plt.figure(figsize=(11.5, 11.5), dpi=115)
    plt.plot([-0.18, 0.13], [-0.18, 0.13],linestyle='--',color="black")
    plt.scatter(y[:,i:i+1], y_pred[:,i:i+1])
    plt.xlim(-0.2, 0.15)
    plt.ylim(-0.2, 0.15)
    plt.xlabel("My Ground Truth(Nm)",font)
    plt.ylabel("My Prediction(Nm)",font)
    plt.tick_params(labelsize=20)
    plt.show()

# mz
if i == 5:
    plt.figure(figsize=(11.5, 11.5), dpi=115)
    plt.plot([-0.07, 0.07], [-0.07, 0.07],linestyle='--',color="black")
    plt.scatter(y[:,i:i+1], y_pred[:,i:i+1])
    plt.xlim(-0.08, 0.08)
    plt.ylim(-0.08, 0.08)
    plt.xlabel("Mz Ground Truth(Nm)",font)
    plt.ylabel("Mz Prediction(Nm)",font)
    plt.tick_params(labelsize=20)
    plt.show()







