import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

def testSameS(x,y):
    if x>0 and y >0:
        return True
    elif x<0 and y<0:
        return True
    elif x == 0 and y==0:
        return True
    else:
        return False


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
# device = torch.device("cpu:0")

net = Net()
net.load_state_dict(torch.load('../model/scalerDataModel.pth'))
net.eval()
net.to(device)

tag_pose_vecs = np.load("../data/trainPose.npy")
force_vecs = np.load('../data/trainForce.npy')
# tag_pose_vecs = np.load("testPose.npy")
# force_vecs = np.load('testForce.npy')
maxForce = [0,0,0,0,0,0]
maxPose = [0,0,0,0,0,0]

for i in range(6):
    maxPose[i] = max(tag_pose_vecs[:,i]) if abs(max(tag_pose_vecs[:,i]))>abs(min(tag_pose_vecs[:,i])) else abs(min(tag_pose_vecs[:,i]))
    maxForce[i] = max(force_vecs[:,i]) if abs(max(force_vecs[:,i]))>abs(min(force_vecs[:,i])) else abs(min(force_vecs[:,i]))

for i in range(6):
    for j in range(len(force_vecs)):
        tag_pose_vecs[j][i] /= maxPose[i]
        force_vecs[j][i] /= maxForce[i]

tag_pose_vecs = np.load("../data/testPose.npy")
force_vecs = np.load('../data/testForce.npy')

for i in range(6):
    for j in range(len(force_vecs)):
        tag_pose_vecs[j][i] /= maxPose[i]
        force_vecs[j][i] /= maxForce[i]

first = 300
end= 300
x = torch.from_numpy(tag_pose_vecs[first :first+end, :]).float()
y = torch.from_numpy(force_vecs[first:first+end, :]).float()
x, y = Variable(x), Variable(y)
time0 = time.time()
prediction = net(x.to(device))
y_pred = prediction.detach().cpu().numpy()
fig, axes = plt.subplots(6,1,figsize=(10, 12), dpi=300)

for i in range(6):
    axes[i].plot(y_pred[:,i],'r',linewidth =1)
    axes[i].plot(y[:,i],'k',linewidth =0.6)


plt.savefig("testResult.png")
plt.show()