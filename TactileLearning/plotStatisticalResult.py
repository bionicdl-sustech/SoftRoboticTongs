import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import math
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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu:0")

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
from sklearn import metrics
import numpy as np

mses =[0,0,0,0,0,0]
rmses = [0,0,0,0,0,0]
maes = [0,0,0,0,0,0]


for i in range(6):
    mse = metrics.mean_squared_error(y[i], y_pred[i])
    mses[i] = mse
    rmse = np.sqrt(mse)
    rmses[i] = rmse

for i in range(6):
    mae = metrics.mean_absolute_error(y[i], y_pred[i])
    maes[i] = mae

print("mses:",mses)
print("rmses",rmses)
print("maes",maes)

import matplotlib.pyplot as plt
plt.subplot(311)
x = [1,2,3,4,5,6]
x_label=['Fx','Fy','Fz','Mx','My','Mz']
plt.xticks(x, x_label)
plt.bar(x, mses)

plt.title("Mean Square Error")
plt.subplot(312)
x_label=['Fx','Fy','Fz','Mx','My','Mz']
plt.xticks(x, x_label)
plt.xlabel("Tactile Info")
plt.bar(x, rmses)
plt.title("Root Mean Square Error")

plt.subplot(313)
x_label=['Fx','Fy','Fz','Mx','My','Mz']
plt.xticks(x, x_label)
plt.xlabel("Tactile Info")
plt.bar(x, maes)
plt.title("Mean Absolute Error")
plt.show()
