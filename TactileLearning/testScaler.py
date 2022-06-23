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
net.load_state_dict(torch.load('trainScaler/2.pth'))
net.eval()
net.to(device)

tag_pose_vecs = np.load("trainPose.npy")
force_vecs = np.load('trainForce.npy')
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

tag_pose_vecs = np.load("testPose.npy")
force_vecs = np.load('testForce.npy')

for i in range(6):
    for j in range(len(force_vecs)):
        tag_pose_vecs[j][i] /= maxPose[i]
        force_vecs[j][i] /= maxForce[i]

first = 1000
end= 300
x = torch.from_numpy(tag_pose_vecs[first :first+end, :]).float()
y = torch.from_numpy(force_vecs[first:first+end, :]).float()
x, y = Variable(x), Variable(y)
time0 = time.time()
prediction = net(x.to(device))
y_pred = prediction.detach().cpu().numpy()
fig, axes = plt.subplots(6,1,figsize=(10, 12), dpi=300)
err = [0,0,0,0,0,0]
# num = [0,0,0,0,0,0]
# 3% 5% 10% 20% 30% ,,  positive?
# 10% 20% 30% 40% 50% ,,  positive?
time1 = time.time()

print(time1-time0)
print("fps:",14417/(time1-time0))
goodPec = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
difPec = [0,0,0,0,0,0]

for i in range(6):
    axes[i].plot(y_pred[:,i],'r',linewidth =1)
    axes[i].plot(y[:,i],'k',linewidth =0.6)

    # print(y_pred[0][i])
    # print(len(y_pred))
    # print(float(y_pred[448][i]))
    # print(float(y[448][i]))

    for j in range(len(y_pred)):
        if float(y[j][i]) !=0:
            stand = abs((abs(float(y_pred[j][i]))-abs(float(y[j][i]))/abs(float(y[j][i]))))
            if stand<=0.1:
                goodPec[i][0] +=1
            elif stand>0.1 and stand<=0.2:
                goodPec[i][1] +=1
            elif stand>0.2 and stand<=0.3:
                goodPec[i][2] += 1
            elif stand>0.3 and stand<=0.4:
                goodPec[i][3] += 1
            elif stand > 0.4 and stand <= 0.5:
                goodPec[i][4] += 1
            else:
                goodPec[i][5] += 1

            Dif = abs((abs(float(y_pred[j][i]))-abs(float(y[j][i]))))
            difPec[i] += Dif
            if testSameS(float(y_pred[j][i]),float(y[j][i])):
                goodPec[i][6] +=1

            # err[i] = err[i] + abs((float(y_pred[j][i])-float(y[j][i]))/float(y[j][i]))
            # print(err[i])
print(len(y_pred))
print(len(y))

print(goodPec)
#
for i in range(6):
    difPec[i] = difPec[i]/len(y_pred)
    for j in range(7):
        goodPec[i][j]= goodPec[i][j]/len(y_pred)
#
# print(err)
print("dif",difPec)
print(goodPec)

plt.savefig("filename.png")
plt.show()