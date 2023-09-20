import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

tag_pose_vecs = np.load("data/trainPose.npy")
force_vecs = np.load('data/trainForce.npy')
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
pthName = "model/rawDataModel.pth"
for i in range(5):
    if i > 0:
        pthName = "model/rawDataModel" + str(i-1) + ".pth"
        checkpoint = torch.load(pthName)
        net.load_state_dict(checkpoint)

    learningRate *= 0.1
    optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    BATCH_SIZE = 4096
    EPOCH = 100000
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
    pthName = "model/rawDataModel" + str(i) +".pth"
    torch.save(net.state_dict(), pthName )
    string = "model/rawDatalosses" + str(i) +".npy"
    np.save(string, losses)
    # plt.plot(losses)
    # plt.show()