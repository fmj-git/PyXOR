import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import dataset as ds

X = torch.from_numpy(ds.x_ready)
Y = torch.from_numpy(ds.y_ready)
x_train = X[:150, :]
y_train = Y[:150, :]
x_test = X[150:, :]
y_test = Y[150:, :]

class XOR(nn.Module):

    def __init__(self, inputs=2, outputs=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(inputs, 2)
        self.lin2 = nn.Linear(2, outputs)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        return x


def weights_init(_model):
    for m in _model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 1)

def train(model, criterion, optimizer, x, y, epochs):
    all_loss = []

    for epoch in range(epochs):
        y_hat = model(x)

        loss = criterion(y_hat, y)
        all_loss.append(loss.item())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return all_loss


model = XOR()
weights_init(model)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

all_loss = train(model, criterion, optimizer, x_train, y_train, 50000)

y_pred = model.forward(x_test)
plt.scatter(y_pred.detach().numpy(), y_test)
plt.show()

plt.plot(all_loss)
plt.show()
