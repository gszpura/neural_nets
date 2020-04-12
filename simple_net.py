import numpy as np
from torch import nn, tensor, optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import normalize
from time import perf_counter

train_x = np.array(
    [[1, 2, 3, 4, 5, 6],
     [2, 4, 6, 8, 10, 12],
     [1, 5, 9, 13, 17, 21],
     [7, 17, 27, 37, 47, 57],
     [10, 15, 20, 25, 30, 35],
     [5, 6, 7, 8, 9, 10],
     [8, 10, 12, 14, 16, 18],
     [102, 104, 106, 108, 110, 112],

     [2, 10, 18, 26, 34, 42],
     [17, 18, 19, 20, 21, 22],
     [7, 9, 11, 13, 15, 17],
     [33, 36, 39, 42, 45, 48],
     [100, 200, 300, 400, 500, 600],
     [3, 33, 63, 93, 123, 153],
     [30, 32, 34, 36, 38, 40],
     [2, 5, 8, 11, 14, 17],

     [3, 10, 3, 4, 2, 8],
     [1, 101, 8, 8, 8, 7],
     [1, 2, 3, 4, 5, 1],
     [7, 8, 9, 8, 7, 4],
     [4, 5, 3, 10, 12, 13],
     [5, 5, 6, 6, 8, 9],
     [9, 10, 10, 8, 7, 10],
     [3, 4, 11, 11, 10, 10],

     [1, 1, 2, 2, 3, 3],
     [9, 9, 8, 8, 7, 6],
     [200, 170, 150, 140, 135, 132],
     [5, 5, 5, 5, 5, 5],
     [32, 32, 33, 33, 117, 118],
     [29, 26, 28, 29, 30, 32],
     [22, 12, 23, 9, 10, 14],
     [20, 20, 15, 10, 25, 30]], dtype=np.float32)

train_y = np.array([1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0])

test_x = np.array([[6, 9, 12, 15, 18, 21],
                  [11, 12, 13, 14, 15, 16],
                  [24, 25, 26, 27, 28, 29],
                  [9, 9, 9, 10, 10, 10],
                  [9, 8, 7, 6, 5, 4],
                  [3, 10, 3, 4, 8, 2],
                  [12, 22, 32, 42, 52, 62]], dtype=np.float32)
test_y = [1, 1, 1, 0, 0, 0, 1]


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(6, 256)
        self.output = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


datap = normalize(tensor(train_x))
labels = tensor(train_y)
train_data = TensorDataset(datap, labels)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

epochs = 32000
lr = 0.03

model = Network()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)


def train(model, criterion, optimizer, epochs=1000):
    t1 = perf_counter()
    for e in range(epochs):
        running_loss = 0
        for points, labels in train_loader:
            optimizer.zero_grad()
            output = model(points)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            if e % 100 == 0:
                print(f"Training loss: {running_loss / len(train_loader)}")
    t2 = perf_counter()
    print("Took:", t2 - t1)

print("Training")
train(model, criterion, optimizer, epochs)

print("Evaluating...")
test_normalized = normalize(tensor(test_x))
correct = 0
for no, item in enumerate(test_normalized):
    prob_prediction = model(item.unsqueeze(0))
    predicted_label = np.argmax(prob_prediction.detach().numpy())
    print(predicted_label, test_y[no], test_x[no])
    if (predicted_label == test_y[no]):
        correct += 1
print("Result...")
print(correct / float(len(test_y)))
