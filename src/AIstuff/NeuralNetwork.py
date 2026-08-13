import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Create a model class that inherits nn.Module
class Model(nn.Module):

    # Input layer (4 features of the flower) --> Hidden Layer1 (number of neurons) --> H2 (n) --> Output (3 classes of iris flowers)
    def __init__(self, in_features = 4, h1 = 8, h2 = 9, out_features = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

# Pick a manual seed for randomization
torch.manual_seed(73)
model = Model()

# Load and prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size = 0.2, random_state = 73
)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).long()
y_test = torch.FloatTensor(y_test).long()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

# Evaluation
print("\n--- Evaluating Model ---")
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    correct = (predicted == y_test).sum().item()

accuracy = (correct / len(y_test)) * 100
print(f"Final Test Accuracy: {accuracy:.2f}%\n")

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

# Replace classes by int numbers directly
my_df['species'] = my_df['species'].replace({
    'setosa': 0, 
    'versicolor': 1, 
    'virginica': 2
})

# Extract dataframe data into numpy arrays
X_np = my_df.drop('species', axis=1).values
y_np = my_df['species'].values.astype(int)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=73)

# Convert to pyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the model
epochs = 100
losses = []

for i in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    losses.append(loss.item())

    if i % 10 == 0:
        print(f"Epoch: {i} and Loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Graph it out
plt.plot(range(epochs), losses)
plt.ylabel('loss/error')
plt.xlabel('Epoch')

plt.show()