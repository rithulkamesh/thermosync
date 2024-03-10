import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import math
import flask_cors
from flask import Flask, request, jsonify

data = pd.read_csv('ac_temp.csv')

data['body_temp'] = (data['body_temp'] - 32) * 5/9

data['ac_temp'] = (data['ac_temp'] - 32) * 5/9

scaler = MinMaxScaler()
data['body_temp'] = scaler.fit_transform(
    data['body_temp'].values.reshape(-1, 1))

X = torch.tensor(data['body_temp'].values, dtype=torch.float32).reshape(-1, 1)
y = torch.tensor(data['ac_temp'].values, dtype=torch.float32)


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.l2_reg = 0.01

    def forward(self, x):
        output = self.linear(x)
        l2_norm = sum([param.pow(2).sum()
                      for param in self.linear.parameters()])
        loss = output + self.l2_reg * l2_norm
        return loss


model = LinearRegression()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


app = Flask(__name__)
flask_cors.CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    body_temp_celsius = request.json['body_temp']
    body_temp_normalized = scaler.transform([[body_temp_celsius]])
    ac_temp_celsius = model(torch.tensor(
        body_temp_normalized, dtype=torch.float32))
    ac_temp_fahrenheit = (ac_temp_celsius.item() * 9/5) + 32
    return jsonify({
        "ac_temp": math.ceil(ac_temp_fahrenheit)
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0')
