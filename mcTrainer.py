import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Define the RNN architecture
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Define the Markov chain class
class MarkovChain:
    def __init__(self, rnn, data):
        self.rnn = rnn
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, lr=0.001, epochs=100, batch_size=32):
        # Prepare the data
        seq_len = self.rnn.rnn.num_layers
        X = []
        Y = []
        for i in range(len(self.data) - seq_len):
            X.append(torch.tensor(self.data[i:i+seq_len]).float().unsqueeze(0))
            Y.append(torch.tensor(self.data[i+seq_len]).float().unsqueeze(0))
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)

        # Move the data to the device
        X = X.to(self.device)
        Y = Y.to(self.device)

        # Instantiate the ADAM optimizer and the MSE loss function
        optimizer = optim.Adam(self.rnn.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Train the model
        for epoch in range(epochs):
            self.rnn.train()
            for i in range(0, len(X), batch_size):
                optimizer.zero_grad()
                batch_x = X[i:i+batch_size].view(-1, seq_len, 1)  # reshape the input tensor
                batch_y = Y[i:i+batch_size]
                output = self.rnn(batch_x)
                loss = criterion(output.squeeze(), Y[i:i+batch_size])
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, epochs, loss.item()))

    def predict(self, x, n_steps=1):
        self.rnn.eval()
        x_test = torch.tensor(x).float().unsqueeze(0).unsqueeze(2)
        x_test = x_test.to(self.device)
        with torch.no_grad():
            y_pred = []
            for i in range(n_steps):
                output = self.rnn(x_test)
                y_pred.append(math.ceil(output.item()))
                x_test = torch.cat([x_test[:, 1:, :], output.unsqueeze(2)], dim=1)
        return y_pred
    
    def score(self, plot=False):
        # Make predictions
        y_true = torch.tensor(self.data[self.rnn.rnn.num_layers:]).to(self.device)
        y_pred = []
        mape_batch = []
        for i in range(0, len(self.data) - self.rnn.rnn.num_layers, 7):
            x = self.data[i:i+self.rnn.rnn.num_layers]
            x = torch.tensor(x).float().unsqueeze(0).unsqueeze(2).to(self.device)
            with torch.no_grad():
                output = self.rnn(x)
                y_pred.append(int(output.item()))
                mape_batch.append(torch.abs((y_true[i:i+7] - output.squeeze()) / y_true[i:i+7]) * 100)
        y_pred = torch.tensor(y_pred).to(self.device)
        mape_batch = torch.cat(mape_batch)
    
        # Calculate MAPE for each batch
        mape_batch = torch.abs((y_true[:len(y_pred)] - y_pred) / y_true[:len(y_pred)]) * 100

        # Calculate mean MAPE across all batches
        mape = torch.mean(mape_batch)

        # Print the metrics
        print(f"MAPE: {mape.item():.2f}%")

        # Plot the time series
        if plot:
            y_pred_all = y_pred.repeat(7).cpu().numpy()
            y_true_all = self.data[self.rnn.rnn.num_layers:]
            plt.plot(y_true_all, label="Actual")
            plt.plot(y_pred_all, label="Predicted")
            plt.legend()
            plt.show()
                
        return mape.item()
    
def train_model(data, hidden_layers, batch_size, n_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn = RNN(7, hidden_layers, 1).to(device)
    markov_chain = MarkovChain(rnn, data)
    markov_chain.train(lr=0.001, epochs=n_epochs, batch_size=batch_size)
    return markov_chain