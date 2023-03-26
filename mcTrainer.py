import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(DEVICE)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

class MarkovChain:
    def __init__(self, rnn, data):
        self.model = rnn
        self.data = data

    def prepare_data(self, data, window_size):
        x, y = [], []
        for i in range(len(data) - window_size):
            x.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return torch.tensor(x).float(), torch.tensor(y).float()
                    
    def train(self, window_size, epochs, batch_size, learning_rate, show_progress=True):
        x, y = self.prepare_data(self.data, window_size)
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()

        total_steps = len(dataloader) * epochs
        progress_bar = tqdm.tqdm(total=total_steps, desc="Training Progress") if show_progress else None

        for epoch in range(epochs):
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                inputs = inputs.unsqueeze(-1)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets.unsqueeze(-1))
                loss.backward()
                optimizer.step()

                if show_progress:
                    progress_bar.update(1)

        if show_progress:
            progress_bar.close()

    def forecast(self, data, window_size, n_steps):
        self.model.eval()
        with torch.no_grad():
            predictions = []
            for _ in range(n_steps):
                inputs = torch.tensor(data[-window_size:]).float().to(DEVICE)
                inputs = inputs.unsqueeze(0).unsqueeze(-1)
                output = self.model(inputs)
                prediction = output.item()
                predictions.append(prediction)
                data.append(prediction)
            return predictions
        
    def score(self, plot=False):
        # Make predictions
        y_true = torch.tensor(self.data[self.model.rnn.num_layers:]).to(DEVICE)
        y_pred = []
        mape_batch = []
        for i in range(0, len(self.data) - self.model.rnn.num_layers, 7):
            x = self.data[i:i+self.model.rnn.num_layers]
            x = torch.tensor(x).float().unsqueeze(0).unsqueeze(2).to(DEVICE)
            with torch.no_grad():
                output = self.model(x)
                y_pred.append(int(output.item()))
                mape_batch.append(torch.abs((y_true[i:i+7] - output.squeeze()) / y_true[i:i+7]) * 100)
        y_pred = torch.tensor(y_pred).to(DEVICE)
        mape_batch = torch.cat(mape_batch)
    
        # Calculate MAPE for each batch
        mape_batch = torch.abs((y_true[:len(y_pred)] - y_pred) / y_true[:len(y_pred)]) * 100

        # Calculate mean MAPE across all batches
        mape = torch.mean(mape_batch)

        # Print the metrics

        # Plot the time series
        if plot:
            print(f"MAPE: {mape.item():.2f}%")
            y_pred_all = y_pred.repeat(7).cpu().numpy()
            y_true_all = self.data[self.model.rnn.num_layers:]
            plt.plot(y_true_all, label="Actual")
            plt.plot(y_pred_all, label="Predicted")
            plt.legend()
            plt.show()
                
        return mape.item()