import torch.nn as nn
import torch
import time
import numpy as np
import pandas as pd
from datetime import datetime

class Encoder(nn.Module):
    def __init__(self, seq_len, input_size, embedding_size=64):
        super(Encoder, self).__init__()

        '''
        Parameters
            seq_len = The size of the time window i.e. number of timesteps
            input_size = The numbers of features or dimensions
            embedding_size = The desired dimension for the embedding layer
        
        The encoder consists of 2 stacked LSTMs
        Input is [batch_size, seq_len, input_size]
        Output 'x' is [batch_size, seq_len, hidden_size]
        Hidden size 'hidden_n' = [num_layers, batch_size, input_size]
        Cell size 'cell_n' = [num_layers, batch_size, hidden_size]
        '''

        self.seq_len = seq_len
        self.input_size = input_size
        self.embedding_size =  embedding_size
        self.hidden_size = embedding_size*2
        
        self.LSTM1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=self.hidden_size,hidden_size=self.embedding_size, batch_first=True)
        self.dr = nn.Dropout(0.2)
    
    def forward(self, x):
        # [30, 1]
        x = x.reshape((1, self.seq_len, self.input_size))
        # [1, 30, 1]
        x, (hidden_n, cell_n) = self.LSTM1(x)
        x = self.dr(x)
        # [1, 30, 256]
        #x, (hidden_n, cell_n) = self.LSTM2(x)
        # [1, 30, 128], [1, 1, 128]
        #hidden_n = hidden_n.reshape((self.input_size, self.embedding_size))
        hidden_n = hidden_n.reshape((self.input_size, self.hidden_size))
        # [1, 128]
        return hidden_n
    

class Decoder(nn.Module):
    def __init__(self, seq_len, input_size=1, embedding_size=128):
        super(Decoder, self).__init__()
        
        '''
        Parameters
            seq_len = The size of the time window i.e. number of timesteps
            input_size = The numbers of features or dimensions
            embedding_size = The desired dimension for the embedding layer
        
        The decoder consists of 2 stacked LSTMs
        Input is [batch_size, seq_len, input_size]
        Output 'x' is [batch_size, seq_len, hidden_size]
        Hidden size 'hidden_n' = [num_layers, batch_size, input_size]
        Cell size 'cell_n' = [num_layers, batch_size, hidden_size]
        '''

        self.seq_len = seq_len
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size*2
        
        self.LSTM1 = nn.LSTM(input_size=self.embedding_size, hidden_size=self.embedding_size, batch_first=True)
        #self.LSTM2 = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.dr = nn.Dropout(0.2)
        self.output_layer = nn.Linear(self.hidden_size, self.input_size)
    
    def forward(self, x):
        # Repeat vector bridging the Encoder-Decoder
        # [1, 128]
        x = x.repeat(self.seq_len, self.input_size) # (seq_len, batch_size)
        # [30, 128]
        #x = x.reshape((self.input_size, self.seq_len, self.embedding_size))
        x = x.reshape((self.input_size, self.seq_len, self.hidden_size))
        # [1, 30, 128]
        #x, (hidden_n, cell_n) = self.LSTM1(x)
        # [1, 30, 128]
        x, (hidden_n, cell_n) = self.LSTM2(x)
        x = self.dr(x)
        # [1, 30, 256]
        x = x.reshape((self.seq_len, self.hidden_size))
        # [30, 256]
        x = self.output_layer(x)
        # [30, 1]
        return x
 

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, input_size, embedding_size=128, batch_size=1, device='cpu'):
        super(LSTMAutoencoder, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self.encoder = Encoder(seq_len, input_size, embedding_size).to(self.device)
        self.decoder = Decoder(seq_len, batch_size, embedding_size).to(self.device)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x.squeeze()
  
    def train_model(self, X_train, X_validation, n_epochs, save=True, model_name="model"):

        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-12)
        criterion = nn.L1Loss(reduction='sum').to(self.device)
        history = {'train':[], 'validation':[]}

        training_iterations = (X_train.shape[0])
        validation_iterations = X_validation.shape[0]
        print(f"Training iterations: {training_iterations}. Validation iterations: {validation_iterations}")

        for epoch in range(1, n_epochs+1):
            
            ts = time.time()
            
            # Training Loop
            self = self.train()
            train_losses = []
            for i in range(training_iterations):
                X_train_batch = X_train[i].to(self.device)
                X_train_batch = X_train_batch.squeeze()

                optimizer.zero_grad()
                output = self.forward(X_train_batch)
                loss = criterion(output, X_train_batch)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                if i % 100 == 0:
                    print(f"Current mean train loss: {np.mean(train_losses):1.5f}")

            te = time.time()

            # Cross-Validation Loop
            validation_losses = []
            with torch.no_grad():
                for i in range(validation_iterations):
                    X_validation_batch = X_validation[i].to(self.device)
                    X_validation_batch = X_validation_batch.squeeze()

                    prediction = self.forward(X_validation_batch)
                    
                    loss = criterion(prediction, X_validation_batch)

                    validation_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            validation_loss = np.mean(validation_losses)

            history['train'].append(train_loss)
            history['validation'].append(validation_loss)
            
            if epoch % 1 == 0:
                print(f"Epoch: {epoch}, train loss: {train_loss:1.5f}, validation loss: {validation_loss:1.5f}, training time: {te-ts:1f}s")

        self = self.eval()

        if save:
            # SAVE MODEL & HISTORY
            history_df = pd.DataFrame(history)
            t = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            history_df.to_csv(f"history/training_{t}.csv", index=False)

            PATH = "models/"+model_name+".pth"
            torch.save(self.state_dict(), PATH)

        return self



