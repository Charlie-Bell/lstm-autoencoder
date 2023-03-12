import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import torch
import seaborn as sns
from matplotlib import pyplot as plt


def preprocess(PATH="data/GS.csv", seq_len=15, save=True):
    # Read in data: Close vs. Date.
    dataframe = pd.read_csv(PATH)
    df = dataframe[['Date', 'Close']]
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'])

    # Export lineplot of Close vs Date
    export_graph(df, "graphs/close_vs_date.png") 

    # Split the data
    train, validation = df.loc[df['Date'] <= '2013-07-01'], df.loc[df['Date'] > '2013-07-01']
    
    # Save the original split data
    if save:
        train.to_csv('data/train_orig.csv')
        validation.to_csv('data/validation_orig.csv')

    # Scale the data
    scaler = StandardScaler()
    scaler = scaler.fit(train[['Close']])
    
    # Save the scaler
    if save:
        with open('utils/standard_scaler.pickle', 'wb') as f:
            pickle.dump(scaler, f)

    # Transform the data
    train.loc[:, 'Close'] = scaler.transform(train[['Close']])
    validation.loc[:, 'Close'] = scaler.transform(validation[['Close']])

    # Save the transformed data
    if save:
        train.to_csv('data/train.csv')
        validation.to_csv('data/validation.csv')

    # To numpy sequences
    train = to_sequences(train['Close'], seq_len)
    validation = to_sequences(validation['Close'], seq_len)
    
    # To pytorch tensors
    train = torch.from_numpy(train).unsqueeze(2)
    validation = torch.from_numpy(validation).unsqueeze(2)

    # Export the data
    if save:
        with open('data\Preprocessed.npy', 'wb') as f:
            np.save(f, train.numpy())
            np.save(f, validation.numpy())

    return train, validation

# Convert df to numpy sequences
def to_sequences(x, seq_len=15):
        x_values = [x.iloc[i:(i+seq_len)] for i in range(len(x)-seq_len)]
        
        return np.array(x_values)

# Export lineplots
def export_graph(df, path):
    plot = sns.lineplot(x=df['Date'], y=df['Close'])
    fig = plot.get_figure()
    fig.savefig(path)
    plt.close(fig)