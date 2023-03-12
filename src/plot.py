import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler


def plot_results(model, train=None, validation=None, scaler=None):
    # Settings
    threshold = 2.5 # set from histogram_train_losses

    # Load all objects
    model, train, validation, scaler = load_all(model, train, validation, scaler)
    
    # Export training loss distribution
    train_pred, train_losses = predict(model, train)
    export_plot(sns.distplot(train_losses, bins=30, kde=False), "graphs/distribution_train_losses.png")
    
    # Export validation loss distribution
    validation_pred, validation_losses = predict(model, validation)
    export_plot(sns.distplot(validation_losses, bins=30, kde=False), "graphs/distribution_validation_losses.png")
    
    #print(validation_pred)

    # Capture all anomalies in a DataFrame for easy plotting
    seq_len = len(train[0])
    validation_orig = pd.read_csv("data/validation_orig.csv")
    anomalies = pd.DataFrame(validation_orig[seq_len:])
    anomalies['MAE_validation'] = validation_losses
    anomalies['threshold'] = threshold
    anomalies['anomaly'] = anomalies['MAE_validation'] > anomalies['threshold']
    anomalies['Close'] = validation_orig[seq_len:]['Close']

    # Plot MAE with threshold
    _, ax = plt.subplots()
    sns.lineplot(x=anomalies['Date'], y=anomalies['MAE_validation'], ax=ax),
    sns.lineplot(x=anomalies['Date'], y=anomalies['threshold'], ax=ax),
    export_plot(ax, "graphs/MAE_threshold")


    # Plot anomalies on original validation data
    anomaly = anomalies.loc[anomalies['anomaly'] == True]
    _, ax = plt.subplots()
    sns.lineplot(x=anomalies['Date'], y=scaler.inverse_transform(anomalies['Close'].to_numpy().reshape(-1, 1)).flatten(), ax=ax)
    sns.scatterplot(x=anomaly['Date'], y=scaler.inverse_transform(anomaly['Close'].to_numpy().reshape(-1, 1)).flatten(), color='r', ax=ax)
    export_plot(ax, "graphs/anomolies.png")

def export_plot(plot, path):
    fig = plot.get_figure()
    fig.savefig(path)
    plt.close(fig)

def predict(model, X):
    predictions, losses = [], []
    with torch.no_grad():
        for x in X:
            pred = model(x.squeeze())
            loss = np.mean(np.abs(pred.numpy() - x.numpy()))
            predictions.append(pred.numpy().flatten())
            losses.append(loss.item())
        predictions = np.stack(predictions, axis=0)
    return predictions, losses

def load_all(model=None, train=None, validation=None, scaler=None):
    if model==None:
        PATH = "models/model.pth"
        model.load_state_dict(torch.load(PATH))

    if train==None or validation==None:
        with open('data\Preprocessed.npy', 'rb') as f:
            train = np.load(f)
            validation = np.load(f)

        train = torch.from_numpy(train)
        validation = torch.from_numpy(validation)

    if scaler==None:    
        scaler = StandardScaler()
        with open('utils\standard_scaler.pickle', 'rb') as f:
            scaler = pickle.load(f)


    return model, train, validation, scaler