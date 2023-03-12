import pandas as pd
import torch

from src.preprocess import preprocess
from src.autoencoder import LSTMAutoencoder
from src.plot import plot_results

if __name__=="__main__":
    # Settings
    pd.options.mode.chained_assignment = None
    training = False
    save = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameters
    seq_len = 15
    model_name = "model"

    # Preprocessing
    train, validation = preprocess("data/GS.csv", seq_len, save)

    # Model training or importing
    model = LSTMAutoencoder(seq_len, input_size=1, embedding_size=64, batch_size=1, device=device).double().to(device)
    if training:
        model = model.train_model(train, validation, n_epochs=2, save=save, model_name=model_name)
    else:
        PATH = "models/"+model_name+".pth"
        model.load_state_dict(torch.load(PATH))

    # Plotting
    plot_results(model)

    print("Success")