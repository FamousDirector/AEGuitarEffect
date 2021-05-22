import argparse
import torch
import torch.nn as nn

from model import Autoencoder
from dataset import AudioSignalDataset


def train(model, input_data_dir, num_epochs=5, batch_size=256, learning_rate=0.001, sample_length=500):
    # set random seed
    torch.manual_seed(42)

    # set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss().to(device)  # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # setup data
    dataset = AudioSignalDataset(input_data_dir, sample_length=sample_length, sample_shift=100)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        for data in train_loader:
            data = data.to(device)
            recon = model(data)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='Number of epochs to train for', type=int, default=3)
    parser.add_argument('-l', '--sample-length', help='Number of values in sample', type=int, default=500)
    parser.add_argument('-i', '--input-data-path', help='Where the .wav files to train are',
                        type=str, default="input_data/")
    parser.add_argument('-o', '--output-save-path', help='Where to save the trained model',
                        type=str, default="saved_models/model.pth")
    args = parser.parse_args()

    model = Autoencoder(args.sample_length)
    model = train(model, args.input_data_path, num_epochs=args.epochs, sample_length=args.sample_length)
    torch.save(model, args.output_save_path)
