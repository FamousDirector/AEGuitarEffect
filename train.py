import torch
import torch.nn as nn

from model import Autoencoder
from dataset import AudioSignalDataset


def train(model, num_epochs=5, batch_size=32, learning_rate=0.01, sample_length=500):
    # set random seed
    torch.manual_seed(42)

    # set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss().to(device)  # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)

    # setup data
    dataset = AudioSignalDataset("input_data/", sample_length=sample_length, sample_shift=1000)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    outputs = []
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
    sample_length = 10000
    model = Autoencoder(sample_length)
    max_epochs = 1
    model = train(model, num_epochs=max_epochs, sample_length=sample_length)
    torch.save(model, "saved_models/model.pth")
