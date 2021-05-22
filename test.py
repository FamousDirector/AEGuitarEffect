import os
import argparse
import torch
import torchaudio

from spectral_transform import SpectralTransform


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--sample-length', help='Number of values in sample', type=int, default=500)
parser.add_argument('-i', '--input-data-path', help='Where the .wav files to train are',
                    type=str, default="testing_data/")
parser.add_argument('-o', '--output-data-path', help='Where to save the trained model',
                    type=str, default="generated_sounds/")
parser.add_argument('--use-spectral', action='store_true', help='To use spectral input to encode or not')
args = parser.parse_args()

input_size = args.sample_length

model = torch.load("saved_models/model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()
model.to(device)

for file in os.listdir(args.input_data_path):
    if file.endswith(".wav"):
        waveform, sample_rate = torchaudio.load(os.path.join(args.input_data_path, file))
        waveform = waveform[0]  # mono channel

        waveform_chunks = []

        if args.use_spectral:
            st = SpectralTransform(sample_rate)

        i = 0
        while len(waveform) > (i + input_size):
            w = waveform[None, None, i:i + input_size]
            if args.use_spectral:
                real_spectral, img_spectral = st.transform(w)
                real_spectral = real_spectral.to(device)
                recon_spectral = model(real_spectral).cpu()
                o = st.inverse_transform(recon_spectral, img_spectral)
            else:
                w = w.to(device)
                o = model(w)

            waveform_chunks.append(o.flatten())
            i += input_size

        new_waveform = torch.cat(waveform_chunks).unsqueeze(0)

        # Save the generated data as a WAV file
        output_filename = f'{args.output_data_path}/{file.split(".")[0]}_generated.wav'
        new_waveform = new_waveform.to('cpu')
        torchaudio.save(output_filename, new_waveform, sample_rate)
