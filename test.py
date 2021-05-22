import os
import argparse
import torch
import torchaudio


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--sample-length', help='Number of values in sample', type=int, default=500)
parser.add_argument('-i', '--input-data-path', help='Where the .wav files to train are',
                    type=str, default="testing_data/")
parser.add_argument('-o', '--output-data-path', help='Where to save the trained model',
                    type=str, default="generated_sounds/")
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

        i = 0
        while len(waveform) > (i + input_size):
            w = waveform[None, None, i:i + input_size]
            w = w.to(device)
            o = model(w)
            waveform_chunks.append(o.flatten())
            i += input_size

        new_waveform = torch.cat(waveform_chunks).unsqueeze(0)

        # Save the generated data as a WAV file
        output_filename = f'{args.output_data_path}/{file.split(".")[0]}_generated.wav'
        new_waveform = new_waveform.to('cpu')
        torchaudio.save(output_filename, new_waveform, sample_rate)
