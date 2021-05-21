import os
import torch
import torchaudio

input_size = 10000

input_directory = "testing_data/"
output_directory = "generated_sounds/"

model = torch.load("saved_models/model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()
model.to(device)


for file in os.listdir(input_directory):
    if file.endswith(".wav"):
        waveform, sample_rate = torchaudio.load(os.path.join(input_directory, file))
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
        output_filename = f'{output_directory}/{file.split(".")[0]}.wav'
        new_waveform = new_waveform.to('cpu')
        torchaudio.save(output_filename, new_waveform, sample_rate)
