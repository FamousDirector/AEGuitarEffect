# Introduction
This repository is trying to create a unique audio effect for guitar using a trained autoencoder.
The goal is to train the autoencoder with a sample played from the guitar.
Using that trained network, incoming audio will processed by the network and hopefully a useful musical effect will be applied.

# Getting started

Use the `record.py` script to capture sound from your computer's default audio input source.
Use the `train.py` script to train on the captured audio files.
Use the `test.py` script to generate audio based on an input audio file.