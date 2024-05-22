import pyaudio
import math
import struct
import wave
import time
import os
import numpy as np
import torch

threshold = 15
chunk = 1024
swidth = 2
short_normalize = 1.0 / 32768.0
format = pyaudio.paInt16
channels = 1
rate = 16000
record_length = 1


class Recorder:

    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * short_normalize
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self, callback=None):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=format,
            channels=channels,
            rate=rate,
            input=True,
            output=True,
            frames_per_buffer=chunk,
        )
        self.callback = callback

    def record(self):
        print("Noise detected, recording beginning")
        rec = []
        current = time.time()
        end = time.time() + record_length

        # record the audio until the end time
        while current <= end:
            print(end - current)
            data = self.stream.read(chunk)

            if self.rms(data) >= threshold:
                end = time.time() + record_length

            current = time.time()
            rec.append(data)

        # remove the last record_length frames to avoid no sound at the end
        for i in range(1, record_length):
            rec.pop()

        # now, process the recorded frames
        self.process(rec)

    def process(self, frames):
        # Convert the byte data to numpy array
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)

        # Convert the numpy array to a PyTorch tensor
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

        # Reshape the tensor to match the format expected by the model: [batch_size, num_channels, num_samples]
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(
            0
        )  # Reshape to [1, 1, num_samples]

        # Normalize the tensor values to be in the range [-1.0, 1.0]
        audio_tensor = audio_tensor / (2**15)

        if self.callback:
            self.callback(audio_tensor)

    def listen(self):
        print("Listening beginning")
        while True:
            input = self.stream.read(chunk)
            rms_val = self.rms(input)
            if rms_val > threshold:
                self.record()
