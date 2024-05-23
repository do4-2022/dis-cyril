import numpy as np
import torch
import typing

from transformers import (  # type: ignore
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from discyril.recorder import RATE as SAMPLE_RATE

SAMPLE_SIZE = 1024


class S2T:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)  # type: ignore
        self.processor: Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained(self.model_name)  # type: ignore

    def transcribe(
        self, frame: typing.Iterable[np.ndarray[np.int16, typing.Any]]
    ) -> str:
        # Convert the numpy array to a PyTorch tensor
        audio_tensor = torch.tensor(frame, dtype=torch.float32)

        # Reshape the tensor to match the format expected by the model: [batch_size, num_channels, num_samples]
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(
            0
        )  # Reshape to [1, 1, num_samples]

        # Normalize the tensor values to be in the range [-1.0, 1.0]
        audio_tensor = audio_tensor / (2**15)

        # Convert back the tensor to a numpy array
        speech = audio_tensor.squeeze(0).squeeze(0).numpy()

        # use the model to transcribe the speech
        features = self.processor(
            speech, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt"
        )

        # perform the inference
        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)
        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)

        # return the transcription
        return self.processor.batch_decode(pred_ids)[0]
