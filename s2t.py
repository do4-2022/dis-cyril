import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from recorder import rate as sampling_rate

model_name = "Ilyes/wav2vec2-large-xlsr-53-french"


class S2T:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def transcribe(self, audio_tensor):
        speech = audio_tensor.squeeze(0).squeeze(0).numpy()

        # use the model to transcribe the speech
        features = self.processor(
            speech, sampling_rate=sampling_rate, padding=True, return_tensors="pt"
        )

        # perform the inference
        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)
        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)

        # return the transcription
        return self.processor.batch_decode(pred_ids)[0]
