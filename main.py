from s2t import S2T
from nlp import NLP
from recorder import Recorder

s2t = S2T()
nlp = NLP()


def process_audio(audio_tensor):
    transcription = s2t.transcribe(audio_tensor)
    intent = nlp.predict_intent(transcription)

    print("Transcription:", transcription)
    print("Intent:", intent)


def bootstrap():
    print("Loading models...")
    s2t.load_model()
    nlp.load_model()

    print("Ready to transcribe and analyze speech")
    recorder = Recorder(process_audio)
    recorder.listen()


if __name__ == "__main__":
    bootstrap()
