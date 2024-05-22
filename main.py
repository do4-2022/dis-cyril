import requests

from s2t import S2T
from nlp import NLP
from t2s import T2S
from recorder import Recorder

s2t = S2T()
nlp = NLP()
t2s = T2S()


def process_audio(audio_tensor):
    transcription = s2t.transcribe(audio_tensor)
    intent = nlp.predict_intent(transcription)

    print("Transcription:", transcription)
    print("Intent:", intent)

    if intent == "weather_montpellier":
        r = requests.get("https://api.tibuzin.do-2021.fr/weather")
        response = r.json()
        t2s.say(response["text"])
    else:
        t2s.say("Je suis désolé, je n'ai pas compris ce que vous avez dit.")


def bootstrap():
    print("Loading models...")
    s2t.load_model()
    nlp.load_model()

    print("Ready to transcribe and analyze speech")
    recorder = Recorder(process_audio)
    recorder.listen()


if __name__ == "__main__":
    bootstrap()
