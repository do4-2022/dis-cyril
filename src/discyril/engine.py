import logging
import numpy as np
import os
import typing
import requests

from discyril.nlp import NLP
from discyril.recorder import Recorder
from discyril.s2t import S2T, SAMPLE_SIZE as S2T_SAMPLE_SIZE
from discyril.wwd import WWD, SAMPLE_SIZE as WWD_SAMPLE_SIZE
from discyril.t2s import T2S

DEFAULT_S2T_MODEL_NAME = "Ilyes/wav2vec2-large-xlsr-53-french"
DEFAULT_NLP_MODEL_PATH = f"{os.getcwd()}/models/nlp"
DEFAULT_WWD_MODEL_PATH = f"{os.getcwd()}/models/wwd/dis-cyril.tflite"


logger = logging.getLogger(__name__)


class Engine:
    def __init__(
        self,
        s2t_model_name: str = DEFAULT_S2T_MODEL_NAME,
        nlp_model_path: str = DEFAULT_NLP_MODEL_PATH,
        wwd_model_path: str = DEFAULT_WWD_MODEL_PATH,
    ):
        logger.info("Initializing Engine...")

        self.s2t = S2T(s2t_model_name)
        self.nlp = NLP(nlp_model_path)
        self.wwd = WWD(wwd_model_path)
        self.t2s = T2S()
        self.recorder = Recorder(
            callback=self.process,
            on_silence=self.on_silence,
            chunk_size=WWD_SAMPLE_SIZE,
        )

        self.awake = False

        logger.info("Engine initialized")

    def bootstrap(self):
        logger.info("Loading models...")

        self.s2t.load_model()
        self.nlp.load_model()
        self.wwd.load_model()

        logger.info("Loaded models")

    def start(self):
        self.recorder.listen()

    def process(self, frames: typing.Iterable[np.ndarray[np.int16, typing.Any]]):
        if not self.awake:
            if self.wwd.detect(frames):
                self.wake_up()
        else:
            transcription = self.s2t.transcribe(frames)
            intent = self.nlp.predict_intent(transcription)

            logger.info(f"Transcription: {transcription}")
            logger.info(f"Intent: {intent}")

            if intent == "weather_montpellier":
                r = requests.get("https://api.tibuzin.do-2021.fr/weather")
                response: dict[str, str] = r.json()
                self.t2s.say(response.get("text", ""))  # type: ignore
            else:
                self.t2s.say(  # type: ignore
                    "Je suis désolé, je n'ai pas compris ce que vous avez dit."
                )

    def wake_up(self):
        if self.awake:
            return

        logger.info("Waking up...")
        self.awake = True
        self.recorder.silence_detection = True
        self.recorder.set_chunk_size(S2T_SAMPLE_SIZE)
        logger.info("Awake word detection disabled and speech-to-text enabled")

    def on_silence(self):
        logger.info("Sleeping...")
        self.awake = False
        self.recorder.silence_detection = False
        self.recorder.set_chunk_size(WWD_SAMPLE_SIZE)
        logger.info("Awake word detection enabled and speech-to-text disabled")
