import logging
import numpy as np
import openwakeword.utils  # type: ignore
import typing

from openwakeword.model import Model  # type: ignore

DEFAULT_THRESHOLD = 0.5
SAMPLE_SIZE = 16000

logger = logging.getLogger(__name__)


class WWD:
    def __init__(self, model_path: str, threshold: float = DEFAULT_THRESHOLD):
        self.model_path = model_path
        self.threshold = threshold

    def load_model(self):
        openwakeword.utils.download_models()
        self._model = Model(wakeword_models=[self.model_path])

    def detect(self, frame: typing.Iterable[np.ndarray[np.int16, typing.Any]]) -> bool:
        # convert to numpy array
        prediction: dict[str, float] = self._model.predict(frame)  # type: ignore
        logger.debug(f"Prediction: {prediction}")

        for label, confidence in prediction.items():
            if confidence >= self.threshold:
                logger.info(f"Detected wake word: {label}")
                return True
        return False
