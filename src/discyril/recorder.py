import pyaudio
import math
import struct
import time
import typing
import logging
import numpy as np

SWIDTH = 2
SHORT_NORMALIZER = 1.0 / 32768.0
MAX_SILENCE_DURATION = 1  # in second

THRESHOLD = 15
FORMAT = pyaudio.paInt16
RATE = 16000
CHANNELS = 1

DEFAULT_CHUNK_SIZE = 1024

logger = logging.getLogger(__name__)


class Recorder:
    pyaudio_instance = pyaudio.PyAudio()

    @staticmethod
    def rms(frame: bytes):
        count = len(frame) / SWIDTH
        struct_format = "%dh" % (count)
        shorts = struct.unpack(struct_format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZER
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(
        self,
        callback: (
            typing.Callable[[typing.Iterable[np.ndarray[np.int16, typing.Any]]], None]
            | None
        ) = None,
        on_noise: typing.Callable[[], None] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        logger.info("Initializing Recorder...")

        self._callback = callback
        self._on_noise = on_noise
        self._chunk_size = chunk_size
        self._stream = Recorder.pyaudio_instance.open(
            format=FORMAT,
            rate=RATE,
            channels=CHANNELS,
            frames_per_buffer=self._chunk_size,
            input=True,
        )

        logger.info("Recorder initialized")

    def record(self):
        rec: list[bytes] = []
        current = time.time()
        end = time.time() + MAX_SILENCE_DURATION

        # record the audio until the end time
        while current <= end:
            chunk = self._stream.read(self._chunk_size)

            # if there is no silence, update the possible end time of the record
            if self.rms(chunk) >= THRESHOLD:
                end = time.time() + MAX_SILENCE_DURATION

            current = time.time()
            rec.append(chunk)

        # remove the last frames to avoid no sound at the end
        nb_frames_to_remove = int(MAX_SILENCE_DURATION * RATE / self._chunk_size)
        for _ in range(nb_frames_to_remove):
            rec.pop()

        # now, process the recorded frames
        self.process(rec)

    def process(self, frames: list[bytes]):
        # Convert the byte data to numpy array
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)

        if self._callback:
            self._callback(audio_data)

    def listen(self):
        logger.info("Listening...")
        while True:
            try:
                chunk = self._stream.read(self._chunk_size)
                rms_val = self.rms(chunk)
                if rms_val > THRESHOLD:
                    logger.info(f"Sound detected ({rms_val})")

                    if self._on_noise:
                        self._on_noise()

                    self.record()
                    logger.info("Silence detected")
                    self.reset_chunk_size()

            except KeyboardInterrupt:
                self._stream.stop_stream()
                self._stream.close()
                logger.info("Recorder stopped by user")
                break

    def reset_chunk_size(self):
        logger.info("Resetting chunk size to default...")
        self.set_chunk_size(DEFAULT_CHUNK_SIZE)
        logger.info("Chunk size reset to default")

    def set_chunk_size(self, chunk_size: int):
        self._chunk_size = chunk_size
        self._stream.stop_stream()
        self._stream.close()
        self._stream = Recorder.pyaudio_instance.open(
            format=FORMAT,
            rate=RATE,
            channels=CHANNELS,
            frames_per_buffer=self._chunk_size,
            input=True,
        )
