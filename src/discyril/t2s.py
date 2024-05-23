from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io


class T2S:
    def say(self, text):
        # create a gTTS object with the desired text and language
        tts = gTTS(text=text, lang="fr")

        # use an in-memory bytes buffer
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)

        # load the audio from the buffer
        audio = AudioSegment.from_file(buffer, format="mp3")

        # play the audio
        play(audio)
