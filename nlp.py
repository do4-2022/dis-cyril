import spacy
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


class NLP:
    def __init__(self):
        self.nlp = None

    def load_model(self):
        if not os.path.exists(f"{dir_path}/models/nlp"):
            raise Exception("Model not found")

        # load model
        self.nlp = spacy.load(f"{dir_path}/models/nlp")

    def predict_intent(self, text):
        if not self.nlp:
            raise Exception("Model not loaded")

        # predict intent from text
        result = self.nlp(text)
        return max(result.cats, key=result.cats.get)
