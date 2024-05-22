import spacy
import os

dir_path = os.getcwd()


class NLP:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.nlp = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise Exception("Model not found")

        # load model
        self.nlp = spacy.load(self.model_path)

    def predict_intent(self, text: str) -> str:
        if not self.nlp:
            raise Exception("Model not loaded")

        # predict intent from text
        result = self.nlp(text)
        return max(result.cats, key=lambda key: result.cats[key])
