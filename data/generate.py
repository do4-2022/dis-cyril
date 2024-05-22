import spacy
import os

from spacy.tokens import DocBin
from weather import weather_phrases
from unknown import unknown_phrases


def apply_default_cats(doc):
    cats = ["weather_montpellier", "unknown"]
    for cat in cats:
        doc.cats[cat] = 0.0
    return doc


# init nlp
nlp = spacy.blank("fr")

# create doc bin for training and testing data
training_doc_bin = DocBin()
test_doc_bin = DocBin()

# set training data
data = [("weather_montpellier", weather_phrases), ("unknown", unknown_phrases)]

# add each phrase to the training and test data with the corresponding label
for cat, phrases in data:
    index = int(len(phrases) * 0.8)
    training_data = phrases[:index]
    test_data = phrases[index:]

    for phrase in training_data:
        doc = nlp(phrase)
        doc = apply_default_cats(doc)
        doc.cats[cat] = 1.0
        training_doc_bin.add(doc)

    for phrase in test_data:
        doc = nlp(phrase)
        doc = apply_default_cats(doc)
        doc.cats[cat] = 1.0
        test_doc_bin.add(doc)

# save training data to disk
dir_path = os.path.dirname(os.path.realpath(__file__))
training_doc_bin.to_disk(f"{dir_path}/training.spacy")
training_doc_bin.to_disk(f"{dir_path}/test.spacy")
