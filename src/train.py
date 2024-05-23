import spacy
import os

from spacy.tokens import DocBin
from spacy.training import Example

# init nlp and textcat
nlp = spacy.blank("fr")
textcat = nlp.add_pipe("textcat")

# add labels in textcat
cats = ["weather_montpellier", "unknown"]
for cat in cats:
    textcat.add_label(cat)

# load training data
dir_path = os.getcwd()
doc_bin = DocBin().from_disk(f"{dir_path}/data/training.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

# create examples
examples = []
for doc in docs:
    example = Example.from_dict(doc, {"cats": doc.cats})
    examples.append(example)

# train params
epochs = 3

print(f"Start training for {epochs} epochs ...")

# train model
optimizer = nlp.begin_training()
for epoch in range(epochs):
    losses = {}
    for doc in docs:
        nlp.update(examples, losses=losses)
    print(f"Epoch {epoch} Loss: {losses}")

print("Training finished")

# save model
nlp.to_disk(f"{dir_path}/models/nlp")

# load test data
doc_bin = DocBin().from_disk(f"{dir_path}/data/test.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

# load model
model = spacy.load(f"{dir_path}/models/nlp")

# test model
correct = 0
for phrase in docs:
    result = model(phrase.text)
    expected_cat = max(phrase.cats, key=phrase.cats.get)
    predicted_cat = max(result.cats, key=result.cats.get)
    if expected_cat == predicted_cat:
        correct += 1

accuracy = correct / len(docs)
print(f"Test data size: {len(docs)}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy}")
