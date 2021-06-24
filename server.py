import random
import json
import torch
from models import model

from os.path import join, dirname, realpath

from flask import Flask, request, jsonify
from utils.nltk_utils import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(join(dirname(realpath(__file__)), 'data/intents.json'), 'rb') as f:
    intents = json.load(f)

FILE = join(dirname(realpath(__file__)), 'data/data.pth')
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

m = model.NeuralNet(input_size, hidden_size, output_size).to(device)
m.load_state_dict(model_state)
m.eval()


def predict(x):
    output = m(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
    else:
        return "I do not understand..."


def transform(sentence):
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)

    return x


app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        sentence = request.values['sentence']
        transformed_sentence = transform(sentence)
        prediction = {"message": predict(transformed_sentence)}

        return jsonify(prediction)

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
