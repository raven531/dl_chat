import random
import json
import torch
from models import model

from utils.nltk_utils import bag_of_words, tokenize

FILE = "data/data.pth"
data = torch.load(FILE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('../data/intents.json', 'rb') as f:
    intents = json.load(f)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

m = model.NeuralNet(input_size, hidden_size, output_size).to(device)
m.load_state_dict(model_state)
m.eval()

bot_name = "Sam"

if __name__ == "__main__":
    print("Let's data! type 'quit' to exit")

    while True:
        sentence = input("You: ")

        if sentence == "quit":
            break
        sentence = tokenize(sentence)
        x = bag_of_words(sentence, all_words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x).to(device)

        output = m(x)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:

            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    print(f'{bot_name}: {random.choice(intent["responses"])}')
        else:
            print(f'{bot_name}: I do not understand...')
