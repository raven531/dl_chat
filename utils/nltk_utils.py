import numpy as np
import nltk

from nltk.stem.porter import PorterStemmer

nltk.download('punkt')

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for i, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[i] = 1.0

    return bag


if __name__ == "__main__":
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]

    print(bag_of_words(sentence, words))
