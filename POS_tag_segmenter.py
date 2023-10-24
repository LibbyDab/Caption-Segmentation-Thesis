import nltk
from nltk.tokenize import word_tokenize

long_test = ["As Julia Child once said, with enough butter anything is good, and that is especially true with fresh butter, and so today I am going to hand churn some fresh butter.", "So thank you to Squarespace for sponsoring this video as I dive in to butter this time on Tasting History.", "So I've been searching for an actual historic recipe for making butter but they're really hard to find because it seems like throughout most of history people just knew how to make butter so nobody really wrote it down."]
short_test = ["I saw the lazy dog, and I saw the red cat."]

for sentence in long_test:
    tokens = word_tokenize(sentence)
    print(nltk.pos_tag(tokens))