import os
# from nltk.tokenize import sent_tokenize
from nltk.parse.corenlp import CoreNLPServer
from nltk.parse.corenlp import CoreNLPParser
from nltk.tree import *

# start Core NLP Server
STANFORD = os.path.join("stanford-corenlp-4.5.5")
server = CoreNLPServer(
    os.path.join(STANFORD, "stanford-corenlp-4.5.5.jar"),
    os.path.join(STANFORD, "stanford-corenlp-4.5.5-models.jar"),
)
server.start()

def syntax_parse(sentences, max_len):
    for sentence in sentences:
        # if sentence can fit on one line, leave as-is
        if len(sentence) <= max_len:
            print(sentence, len(sentence))
        # else, parse sentence and convert to Parented Tree
        else:
            parser = CoreNLPParser()
            parse = next(parser.raw_parse(sentence))
            ptree = ParentedTree.convert(parse)
            ptree.pretty_print()

server.stop()

test = ["As Julia Child once said with enough butter anything is good, and that is especially true with fresh butter, and so today I am going to hand churn some fresh butter.", "So thank you to Squarespace for sponsoring this video as I dive in to butter this time on Tasting History.", "So I've been searching for an actual historic recipe for making butter but they're really hard to find because it seems like throughout most of history people just knew how to make butter so nobody really wrote it down."]
syntax_parse(test, 32)