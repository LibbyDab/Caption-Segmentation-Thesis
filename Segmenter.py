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

parser = CoreNLPParser()
tree = next(parser.raw_parse("I saw the lazy dog and I saw the orange cat."))
tree.pretty_print()

words = tree.leaves()
treepositions = []
for i in range(len(words)):
    treeposition = list(tree.leaf_treeposition(i))
    treepositions.append(treeposition)
# for i in range(len(words)):
#     print(words[i], treepositions[i])
shared_parents = {}
for i in range(len(words)-1):
    shared_parents[f'{i}_{words[i]}_{words[i+1]}'] = 0
    for j in range(len(treepositions[i])):
        if treepositions[i][j] == treepositions[i+1][j]:
            shared_parents[f'{i}_{words[i]}_{words[i+1]}'] += 1
        else:
            break
print("Number of shared parents for word pair:")
for key, value in shared_parents.items():
    print(key, ":", value)

server.stop()