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

trailing_punctuation = ['"', '.', ',', ':']
leading_words = ['CC']

def syntax_parse(sentences, max_len):
    for sentence in sentences:
        # if sentence can fit on one line, leave as-is
        if len(sentence) <= max_len:
            print('22: short sentence', sentence, len(sentence))
        # else, parse sentence and convert to Parented Tree
        else:
            parser = CoreNLPParser()
            parse = next(parser.raw_parse(sentence))
            ptree = ParentedTree.convert(parse)
            # ptree.pretty_print()
            skip = False
            parent_index = 0
            for i in range(len(ptree[parent_index])):
                # check if upcoming tree was already added; if so, skip
                if skip == True:
                    skip = False
                    continue
                # update working tree and begin caption
                working_tree = ptree[0, i]
                caption = ' '.join(working_tree.leaves())
                
                # if working tree is a trailing punctuation, it was already added, so skip
                if working_tree.label() in trailing_punctuation:
                    continue
                # if working tree starts with leading word, add the right sibling tree and prepare to skip next iteration
                if working_tree.label() in leading_words:
                    # working_tree.append(ParentedTree.fromstring(str(working_tree.right_sibling())))
                    caption += ' ' + ' '.join(working_tree.right_sibling().leaves())
                    skip = True
                # check if right sibling tree is trailing punctuation, if so: add it to working tree
                try:
                    if working_tree.right_sibling().label() in trailing_punctuation:
                        caption += working_tree.right_sibling().leaves()[0]
                    if skip == True:
                        if working_tree.right_sibling().right_sibling().label() in trailing_punctuation:
                            caption += working_tree.right_sibling().right_sibling().leaves()[0]
                except IndexError:
                    continue

                print(caption, len(caption))

long_test = ["As Julia Child once said, with enough butter anything is good, and that is especially true with fresh butter, and so today I am going to hand churn some fresh butter.", "So thank you to Squarespace for sponsoring this video as I dive in to butter this time on Tasting History.", "So I've been searching for an actual historic recipe for making butter but they're really hard to find because it seems like throughout most of history people just knew how to make butter so nobody really wrote it down."]
short_test = ["I saw the lazy dog, and I saw the red cat."]
syntax_parse(short_test, 32)

server.stop()