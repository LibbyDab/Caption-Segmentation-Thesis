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

# fill the line with the maximun number of words without going over the char. limit
# return the index of the word right before the split and the key of the word pair for shared parents index_word1_word2
def line_fill_index(words, max_len):
    top_line = str()
    index = -1
    for word in words:
        if len(top_line) < max_len:
            top_line += word + ' '
            index += 1
        else:
            break
    # print(top_line)
    key = '_'.join([str(index), words[index], words[index+1]])
    return index, key

# for every consecutive word pair, count the number of shared parents
# return as a dictionary {index_word1_word2 : num of shared parents}
def count_shared_parents(tree, words):
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
    # print("Number of shared parents for word pair:")
    # for key, value in shared_parents.items():
    #     print(key, ":", value)
    return shared_parents

def syntax_segment(sentences, max_len):
    for sentence in sentences:
        # if sentence can fit on one line, leave as-is
        if len(sentence) <= max_len:
            print(sentence, len(sentence))
        # else use syntax to identify break candidates with minimal shared parents
        else:
            parser = CoreNLPParser()
            parse = next(parser.raw_parse(sentence))
            words = parse.leaves()
            # parse.pretty_print()
            print('63:', parse.pos())
            index, name = line_fill_index(words, max_len)
            shared_parents = count_shared_parents(parse, words)
            # break_candidates = sorted(shared_parents.items(), key=lambda x:x[1])
            # for i in range(len(break_candidates)):
            #     print(break_candidates[i])
            #     index = int(break_candidates[i][0].split("_")[0])
            #     sent1 = words[:index+1]
            #     sent2 = words[index+1:]
            #     print(break_candidates[i], ' '.join(sent1), '/', ' '.join(sent2))

syntax_segment(["So thank you to Squarespace for sponsoring this video as I dive in to butter this time on Tasting History."], 32)

server.stop()