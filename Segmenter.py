import os
from nltk.tokenize import sent_tokenize
from nltk.parse.corenlp import CoreNLPServer
from nltk.parse.corenlp import CoreNLPParser
from nltk.tree import *
from nltk.tokenize.treebank import TreebankWordDetokenizer

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

dont_split_leading = ['CC', 'CD', 'DT', 'IN', 'JJ']
dont_split_trailing = ['"', '.', ',', ':']

def syntax_segment(sentence):
    segments = []
    # use syntax to identify break candidates with minimal shared parents
    parser = CoreNLPParser()
    parse = next(parser.raw_parse(sentence))
    words = parse.leaves()
    # parse.pretty_print()
    pos_tags = parse.pos()
    # index, name = line_fill_index(words, max_len)
    break_candidates = count_shared_parents(parse, words)
    for name in (break_candidates):
        index = int(name.split("_")[0])
        if pos_tags[index][1] in dont_split_leading:
            break_candidates[name] = break_candidates.get(name) + 10
        if pos_tags[index+1][1] in dont_split_trailing:
            break_candidates[name] = break_candidates.get(name) + 10
    # for key, value in break_candidates.items():
    #     print(key, ":", value)
    optimal_break_values = min(break_candidates.values())
    optimal_break_index = [int(key.split("_")[0]) for key in break_candidates if break_candidates[key] == optimal_break_values]
    start_caption_index = 0
    for optimal_break in optimal_break_index:
        segment = words[start_caption_index:optimal_break+1]
        start_caption_index = optimal_break+1
        segments.append(TreebankWordDetokenizer().detokenize(segment))
    segments.append(TreebankWordDetokenizer().detokenize(words[start_caption_index:]))
    return segments

# sentences = ["As Julia Child once said with enough butter anything is good, and that is especially true with fresh butter, and so today I am going to hand churn some fresh butter."]

with open('How to Make Old Fashioned Butter transcript.txt') as transcript:
    sentences = sent_tokenize(transcript.read())
    max_len = 32
    while sentences:
        if len(sentences[0]) < max_len:
            print(sentences[0], len(sentences[0]))
            sentences.pop(0)
        else:
            sentences = syntax_segment(sentences[0]) + sentences[1:]

server.stop()