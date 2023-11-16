import os
from nltk.tokenize import sent_tokenize, word_tokenize
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

# takes a list of words and cuts it to the maximun number of words without going over the character limit
# returns the index of the word right after the split and the list keys of the word pairs in the line
def line_fill(sentence, max_len, start_index):
    words = sentence
    end_index = 0
    keys = []
    line = str()
    for word in words:
        if len(TreebankWordDetokenizer().detokenize([line, word])) <= max_len:
            line = TreebankWordDetokenizer().detokenize([line, word])
            end_index += 1
            try:
                key = '_'.join([str(start_index + end_index - 1), words[end_index-1], words[end_index]])
                keys.append(key)
            except IndexError:
                keys.append('end_of_sent')
        else:
            break
    return end_index, keys

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

# lists of POS rules
dont_split_after = ['$', '-', 'CC', 'DT', 'TO', 'IN', 'RB', 'RBR', 'RBS', 'PDT', 'WDT']
dont_split_before = ['.', ',', ':', '-', '%', 'POS', "'s", "'re", "'t", "'ll", "'d", "'ve", "'m", "n't"]
dont_split_between = {'JJ': ['NNS', 'NN', 'NNP', 'NNPS'], 'DT' : ['NNS', 'NN', 'NNP', 'NNPS']}
split_leading = ['.', ',', ':']

def syntax_segment(sentence, max_len):
    parser = CoreNLPParser()
    parse = next(parser.raw_parse(sentence))
    words = parse.leaves()
    # parse.pretty_print()
    pos_tags = parse.pos()
    # print(pos_tags)

    # find number of shared parents for each word pair
    break_candidates = count_shared_parents(parse, words)

    # apply POS rules as listed above
    for name in (break_candidates):
        index = int(name.split("_")[0])
        if pos_tags[index][1] in dont_split_after:
            break_candidates[name] = break_candidates.get(name) + 5
        if pos_tags[index][1] in split_leading:
            break_candidates[name] = break_candidates.get(name) - 5
        if pos_tags[index+1][1] in dont_split_before or words[index+1] in dont_split_before:
            break_candidates[name] = break_candidates.get(name) + 5

    segments = []
    start_index = 0
    while words:
        end_index, keys = line_fill(words, max_len, start_index)
        possible_breaks = {}
        for key in keys:
            try:
                possible_breaks[key] = break_candidates[key]
            except KeyError as e:
                if e.args[0] == 'end_of_sent':
                    segment = TreebankWordDetokenizer().detokenize(words)
                    segments.append(segment)
                    return segments
        optimal_break_value = min(possible_breaks.values())
        optimal_break_key = [key for key in possible_breaks if possible_breaks[key] == optimal_break_value]
        optimal_break_index = (int(optimal_break_key[-1].split("_")[0]))

        segment = TreebankWordDetokenizer().detokenize(words[:optimal_break_index-start_index+1])
        segments.append(segment)

        words = words[optimal_break_index-start_index+1:]
        start_index = optimal_break_index+1

caption_file = open('test captions.txt', 'w')
with open('test transcript.txt', 'r') as transcript:
    sentences = sent_tokenize(transcript.read())
    max_len = 32
    segments = []
    while sentences:
        if len(sentences[0]) > max_len:
            lines = syntax_segment(sentences[0], max_len)
        else:
            lines = [sentences[0]]
        line_num = 1
        while lines:
            try:
                if len(TreebankWordDetokenizer().detokenize([lines[0], lines[1]])) <= max_len:
                    lines[1] = TreebankWordDetokenizer().detokenize([lines[0], lines[1]])
                    lines.pop(0)
                    continue
                else:
                    pass
            except IndexError:
                pass
            if line_num % 2 == 0 or len(lines) == 1:
                caption_file.write(lines[0] + ' ' + str(len(lines[0])) + '\n' + '\n')
            else:
                caption_file.write(lines[0] + ' ' + str(len(lines[0])) + '\n')
            lines.pop(0)
            line_num += 1
        sentences.pop(0)
    caption_file.close()

server.stop()