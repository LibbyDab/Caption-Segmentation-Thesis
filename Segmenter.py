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

# custom detokenization of a list of strings to handle issues
def custom_detokenize(tokens):
    while len(tokens) > 1:
        # detect and correct parentheses in text
        if tokens[0] == '-LRB-':
            tokens[0] = '(' + tokens[1]
        elif tokens[1] == '-LRB-':
            tokens[0] = tokens[0] + ' ('
        elif tokens[0] == '-RRB-':
            tokens[0] = ')' + tokens[1]
        elif tokens[1] == '-RRB-':
            tokens[0] = tokens[0] + ')'
        # detect hyphens and fix spacing
        elif tokens[0][-1] == '-' or tokens[1] == '-':
            tokens[0] = tokens[0] + tokens[1]
        else:
            tokens[0] = TreebankWordDetokenizer().detokenize([tokens[0], tokens[1]])
        tokens.pop(1)
    return tokens[0]

# takes a list of words and cuts it to one over the maximun number of words without going over the character limit
# returns the index of the word right after the split and the list keys of the word pairs in the line
def line_fill(sentence, max_len, start_index):
    words = sentence
    end_index = 0
    keys = []
    line = words[0]
    end_of_sent = True
    for i in range(1, len(words)):
        line = custom_detokenize([line, words[i]])
        key = '_'.join([str(start_index + end_index), words[end_index], words[end_index+1]])
        keys.append(key)
        end_index += 1
        if len(line) <= max_len:
            continue
        else:
            end_of_sent = False
            break
    if len(words) == 1 or end_of_sent:
        keys.append('end_of_sent')
    return keys

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
contractions = ["'s", "'re", "'t", "'ll", "'d", "'ve", "'m", "n't"]
dont_split_after = ['$', '-LRB-', 'DT', 'PDT', 'WDT', 'TO', 'CC', 'IN', 'PRP$', 'POS']
dont_split_before = ['.', ',', ':', '-RRB-']
dont_split_between = {'JJ': ['JJ', 'JJR', 'JJS', 'CD', 'NNS', 'NN', 'NNP', 'NNPS'], 
                      'JJR': ['JJ', 'JJR', 'JJS', 'CD', 'NNS', 'NN', 'NNP', 'NNPS'], 
                      'JJS': ['JJ', 'JJR', 'JJS', 'CD', 'NNS', 'NN', 'NNP', 'NNPS'], 
                      'RB' : ['RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS',],
                      'RBR' : ['RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS',],
                      'RBS' : ['RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS',],
                      'VBZ' : ['VBG', 'VBN', 'TO', 'RB', 'RBR', 'RBS'],
                      'VBD' : ['VBG', 'VBN', 'TO', 'RB', 'RBR', 'RBS'],
                      'VBP' : ['VBG', 'VBN', 'TO', 'RB', 'RBR', 'RBS'],
                      'VB' : ['VBG', 'VBN', 'TO', 'RB', 'RBR', 'RBS'],
                      'VBN' : ['VBG', 'TO', 'RB', 'RBR', 'RBS'],
                      'VBG' : ['TO', 'RB', 'RBR', 'RBS'],
                      'MD' : ['VB', 'RB', 'RBR', 'RBS']
                      }
do_split_after = {',', ':'}

def syntax_segment(sentence, max_len):
    parser = CoreNLPParser()
    parse = next(parser.raw_parse(sentence))
    words = parse.leaves()
    # parse.pretty_print()
    pos_tags = parse.pos()
    # print(pos_tags)

    # find number of shared parents for each word pair
    break_candidates = count_shared_parents(parse, words)

    # apply penalty for breaking rules
    for name in (break_candidates):
        index = int(name.split("_")[0])
        # don't break within words
        if words[index+1] in contractions or \
        pos_tags[index+1][1] == 'POS' or \
        pos_tags[index][1] == 'HYPH' or \
        pos_tags[index+1][1] == 'HYPH':
            break_candidates[name] = break_candidates.get(name) + 20
        # don't break after certain symbols or parts-of-speech
        if pos_tags[index][1] in dont_split_after:
            break_candidates[name] = break_candidates.get(name) + 10
        # don't break before end punctuation
        if pos_tags[index+1][1] in dont_split_before:
            break_candidates[name] = break_candidates.get(name) + 20
        # discourage split between certain part-of-speech pairs
        if pos_tags[index][1] in dont_split_between.keys():
            if pos_tags[index+1][1] in dont_split_between[pos_tags[index][1]]:
                break_candidates[name] = break_candidates.get(name) + 5
        # encourage split after certain punctuation
        if pos_tags[index][1] in do_split_after:
            break_candidates[name] = break_candidates.get(name) - 5
    
    # print("Cost for splitting word pair:")
    # for key, value in possible_breaks.items():
    #     print(key, ":", value)
        
    segments = []
    start_index = 0
    while words:
        keys = line_fill(words, max_len, start_index)
        possible_breaks = {}
        try:
            for i in range(len(keys)):
                possible_breaks[keys[i]] = break_candidates[keys[i]]
        except KeyError as e:
            if e.args[0] == 'end_of_sent':
                segment = custom_detokenize(words)
                segments.append(segment)
                # print(segments)
                return segments

        if len(possible_breaks) == 1:
            optimal_break_index = int(list(possible_breaks.keys())[0].split("_")[0]) + 1
        else:
            optimal_break_value = min(possible_breaks.values())
            optimal_break_key = [key for key in possible_breaks if possible_breaks[key] == optimal_break_value]
            optimal_break_index = int(optimal_break_key[-1].split("_")[0])

        segment = custom_detokenize(words[:optimal_break_index-start_index+1])
        segments.append(segment)

        words = words[optimal_break_index-start_index+1:]
        start_index = optimal_break_index+1
    
    # print(segments)
    return segments

caption_file = open('NatGeo The Barking Deer System Captions.txt', 'w')
with open('NatGeo The Barking Deer Transcript.txt', 'r') as transcript:
    lines = transcript.readlines()
    sentences = []
    for line in lines:
        sentences.extend(sent_tokenize(line))
    max_len = 32
    segments = []
    caption_number = 1
    while sentences:
        if len(sentences[0]) > max_len:
            # print('\n', sentences[0])
            lines = syntax_segment(sentences[0], max_len)
        else:
            lines = [sentences[0]]
        line_num = 1
        sentences.pop(0)
        # for line in lines:
        #     print(line, len(line))
        while lines:
            try:
                if lines[0][-1] in ['.', ',', ':', ')', '}', ']']:
                    pass
                elif lines[1][0] in ['(', '{', '[']:
                    pass
                elif len(custom_detokenize([lines[0], lines[1]])) <= max_len:
                    lines[1] = custom_detokenize([lines[0], lines[1]])
                    lines.pop(0)
                    continue
                else:
                    pass
            except IndexError:
                pass
            if line_num % 2 == 0:
                caption_file.write(lines[0] + '\n' + '\n')
                caption_number += 1
            elif len(lines) == 1: 
                caption_file.write(str(caption_number) + '\n')
                caption_file.write(lines[0] + '\n' + '\n')
                caption_number += 1
            else:
                caption_file.write(str(caption_number) + '\n')
                caption_file.write(lines[0] + '\n')
            lines.pop(0)
            line_num += 1
    caption_file.close()

server.stop()