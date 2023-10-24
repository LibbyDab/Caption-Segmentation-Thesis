from nltk.tokenize import sent_tokenize

# splits a caption into two lines where the top is always a little shorter than the bottom
def pyramid_parse(sentences, max_len):
    for sentence in sentences:
        # if sentence can fit on one line, leave as-is
        if len(sentence) <= max_len:
            print(sentence, len(sentence))
        # elif sentence can fit on two lines, add words to shorter line (top or bottom) until no more words
        elif len(sentence) <= max_len*2:
            top_line = str()
            bottom_line = str()
            words = sentence.split()
            for word in words:
                if len(top_line) < max_len:
                    top_line += word + ' '
                else:
                    bottom_line += word + ' '
            top_line = top_line.strip()
            bottom_line = bottom_line.strip()
            # if top line is longer, move word(s) to bottom line
            while len(top_line) > len(bottom_line):
                word = top_line.split()[-1]
                bottom_line = word + ' ' + bottom_line
                top_line = ' '.join([i for i in top_line.split()[:-1]])
            print(top_line, len(top_line))
            print(bottom_line, len(bottom_line))
        # else sentence must be split across multiple captions
        else:
            pass
            # print(sentence, len(sentence))

with open('How to Make Old Fashioned Butter transcript.txt') as transcript:
    sentences = sent_tokenize(transcript.read())
    pyramid_parse(sentences, 32)