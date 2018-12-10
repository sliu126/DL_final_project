import pickle
import io
import sys
import logging

SOS_token = "<SOS>"
EOS_token = "<EOS>"
SOS_index = 0
EOS_index = 1

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
file=open('data_all.pkl','rb')
data=pickle.load(file)

for example in data["train"]:
	example["title"]=example["title"].lower()
	for char in [':',',','\'','\"','?','-','.','!','(',')','&','/']:
		example["title"]=example["title"].replace(char,' ')

for example in data["dev"]:
    example["title"]=example["title"].lower()
    for char in [':',',','\'','\"','?','-','.','!','(',')','&','/']:
        example["title"]=example["title"].replace(char,' ')

for example in data["test"]:
    example["title"]=example["title"].lower()
    for char in [':',',','\'','\"','?','-','.','!','(',')','&','/']:
        example["title"]=example["title"].replace(char,' ')

#for i in range(10):
#	print(data["train"][i]["title"])

class Vocab:
    #This class handles the mapping between the words and their indicies
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class CharVocab:
    #This class handles the mapping between the chars and their indicies
    def __init__(self):
        self.char2index = {}
        self.char2count = {}
        self.index2char = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_chars = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for char in sentence:
            self._add_char(char)

    def _add_char(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1  

			
def make_vocabs(data):
    #Creates the vocabs based on the training corpus.
    vocab = Vocab()
    char_vocab = CharVocab()
    for example in data:
        sent=example["title"]
        vocab.add_sentence(sent)
        char_vocab.add_sentence(sent)

    logging.info('vocab size: %s',vocab.n_words)
    logging.info('char vocab size: %s',char_vocab.n_chars)
    #print(vocab.n_words)
    #print(vocab.index2word)
    return vocab, char_vocab

vocab, char_vocab = make_vocabs(data["train"])

# print(char_vocab.n_chars + 1)

def make_data_input(data_all, vocab, char_vocab):
    data_input = {}
    data_input["train"] = []
    data_input["dev"] = []
    data_input["test"] = []
    char_data_input = {}
    char_data_input["train"] = []
    char_data_input["dev"] = []
    char_data_input["test"] = []
    word2index = vocab.word2index
    char2index = char_vocab.char2index
    for dataset_key in data:
        dataset = data[dataset_key]
        for data_point in dataset:
            title = data_point["title"]
            cat = data_point["cat"]
            words = title.split(' ')
            words_indices = []
            for word in words:
                if word in word2index:
                    vocab_index = word2index[word]
                else:
                    vocab_index = vocab.n_words  # OOV
                words_indices.append(vocab_index)
            data_input[dataset_key].append((words_indices, cat))
            chars_indices = []
            for char in title:
                if char in char2index:
                    char_vocab_index = char2index[char]
                else:
                    char_vocab_index = char_vocab.n_chars # OOC
                chars_indices.append(char_vocab_index)
            char_data_input[dataset_key].append((chars_indices, cat))

    # print(data_input)
    # print(char_data_input)
    return data_input, char_data_input

data_input, char_data_input = make_data_input(data, vocab, char_vocab)

# pickle.dump(data_input, open("data_input.pkl", "wb"))
# pickle.dump(char_data_input, open("char_data_input.pkl", "wb"))
