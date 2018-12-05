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

for i in range(10):
	print(data["train"][i]["title"])

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
			
def make_vocabs(data):
    #Creates the vocabs based on the training corpus.
    vocab = Vocab()
    for example in data:
        sent=example["title"]
        vocab.add_sentence(sent)

    logging.info('vocab size: %s',vocab.n_words)
    print(vocab.n_words)
    print(vocab.index2word)
    return vocab
	
make_vocabs(data["train"])



