from numpy.random import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64



import numpy as np

class PrepareDataset:
	def __init__(self,n_sentences = 1000,train_split = 0.9, **kwargs):
		# super(PrepareDataset, self).__init__(**kwargs)
		self.n_sentences = n_sentences# Number of sentences to include in the dataset
		self.train_split = train_split # Ratio of the training data split
		self.tokenizer = None
		self.distinct = []
	# Fit a tokenizer
	def create_tokenizer(self, dataset):
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(dataset)
		self.tokenizer = tokenizer
		return tokenizer

	def find_seq_length(self, dataset):
		return max(len(seq.split()) for seq in dataset)

	def find_vocab_size(self, tokenizer, dataset):
		tokenizer.fit_on_texts(dataset)

		return len(tokenizer.word_index) + 1

	def encode_disease(self,diseases):
		data_encodings = []
		for disease_list in diseases:
			encoding = np.zeros(len(self.distinct))
			for category in self.distinct:
				if category in disease_list:
					encoding[self.distinct.index(category)]=1
			data_encodings.append(encoding)
		return data_encodings 
	
	def find_distinct(self,labels):
		elements_to_remove = ['chest x-ray; ','; ','','No Finding']
		for text in labels:
			split = text.split('\'')
			for word in split:
				if word not in self.distinct:
					self.distinct.append(word)
		for element in elements_to_remove:
			self.distinct.remove(element)
	
	def __call__(self, data, **kwargs):

		# Reduce dataset size
		reports =  data[0]

		# Include start and end of string tokens
		for i in range(len(reports)):
			reports[i] = 	"<START> " + reports[i] + " <EOS>"



		tokenizer = self.create_tokenizer(reports)
		seq_length = self.find_seq_length(reports)
		vocab_size = self.find_vocab_size(tokenizer, reports)



		# Split the dataset
		train = [reports[:int(self.n_sentences * self.train_split)]]

		test = [reports[int(self.n_sentences * self.train_split):]]
		
		# Encode and pad the input sequences
		sequnces = tokenizer.texts_to_sequences(train[0])
		sequnces = pad_sequences(sequnces, maxlen=seq_length, padding='post')



		test_sequnces = tokenizer.texts_to_sequences(test[0])
		test_sequnces = pad_sequences(test_sequnces, maxlen=seq_length, padding='post')

		return sequnces, None, test_sequnces, None, seq_length, vocab_size
	