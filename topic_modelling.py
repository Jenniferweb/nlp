import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English 
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import random
from gensim import corpora
import pickle
import gensim
import re


# VARIABLES AND CONSTANTS
parser = English()
FILENAME = 'amazon_reviews.csv'

def tokenize(text):
	lda_tokens = []
	tokens = parser(text)
	for token in tokens:
		if token.orth_.isspace():
			continue
		elif token.like_url:
			lda_tokens.append('URL')
		elif token.orth_.startswith('@'):
			lda_tokens.append('SCREEN_NAME')
		else:
			lda_tokens.append(token.lower_)
	return lda_tokens

def get_lemma(word):
	lemma = wn.morphy(word)
	if lemma is None:
		return word
	else:
		return lemma

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)
    
stopword_list = set(nltk.corpus.stopwords.words('english'))

def text_prep(text):
	tokens = tokenize(text)
	tokens = [token for token in tokens if token != '/><br']
	tokens = [re.sub(r'[^\w\s]','',token) for token in tokens]
	tokens = [token for token in tokens if len(token) > 4] 
	#tokens = [spell.correction(token) if token in spell.unknown(tokens) else token for token in tokens]
	tokens = [token for token in tokens if token not in stopword_list]
	tokens = [get_lemma(token) for token in tokens]

	return tokens

text_data = []
with open(FILENAME, encoding="utf8") as f:
	for line in f:
		tokens = text_prep(line)
		if random.random() > 0.99:
			text_data.append(tokens)

# use LDA model: transform new doc to bag-of-words, then apply lda
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')


NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for idx, topic in ldamodel.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))








