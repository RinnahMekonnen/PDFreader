# install this via terminal
# pip3 install PyPDF2
# pip install sumy or pip3 install sumy
# May need to upgrade to the latest pip if it doesn't work^^^
# pip install --upgrade pip
import nltk as nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer



# importing required modules
import PyPDF2
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('wordnet')

all_words_train = []
v_output_train = []
lemmatized_train = []

tokenized_train = []

train_data = []
y_train_data = []

y = []

all_words = []
all_sentences = []
no_stopwords = []
v_output = []
lemmatized = []
stemmed = []
train_data_all = []

# Pre-process training data 
training_data = pd.read_csv("train_dataset_short.csv")
training_data.pop('id')
# print(training_data)

training_abstracts = training_data.iloc[:, 0]  # returns only training data abstracts
training_abstracts = training_abstracts.to_numpy().tolist()  # convert to numpy array
# training_abstarcts = training_abstracts['ABSTRACT'].tolist() #convert to list
# print(training_abstracts)
for abstract in training_abstracts:
    tokenized_train = [w for w in word_tokenize(abstract)]
    # remove numbers and symbols
    no_symbols_train = [x for x in tokenized_train if (x.isalnum())]
    no_integers_train = [x for x in no_symbols_train if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    # print(len(no_symbols_train))
    # print(len(no_integers_train))

    # remove stop words
    # nltk.download('stopwords')
    stop_words = stopwords.words("english")
    no_stopwords_train = [w for w in no_integers_train if w not in stop_words]

    # print(len(no_stopwords_train))

    # Lemmatization
    # nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

    # run through verb lemmatizer
    for word in no_stopwords_train:
        output_train = lemmatizer.lemmatize(word, pos='v')
        v_output_train.append(output_train)

    # after verb lemmatizer, run through again with noun lemmatizer
    for word in v_output_train:
        # print(output_n_train)
        # print(len(output_n_train))
        output_n_train = lemmatizer.lemmatize(word, pos='n')
        lemmatized_train.append(output_n_train)
    item = ' '.join(lemmatized_train)

    train_data_all.append(item)
    # print(len(train_data_all))
    # lemmatized_train.clear()
    # print(lemmatized_train)
    # print(len(lemmatized_train))

# print("here")
# print(type(lemmatized_train))
# print(len(lemmatized_train))

# Build a count vectorizer and extract term counts
# count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.1, max_df=0.7, max_features=100)
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(train_data_all)  # generates word counts for the words in your docs

# print(train_tc)

# create dataframe which gives us words
word_matrix = pd.DataFrame(train_tc.toarray(), columns=count_vectorizer.get_feature_names_out())
# print(word_matrix)

# Create the tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)

#print(train_tfidf.toarray())
#print(train_tfidf.shape)

y_training_data = training_data.iloc[:, 1:5]
i = 0

# label encoding 0=computer science, 1=math, 2=physics, 3=none
for i in range(len(y_training_data)):
    if y_training_data.loc[i, 'Computer Science']:
        y.append(0)
    elif y_training_data.loc[i, 'Mathematics']:
        y.append(1)
    elif y_training_data.loc[i, 'Physics']:
        y.append(2)
    else:
        y.append(3)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
# y_train =  training_data.iloc[:,1:5].to_numpy().flatten()

# print(y)

# how do I make this be an array of tfidf to be n-samples
classifier.fit(train_tfidf, y)

# creating a pdf file object
file = open('report.pdf', 'rb')  # rb = open in binary format for reading

# creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(file)

# number of pages
number_of_pages = pdfReader.getNumPages()

for i in range(number_of_pages):
    page = pdfReader.pages[i]
    page_content = page.extract_text()
    all_sentences.append(sent_tokenize(page_content))  # Tokenization
    # sentences = sent_tokenize(page_content) #sentences is of type list
    all_words = all_words + word_tokenize(page_content)  # Tokenization

# remove numbers and symbols
no_symbols = [x for x in all_words if (x.isalnum())]
no_integers = [x for x in no_symbols if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

# remove stop words
no_stopwords = [w for w in no_integers if w not in stop_words]

# run through verb lemmatizer
for word in no_stopwords:
    output = lemmatizer.lemmatize(word, pos='v')
    v_output.append(output)
    # print(word, output_v)

# after verb lemmatizer, run through again with noun lemmatizer
for word in v_output:
    output_n = lemmatizer.lemmatize(word, pos='n')
    lemmatized.append(output_n)

item_test = ' '.join(lemmatized)

# Build a count vectorizer and extract term counts
# train_tc = count_vectorizer.transform(lemmatized)  # generates word counts for the words in your docs
train_tc = count_vectorizer.transform([item_test])
# create dataframe which gives us words
word_matrix = pd.DataFrame(train_tc.toarray(), columns=count_vectorizer.get_feature_names_out())

# Create the tf-idf transformer
input_tfidf = tfidf.transform(train_tc)
# print(input_tfidf.shape)

# Predict the output categories
predictions = classifier.predict(input_tfidf)

print(predictions)

# SUMMARIZER
# get text from abstract
page = pdfReader.pages[0]
page_text = page.extract_text()
formatted = page_text.lower()
# print(formatted)
start = formatted.find("abstract")
end = formatted.find("i. ")
abstract = formatted[start:end]

# sumy summarizer
LANGUAGE = "english"
SENTENCES_COUNT = 5

parser = PlaintextParser.from_string(abstract, Tokenizer(LANGUAGE))
stemmer = Stemmer(LANGUAGE)

summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)

for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)

# printing number of pages in
#  pdf file
# print(file.numPages())

# with open("example.pdf", "rb") as pdf_file:
#     read_pdf = PyPDF2.PdfFileReader(pdf_file)
#     number_of_pages = read_pdf.getNumPages()
#     print(number_of_pages)
#     page = read_pdf.pages[0]
#     print(page)
#     page_content = page.extractText()
#     print(page_content)
# print(file.)


# creating a page object
# page = pdfReader.getPage(0)

# # extracting text from page
# print("HERE")
# print(page.extractText())
# print(pdfReader.getPage(1).extract_text())

# # closing the pdf file object
# file.close()

if __name__ == "__main__":
    print("END")
