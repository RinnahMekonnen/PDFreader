# importing required modules
import PyPDF2
import pandas as pd
import nltk
import ssl
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Controlling error in relation to downloading stopwords and wordnet
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('wordnet')

# Training data variables
v_output_train = []
lemmatized_train = []
train_data_all = []
y = []

# Testing data variables
all_words = []
v_output = []
lemmatized = []

# Pre-process training data 
training_data = pd.read_csv("train_dataset_short.csv")
training_data.pop('id')
print("\n--------------------------------------------Training Data Dataframe--------------------------------------------")
print(training_data)

training_abstracts = training_data.iloc[:, 0]  # returns only training data abstracts
training_abstracts = training_abstracts.to_numpy().tolist()  # convert to numpy array

for abstract in training_abstracts:
    tokenized_train = [w for w in word_tokenize(abstract)]
    
    # Remove numbers and symbols
    no_symbols_train = [x for x in tokenized_train if (x.isalnum())]
    no_integers_train = [x for x in no_symbols_train if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

    # remove stop words
    stop_words = stopwords.words("english")
    no_stopwords_train = [w for w in no_integers_train if w not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    # Run through verb lemmatizer
    for word in no_stopwords_train:
        output_train = lemmatizer.lemmatize(word, pos='v')
        v_output_train.append(output_train)

    # After verb lemmatizer, run through again with noun lemmatizer
    for word in v_output_train:
        output_n_train = lemmatizer.lemmatize(word, pos='n')
        lemmatized_train.append(output_n_train)
    item = ' '.join(lemmatized_train)

    train_data_all.append(item)

# Build a count vectorizer and extract term counts
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(train_data_all)  # generates word counts for the words in the training data

# Create dataframe with which provides words matrix
word_matrix = pd.DataFrame(train_tc.toarray(), columns=count_vectorizer.get_feature_names_out())
print("\n--------------------------------------------Training Data Word Matrix--------------------------------------------")
print(word_matrix)

# Create the tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)

y_training_data = training_data.iloc[:, 1:5]

# Label Encoding 0=computer science, 1=math, 2=physics, 3=none
for i in range(len(y_training_data)):
    if y_training_data.loc[i, 'Computer Science']:
        y.append(0)
    elif y_training_data.loc[i, 'Mathematics']:
        y.append(1)
    elif y_training_data.loc[i, 'Physics']:
        y.append(2)
    else:
        y.append(3)

# Pre-process testing data 
# Creating a pdf file object
file = open('report.pdf', 'rb')  # rb = open in binary format for reading

# Creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(file)

number_of_pages = pdfReader.getNumPages() # number of pages

for i in range(number_of_pages):
    page = pdfReader.pages[i]
    page_content = page.extract_text()
    all_words = all_words + word_tokenize(page_content)  # Tokenization

# Remove numbers and symbols
no_symbols = [x for x in all_words if (x.isalnum())]
no_integers = [x for x in no_symbols if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

# Remove stop words
no_stopwords = [w for w in no_integers if w not in stop_words]

# Run through verb lemmatizer
for word in no_stopwords:
    output = lemmatizer.lemmatize(word, pos='v')
    v_output.append(output)

# after verb lemmatizer, run through again with noun lemmatizer
for word in v_output:
    output_n = lemmatizer.lemmatize(word, pos='n')
    lemmatized.append(output_n)

item = ' '.join(lemmatized)

# Build a count vectorizer and extract term counts
test_tc = count_vectorizer.transform([item])

# Create dataframe with which provides words matrix
word_matrix = pd.DataFrame(test_tc.toarray(), columns=count_vectorizer.get_feature_names_out())
print("\n--------------------------------------------Testing Data Word Matrix--------------------------------------------")
print(word_matrix)

# Use tf-idf transformer
input_tfidf = tfidf.transform(test_tc)


# Predicts the topic of the paper
def classify(X_train, y_train, X):
    # Train a Multinomial Naive Bayes classifier
    classifier = MultinomialNB()

    # Train the classifier
    classifier.fit(X_train, y_train) # X is the abstract and y is the topic of the paper (both expressed numerical form)

    # Predict the output categories
    predictions = classifier.predict(X)
    return predictions


# Provides a summary of the paper
def summarize():
    summary = []

    # Get text from abstract
    page = pdfReader.pages[0]
    page_text = page.extract_text()
    formatted = page_text.lower()
    start = formatted.find("abstract")
    end = formatted.find("i. ")
    abstract = formatted[start:end]

    # Sumy summarizer
    LANGUAGE = "english"
    SENTENCES_COUNT = 5

    parser = PlaintextParser.from_string(abstract, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        summary.append(sentence)
    
    return summary

# Determines the textual name of the numerical predicted topic
def topic_name(topic_num):
    if topic_num == 0:
        return 'Computer Science'
    elif topic_num == 1:
        return 'Mathematics'
    elif topic_num == 2: 
        return 'Physics'
    else:
        return 'Other'


if __name__ == "__main__":
    topic = classify(train_tfidf,y, input_tfidf)
    topic_text = topic_name(topic[0])
    
    print("\n--------------------------------------------PDF Topic--------------------------------------------")
    print(str(topic[0]) + ' : ' + topic_text)


    summary = summarize()

    print("\n--------------------------------------------PDF Summary--------------------------------------------")

    for sentence in summary:
        print(sentence)
