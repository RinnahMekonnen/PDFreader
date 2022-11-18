# install this via terminal
# pip3 install PyPDF2
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

# importing required modules
import PyPDF2
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

all_words_train = []
v_output_train = []
lemmatized_train = []

tokenized_train =[]

train_data = []
y_train_data = []

y = []

all_words = []
all_sentences =[]
no_stopwords = []
v_output = []
lemmatized = []
stemmed = []

# Pre-process training data 
training_data = pd.read_csv("train_dataset_short.csv")
training_data.pop('id')
#print(training_data)

training_abstracts = training_data.iloc[:,0] #returns only training data abstracts
training_abstracts = training_abstracts.to_numpy().tolist() #convert to numpy array
#training_abstarcts = training_abstracts['ABSTRACT'].tolist() #convert to list
print(training_abstracts)
for abstract in training_abstracts:
    tokenized_train = word_tokenize(abstract)
    # remove numbers and symbols
    no_symbols_train = [x for x in tokenized_train if (x.isalnum())]
    no_integers_train = [x for x in tokenized_train if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

    # remove stop words
    nltk.download('stopwords')
    stop_words = stopwords.words("english")
    no_stopwords_train = [w for w in no_integers_train if w not in stop_words]

    # Lemmatization
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

    # run through verb lemmatizer
    for word in no_stopwords_train:
        output_train = lemmatizer.lemmatize(word, pos='v')
        v_output_train.append(output_train)

    # after verb lemmatizer, run through again with noun lemmatizer
    for word in v_output_train:
        output_n_train = lemmatizer.lemmatize(word, pos='n')
        lemmatized_train.append(output_n_train)
print(lemmatized_train)

# Build a count vectorizer and extract term counts
count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.1, max_df=0.7, max_features=100)
train_tc = count_vectorizer.fit_transform(training_abstracts)  # generates word counts for the words in your docs
#print(train_tc)

#create dataframe which gives us words
word_matrix=pd.DataFrame(train_tc.toarray(),columns=count_vectorizer.get_feature_names_out())
#print(word_matrix)

# Create the tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)

print(train_tfidf)

# for abstract in training_abstracts:
#     #print(word_tokenize(abstract))
#     tokenized_train = word_tokenize(abstract)
    
#     # remove numbers and symbols
#     no_symbols_train = [x for x in tokenized_train if (x.isalnum())]
#     no_integers_train = [x for x in tokenized_train if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

#     # remove stop words
#     nltk.download('stopwords')
#     stop_words = stopwords.words("english")
#     no_stopwords_train = [w for w in no_integers_train if w not in stop_words]

#     # Lemmatization
#     nltk.download('wordnet')
#     lemmatizer = WordNetLemmatizer()

#     # run through verb lemmatizer
#     for word in no_stopwords_train:
#         output_train = lemmatizer.lemmatize(word, pos='v')
#         v_output_train.append(output_train)

#     # after verb lemmatizer, run through again with noun lemmatizer
#     for word in v_output_train:
#         output_n_train = lemmatizer.lemmatize(word, pos='n')
#         lemmatized_train.append(output_n_train)
    
    #y_value = ''
    #y_train_data.clear()
    #y_train_data.append(training_data.iloc[:,1:5].to_numpy().flatten())
    #for x in y_train_data:
    #    print(x)
    #    y_value = y_value + x
    
    #y.append(int(y_value))

y_training_data= training_data.iloc[:,1:5]
i = 0;

# label encoding 0=computer science, 1=math, 2=physics, 3=none
for i in range(len(y_training_data)):
    if y_training_data.loc[i,'Computer Science']:
        y.append(0)
    elif y_training_data.loc[i,'Mathematics']:
        y.append(1)
    elif y_training_data.loc[i,'Physics']:
        y.append(2)
    else:
        y.append(3)
        



#train_data = train_data.append(train_tfidf);
#print(train_data)
    
    #all_words_train.append([word_tokenize(abstract)]) #Tokenization- each abastract is an array within the array all_words_train
# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
#y_train =  training_data.iloc[:,1:5].to_numpy().flatten()
classifier.fit(train_tfidf, y)

# Transform input data using count vectorizer
#input_tc = count_vectorizer.transform(test_data)

# Transform vectorized data using tfidf transformer
#input_tfidf = tfidf.transform(input_tc)

# Predict the output categories




  
# creating a pdf file object
file = open('report.pdf', 'rb') #rb = open in binary format for reading
  
# creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(file)

# number of pages
number_of_pages = pdfReader.getNumPages()

for i in range(number_of_pages):
    page = pdfReader.pages[i]
    page_content = page.extract_text()
    all_sentences.append(sent_tokenize(page_content)) #Tokenization
    #sentences = sent_tokenize(page_content) #sentences is of type list 
    all_words = all_words + word_tokenize(page_content) #Tokenization
   
    
    #words = word_tokenize(page_content)
    #print(type(page_content))
    #print('page ' + str(i))
    #print(page_content)

# remove hyphen from words spanning 2 lines (this doesn't work yet)
# hyphenated = re.findall(r'\w+(?:-\w+)+',all_words)

# remove numbers and symbols
no_symbols = [x for x in all_words if (x.isalnum())]
no_integers = [x for x in no_symbols if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

# remove stop words
#nltk.download('stopwords')
#stop_words = stopwords.words("english")
no_stopwords = [w for w in no_integers if w not in stop_words]
#print(no_stopwords)

# Stemming
# ps = PorterStemmer()
# for w in no_stopwords:
#     output = ps.stem(w)
#     stemmed.append(output)
#     print(w, " : ", ps.stem(w))

# Lemmatization
#nltk.download('wordnet')
#lemmatizer = WordNetLemmatizer()

# run through verb lemmatizer
for word in no_stopwords:
    output = lemmatizer.lemmatize(word, pos='v')
    v_output.append(output)
    # print(word, output_v)

# after verb lemmatizer, run through again with noun lemmatizer
for word in v_output:
    output_n = lemmatizer.lemmatize(word, pos='n')
    lemmatized.append(output_n)
    #print(word, output_n)

# Build a count vectorizer and extract term counts
#count_vectorizer = CountVectorizer()  # This counts the # of words
train_tc = count_vectorizer.transform(lemmatized)  # generates word counts for the words in your docs
#print(train_tc)

#create dataframe which gives us words
word_matrix=pd.DataFrame(train_tc.toarray(),columns=count_vectorizer.get_feature_names_out())
#print(word_matrix)

# Create the tf-idf transformer
#tfidf = TfidfTransformer()
input_tfidf = tfidf.transform(train_tc)

# Predict the output categories
predictions = classifier.predict(input_tfidf)

print(predictions)

  
# printing number of pages in
#  pdf file
#print(file.numPages())

# with open("example.pdf", "rb") as pdf_file:
#     read_pdf = PyPDF2.PdfFileReader(pdf_file)
#     number_of_pages = read_pdf.getNumPages()
#     print(number_of_pages)
#     page = read_pdf.pages[0]
#     print(page)
#     page_content = page.extractText()
#     print(page_content)
#print(file.)

  
# creating a page object
# page = pdfReader.getPage(0)
  
# # extracting text from page
# print("HERE")
# print(page.extractText())
# print(pdfReader.getPage(1).extract_text())
  
# # closing the pdf file object
# file.close()

if __name__ == "__main__":
    print("Hello World")
