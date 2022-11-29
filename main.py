# install this via terminal
# pip3 install PyPDF2

# importing required modules

import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download()

import PyPDF2
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

all_words = []
all_sentences = []
no_stopwords = []
v_output = []
lemmatized = []
stemmed = []

#remove & order id's in train_dataset.csv
data = pd.read_csv('train_dataset.csv')
data.pop('id')
print(data)

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
nltk.download('stopwords')
stop_words = stopwords.words("english")
no_stopwords = [w for w in no_integers if w not in stop_words]
print(no_stopwords)

# Stemming
# ps = PorterStemmer()
# for w in no_stopwords:
#     output = ps.stem(w)
#     stemmed.append(output)
#     print(w, " : ", ps.stem(w))

# Lemmatization
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# run through verb lemmatizer
for word in no_stopwords:
    output = lemmatizer.lemmatize(word, pos='v')
    v_output.append(output)
    # print(word, output_v)

# after verb lemmatizer, run through again with noun lemmatizer
for word in v_output:
    output_n = lemmatizer.lemmatize(word, pos='n')
    lemmatized.append(output_n)
    print(word, output_n)

# Build a count vectorizer and extract term counts
count_vectorizer = CountVectorizer()  # This counts the # of words
train_tc = count_vectorizer.fit_transform(lemmatized)  # generates word counts for the words in your docs
#print(train_tc)

#create dataframe which gives us words
word_matrix=pd.DataFrame(train_tc.toarray(),columns=count_vectorizer.get_feature_names_out())
print(word_matrix)

# Create the tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)

  
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