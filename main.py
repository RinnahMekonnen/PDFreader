# install this via terminal
# pip3 install PyPDF2

# importing required modules
import PyPDF2
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

all_words = []
all_sentences =[]
  
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
    

#Stemming or Lemmatization NEEDS TO BE DONE



# Build a count vectorizer and extract term counts
count_vectorizer = CountVectorizer()  # This counts the # of words
train_tc = count_vectorizer.fit_transform(all_words)  # generates word counts for the words in your docs
#print(train_tc)

#create dataframe
bow_matrix=pd.DataFrame(train_tc.toarray(),columns=count_vectorizer.get_feature_names_out())
print(bow_matrix)

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