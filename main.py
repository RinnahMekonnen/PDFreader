# install this via terminal
# pip3 install PyPDF2

# importing required modules
import PyPDF2
  
# creating a pdf file object
file = open('example.pdf', 'rb')
  
# creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(file)
  
# printing number of pages in pdf file
#print(file.numPages())

with open("example.pdf", "rb") as pdf_file:
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()
    print(number_of_pages)
    page = read_pdf.pages[0]
    print(page)
    page_content = page.extractText()
    print(page_content)
#print(file.)

  
# creating a page object
page = pdfReader.getPage(0)
  
# extracting text from page
print("HERE")
print(page.extractText())
  
# closing the pdf file object
file.close()

if __name__ == "__main__":
    print("Hello World")