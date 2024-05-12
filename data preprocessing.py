import os
import json
import PyPDF2
import re
from cleantext import clean  # Importing necessary libraries

def clean_text(text):
    # Remove special characters and extra spaces
    cleaned_text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove special characters
    cleaned_text.strip()  # Strip leading and trailing spaces
    cleaned_text = clean(cleaned_text, no_line_breaks=True, no_urls=True, no_emails=True, no_phone_numbers=True, no_currency_symbols=True, no_punct=True, replace_with_punct='', replace_with_url='', replace_with_email='', replace_with_phone_number='', replace_with_currency_symbol='', lang='en')
    return cleaned_text

def extract_text_from_pdf(pdf_path):
    # Function to extract text from PDF
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return clean_text(text)  # Clean the extracted text

def extract_abstract_and_references(text):
    # Function to extract abstract and references from text
    abstract_match = re.search(r'Abstract', text, re.IGNORECASE)
    references_match = re.search(r'References', text,re.IGNORECASE)

    if abstract_match and references_match:
        abstract_start = abstract_match.start()
        references_end = text.rfind('REFERENCES')

        abstract_text = (text[abstract_start:references_end]).lower()
        return abstract_text
    elif abstract_match:
        abstract_start = abstract_match.start()

        abstract_text = (text[abstract_start:]).lower()
        return abstract_text
    else:
        return (text).lower()


def process_pdf(pdf_path):
    # Function to process PDF file
    text = extract_text_from_pdf(pdf_path)
    extracted_text = extract_abstract_and_references(text)
    return extracted_text

def main():
    # Main function
    pdf_directory = 'D:\University\8th semester\Data Science\Assignment no 5\Research papers/'
    # Directory where PDF files are stored
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    # List of PDF files in the directory

    data = []
    # List to store extracted text
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        # Constructing full path to the PDF file
        extracted_text = process_pdf(pdf_path)
        # Extracting text from the PDF
        data.append(extracted_text)
        # Appending extracted text to the data list

    with open('extracted_text.json', 'w') as json_file:
        # Writing extracted text to a JSON file
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    main()
    # Calling the main function when the script is executed
