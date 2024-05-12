# Introduction to Fine-Tuning LLMs

Fine-tuning Language Models (LLMs) has become a crucial technique in Natural Language Processing (NLP) tasks, allowing models to adapt to specific domains or tasks with improved performance. In this blog, we’ll explore the process of fine-tuning LLMs using Python, focusing on techniques to preprocess text data efficiently.

Model Link: https://huggingface.co/Fasih44/Researcher_GPT/tree/main

# Initial Attempt

The pursuit commenced with the aspiration to train models using unstructured data sourced from diverse documents, including research papers and articles. Leveraging advanced techniques and tools, the objective was to partition the data, extract pertinent text elements, and train language models for generating coherent and contextually relevant output.

**The Approach: Leveraging Advanced Tools**
```from IPython.display import JSON
import json
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

# Initialize UnstructuredClient
s = UnstructuredClient(api_key_auth="Api_Key")

# Specify file path
filename = "/content/paper_1.pdf"

# Read file contents
with open(filename, "rb") as f:
    files=shared.Files(
        content=f.read(),
        file_name=filename,
    )

# Define partition parameters
req = shared.PartitionParameters(
    files=files,
    strategy='hi_res',
    pdf_infer_table_structure=True,
    languages=["eng"],
)

# Perform partitioning
try:
    resp = s.general.partition(req)
    print(json.dumps(resp.elements[:3], indent=2))
except SDKError as e:
    print(e)

# Extract titles from partitioned elements
[x for x in resp.elements if x['type'] == 'Title']
```
Through meticulous configuration, model selection, and parameter fine-tuning, I aimed to optimize the training process and enhance model performance, thereby ensuring effective text generation capabilities.

**The Roadblock: Model Acceptance and Output Coherence**
```
# Extract narrative text elements
Title =[]
for x in resp.elements:
   if x['type'] == 'Title':
    Title.append(x['text'])

# Find narrative texts associated with titles
Title_ids = {}
for element in resp.elements:
  for title in Title:
        if element["type"] == Title:
            Title_ids[element["element_id"]] = Title
            break

# Find narrative text elements
narrative_texts = [x for x in resp.elements if x["metadata"].get("parent_id") == title_id and x["type"] == "NarrativeText"]

# Print narrative text content
for element in narrative_texts:
    print(element)
```
Despite my endeavors, a significant roadblock emerged — the trained models encountered difficulty accepting the given file structure and producing coherent output. Despite meticulous configuration and parameter adjustments, the generated neutral text often lacked context and coherence, rendering it unsuitable for practical applications.

Thus I selected a new approach where i just provided the normal data and trained the model on that.

# Preprocessing Text Data
**Cleaning Text Data**

Before fine-tuning an LLM, it’s essential to preprocess the text data to ensure consistency and remove noise. The provided code snippet demonstrates a comprehensive approach to cleaning text data:
```
import re
from cleantext import clean

def clean_text(text):
    # Remove special characters and extra spaces
    cleaned_text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove special characters
    cleaned_text.strip()
    cleaned_text = clean(cleaned_text, no_line_breaks=True, no_urls=True, no_emails=True, no_phone_numbers=True, no_currency_symbols=True, no_punct=True, replace_with_punct='', replace_with_url='', replace_with_email='', replace_with_phone_number='', replace_with_currency_symbol='', lang='en')
    return  cleaned_text # Strip leading and trailing spaces
```
The clean_text() function utilizes regular expressions and the cleantext library to remove special characters, URLs, emails, and other noisy elements from the text, ensuring a clean input for the LLM.

# Extracting Text from PDFs
**Parsing PDF Documents**

PDF documents often contain valuable text data, but extracting this data can be challenging. The following function extract_text_from_pdf() efficiently extracts text from PDF files using PyPDF2:
```
import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return clean_text(text)
```
This function iterates through each page of the PDF document, extracts the text, and cleans it using the previously defined clean_text() function.

# Fine-Tuning Process
**Extracting Relevant Sections**

Fine-tuning LLMs often requires focusing on specific sections of text, such as abstracts and references. The extract_abstract_and_references() function efficiently isolates these sections:
```
import re

def extract_abstract_and_references(text):
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
```

# Simplifying Text Using LangChain

In this section, we’ll explore a Python script that leverages LangChain, a powerful language model (LLM), to simplify text. Let’s delve into the code and understand its functionality.

**Importing Necessary Modules**

The script starts by importing required modules, including Together from langchain_together, PromptTemplate from langchain.prompts, json, and asyncio. These modules enable text simplification using LangChain.

**Defining Example Texts**

The script defines example texts, each containing a raw text and its corresponding neutral version. These examples serve as input for the LangChain model to learn and generate simplified versions.

**Setting Up Prompts**

The prompt variable defines a prompt template instructing LangChain to simplify the given text. It specifies the input variable text, which will be replaced with the actual text during processing.

**Initializing LangChain**

LangChain is initialized using the Together class, specifying the model to be used (mistralai/Mistral-7B-Instruct-v0.2), temperature, top-k value, and API key for LangChain.

**Processing Texts**

The script reads input text chunks from a JSON file, creates prompts for each chunk, and sends batches of prompts to LangChain for simplification. LangChain generates simplified versions of the input texts, which are then stored in a list.
```
from langchain_together import Together
from langchain.prompts import PromptTemplate
import json
import asyncio
from langchain_core.prompts.few_shot import FewShotPromptTemplate
import pprint


example_prompt = PromptTemplate(
    input_variables=["Raw_text", "Neutral_text"], template="Raw_text: {Raw_text}\nNeutral_text:{Neutral_text}"
)

prompt=PromptTemplate(
    template=
    """Covert the given text into neutral text.
Given text:\n{text}""",
    input_variables=["text"]
)

llm = Together(
    # model="meta-llama/Llama-3-70b-chat-hf",
    model="codellama/CodeLlama-70b-Instruct-hf",
    temperature=0.7,
    top_k=1,
    together_api_key="API_Key",
    max_tokens=500
)

with open('./output_chunks.json', 'r') as f:
    data = json.load(f)

all_prompts = []
for chunk in data:
    formatted_prompt = prompt.format(text=chunk)
    # print(formatted_prompt)
    # print("===============================================")
    all_prompts.append(formatted_prompt)

chain=prompt|llm

print(len(data))

count=0

neutral_texts=[]
while len(all_prompts)>count:
    ans=asyncio.run(chain.abatch(all_prompts[count:count+100]))
    for a in ans:
        neutral_texts.append({"Raw_text":data[count],"Neutral_text":a})
        count+=1
    print("Processed ",count," texts")
    with open('neutral_texts.json', 'w') as f:
        json.dump(neutral_texts, f, indent=4)
```

# Efficient Text Chunking with LangChain

In this section, we’ll explore a Python script that efficiently splits text into smaller chunks using LangChain. Let’s break down the code and understand its functionality.

**Importing Necessary Modules**
```
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
```
The script begins by importing essential modules, including os, json, and RecursiveCharacterTextSplitter from langchain.text_splitter. These modules facilitate text chunking using LangChain.

**Initializing Text Splitter**
```
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1050,
    chunk_overlap=70,
    length_function=len,
    is_separator_regex=False,
)
```
A RecursiveCharacterTextSplitter object is initialized to split text into chunks. Parameters such as chunk_size and chunk_overlap are configured to control the chunking process.

**Loading Text Data**
```
# Load text from JSON file
with open('extracted_text.json', 'r') as f:
    data = json.load(f)
```
The script reads text data from a JSON file named ‘extracted_text.json’. This data likely contains large text passages that need to be split into smaller, manageable chunks.

**Splitting Text into Chunks**
```
# Initialize dictionary to store chunks for each text item
chunks_dict = []

# Split text into chunks and store them in dictionary
for index, text_item in enumerate(data):
    chunks = text_splitter.split_text(text_item)
    chunks_dict.extend(chunks)
```
Text items from the loaded data are iterated over, and each item is split into smaller chunks using the initialized text splitter. The resulting chunks are stored in a dictionary named chunks_dict.

**Saving Chunks to JSON**
```
# Save chunks to a new JSON file
with open('output_chunks.json', 'w') as f:
    json.dump(chunks_dict, f, indent=4)
```
Finally, the script saves the generated chunks to a new JSON file named ‘output_chunks.json’. This file serves as the output containing the split text chunks, ready for further processing or analysis.


# Running Tiny Llama: A Step-by-Step Guide

In this section, we’ll walk through the process of setting up and running the Llama Factory using code snippets from a Google Colab notebook. Let’s break down each step and understand how to execute Tiny Llama effectively for Fine-Tuning.
**Step 1: Text Chunking with LangChain**
```
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1050,
    chunk_overlap=70,
    length_function=len,
    is_separator_regex=False,
)

# Load text from JSON file
with open('extracted_text.json', 'r') as f:
    data = json.load(f)

# Initialize dictionary to store chunks for each text item
chunks_dict = []

# Split text into chunks and store them in dictionary
for index, text_item in enumerate(data):
    chunks = text_splitter.split_text(text_item)
    chunks_dict.extend(chunks)

# Save chunks to a new JSON file
with open('output_chunks.json', 'w') as f:
    json.dump(chunks_dict, f, indent=4)

print("Chunks have been saved to 'output_chunks.json'.")
```
This snippet demonstrates how to use LangChain to split large text passages into smaller, manageable chunks, which is a crucial preprocessing step before feeding data into our Tiny Llama Model.

**Step 2: Installing Required Packages**
```
!pip install bitsandbytes
```
This command installs the necessary Python package bitsandbytes, which is required for interacting with the Llama Factory.

**Step 3: Verifying GPU Availability**
```
import torch
assert torch.cuda.is_available() is True
```
It’s essential to ensure that a GPU is available for running the Llama Factory efficiently. This assertion verifies GPU availability.

**Step 4: Logging into Hugging Face**
```
!huggingface-cli login --token Hugging_Face_Token
```
This command logs into Hugging Face using the provided token, enabling access to models and resources required by the Llama Factory.

**Step 5: Installing TensorRT**
```
!pip install tensorrt
```
TensorRT is a high-performance deep learning inference library. Installing it is necessary for optimizing model inference performance.

**Step 6: Running the Llama Factory Web UI**
```
!CUDA_VISIBLE_DEVICES=0 llamafactory-cli webui
```
This command launches the Llama Factory’s web user interface (UI), allowing users to interact with the factory and perform various text-related tasks.

**Step 7: Exporting Environment Variables**
```
!export GRADIO_SHARE=False
```
Exporting environment variables ensures proper configuration for running the Llama Factory’s web interface.

**Step 8: Training the Web Interface**
```
!CUDA_VISIBLE_DEVICES=0 python src/train_web.py
```
This command initiates the training process for the Llama Factory’s web interface, enabling customization and optimization for specific use cases.

**For further information checkout my blog at https://medium.com/@fasihahmad44/a-comprehensive-guide-on-fine-tuning-llms-be99c89a97a0**
