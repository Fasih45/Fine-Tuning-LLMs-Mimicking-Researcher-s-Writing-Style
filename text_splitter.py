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

# Initialize semantic text splitter
# semantic_text_splitter = AI21SemanticTextSplitter(chunk_size=1000)

# Initialize dictionary to store chunks for each text item
chunks_dict = []

# Split text into chunks and store them in dictionary
for index, text_item in enumerate(data):
    # if len(text_item)>100000-2000:
    chunks=text_splitter.split_text(text_item)
    chunks_dict.extend(chunks)
    # for chunk in chunks:
    #     chunks_dict.extend(semantic_text_splitter.split_text(chunk))

    # else:
    #     chunks = semantic_text_splitter.split_text(text_item)
    #     chunks_dict.extend(chunks)

# Save chunks to a new JSON file
with open('output_chunks.json', 'w') as f:
    json.dump(chunks_dict, f, indent=4)

print("Chunks have been saved to 'output_chunks.json'.")
