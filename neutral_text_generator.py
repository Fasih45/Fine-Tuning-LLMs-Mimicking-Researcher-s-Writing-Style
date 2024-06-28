from langchain_together import Together  # Importing Together class for language chaining
from langchain.prompts import PromptTemplate  # Importing PromptTemplate for creating prompts
import json  # Importing JSON module for working with JSON data
import asyncio  # Importing asyncio for asynchronous operations
from langchain_core.prompts.few_shot import FewShotPromptTemplate  # Importing FewShotPromptTemplate for few-shot learning
import pprint  # Importing pprint for pretty-printing data



# Prompt template for text simplification
prompt = PromptTemplate(
    template=
    """Convert the given text into . Your output should only be the simplified text nothing else.\n\n
Given text:\n{text}""",
    input_variables=["text"]
)

# Creating a Together instance for language chaining
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",  # Model for language processing
    temperature=0.5,  # Temperature parameter for generation
    top_k=1,  # Top-k sampling parameter
    together_api_key="API_KEY",  # API key for using Together
    max_tokens=300  # Maximum tokens for generation
)

# Reading data from a JSON file
with open('./output_chunks.json', 'r') as f:
    data = json.load(f)

# Creating prompts for each chunk of text
all_prompts = []
for chunk in data:
    formatted_prompt = prompt.format(text=chunk)
    all_prompts.append(formatted_prompt)

# Chaining prompt and language model
chain = prompt | llm

print(len(data))  # Print number of chunks

count = 0  # Initialize counter
neutral_texts = []  # List to store neutralized texts
while len(all_prompts) > count:
    # Asynchronously running the language chain for each chunk of text
    ans = asyncio.run(chain.abatch(all_prompts[count:count + 100]))
    # Processing each response
    for a in ans:
        neutral_texts.append({"Raw_text": data[count], "Neutral_text": a})

        count += 1  # Increment counter
    print("Processed ", count, " texts")  # Print progress
    # Writing neutralized texts to a JSON file
    with open('neutral_texts.json', 'w') as f:
        json.dump(neutral_texts, f, indent=4)
