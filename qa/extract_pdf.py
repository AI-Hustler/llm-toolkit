import json
import os
import time
import pandas as pd
import pickle
import numpy as np 
from dotenv import load_dotenv
from functools import wraps

from openai import OpenAI
import tiktoken
import pdfplumber
from utils import timed
import time


load_dotenv()


client = OpenAI()
pdf_path = 'Boeing-2022-Annual-Report.pdf'
encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

chunk_size = 300


@timed
def extract_pdf(pdf_path):
    all_text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + '\n'
    return all_text


def calc_tokens(text: str) -> int:
    return len(encoding.encode(text))


@timed
def chunker(text):
    window = []
    window_size = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line_size = calc_tokens(line)
        # grow window until largest possible chunk
        if window_size + line_size < chunk_size:
            window.append(line)
            window_size += line_size + 1
        # reset window
        else:
            if window:
                yield "\n".join(window)
            window = [line]
            window_size = line_size
    # return the leftover
    if window:
        yield "\n".join(window)

@timed
def get_embed(text):
    return client.embeddings.create(input = [text],  model = 'text-embedding-3-small').data[0].embedding



@timed
def run_main(pdf_path):

    
    text_content = extract_pdf(pdf_path)
    print('pdf extracted')
    chunks = list(chunker(text_content))
    embeddings_l = list(map(get_embed, chunks))
    print('embeddings generated')

    embeddings = np.array(embeddings_l)
    embeddings = embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True))

    return chunks , embeddings

chunks , embeddings = run_main(pdf_path)

with open('boeing.pkl', 'wb') as file:
    pickle.dump((chunks, embeddings), file)



