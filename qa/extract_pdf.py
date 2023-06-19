import docx2txt
from PyPDF2 import PdfReader

from dsx_genai import Embeddings
import json
import ast

from utils import create_embeddings_for_text
from config import *


from langchain.embeddings import OpenAIEmbeddings
model = OpenAIEmbeddings(openai_api_key="your-api-key")

#filename='sandra_wise.pdf'
filename = 'Boeing-2022-Annual-Report.pdf'

reader = PdfReader(filename)
extracted_text = ""
for page in reader.pages:
    extracted_text += page.extract_text()
print('done')

# with open("./acc_clinton.txt") as f:
#     txt = f.read()
# print('done')

text_plus_embeddings = create_embeddings_for_text(model, extracted_text)
#print(text_plus_embeddings)
docs=[]
for i, (text_chunk, embedding) in enumerate(text_plus_embeddings):
    doc ={}
    doc['filename']='boing_10k'
    doc['chunk_index']=i
    doc['text_chunk']=text_chunk
    doc['embeddings']=embedding
    docs.append(doc)
    print('done chunking')

with open('docs_boeing.json','w') as docs_json:
    json.dump(docs, docs_json)