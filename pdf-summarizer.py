import os
from dotenv import load_dotenv

load_dotenv()

from langchain.llms import OpenAI


os.environ["OPENAI_API_KEY"] = "sk-urAPIkey" # u can save it in .env or directly pass key in below object

llm = OpenAI(openai_api_key="...")


import docx2txt
from PyPDF2 import PdfReader

filename =  "./Boeing-2022-Annual-Report.pdf"  #10k report for boeing for FY2022

reader = PdfReader(filename)
docs = ""
for page in reader.pages:
    docs += page.extract_text()

print('data loaded')
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def sum_parallel(text):
    prompt = f"""
        Your task is to generate a short summary of given text in atmost 100 words

        Summarize the text below, delimited by triple backticks.
        text: ```{text}```
        """


    try:   
        sum_par = llm(prompt)
        print('done')
    except:
        sum_par = llm(prompt)
        print('done in next try')


    return sum_par

# a very simple implementation of chunking pdf as whole pdf cannot be summarized in one go.

def text_chunking(docs):

    i=0
    j =5000
    
    text_l = []
    for t in range(0, len(docs)+1, 5000):
    
            
            text = docs[i:j]
            text_l.append(text)
            i = j
            j = j+ 5000
            if j > len(docs)+1:
                j = len(docs)+1
            else:
                j = j
    return text_l


# below approach of chunking is much better and taken from this notebook
# https://colab.research.google.com/drive/1Icoxgd2IJAjMU2fyD-MET5a706HGpwVg

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
model_max_tokens = 4000 # gpt-3's token limit
max_workers = 4 # max parallel API calls

# the output size you want
safety_buffer = 128
output_size = 1000

def calc_tokens(text: str) -> int:
    return len(encoding.encode(text))
t = ""
prompt = f"""
        Your task is to generate a short summary of given text in atmost 100 words

        Summarize the text below, delimited by triple backticks.
        text: '''{t}'''
        """



prompt_size = calc_tokens(prompt) + safety_buffer
# to merge 2 outputs, we need to have at least 1/3 of the max tokens available
max_output_size = (model_max_tokens - prompt_size) // 3
output_size = max_output_size
# total tokens = prompt + chunk + output
chunk_size = model_max_tokens - (prompt_size + output_size)
print('Output size:', output_size, '\nChunk Size:', chunk_size)



import concurrent.futures
import time
import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor



# time parameters to be defined so that number of tokens within a minute don't exceed their limits.
max_workers = 6
ct = mp.cpu_count()
start_time = time.perf_counter()
while len(encoding.encode(docs)) > 2000:
    print('started', '\n')
    if __name__ == "__main__":
        text_l = text_chunking(docs)

        chunks = list(chunker(docs))
        print('Chunks:', list(map(calc_tokens, chunks)))


        print(len(chunks))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            
            result = list(executor.map(sum_parallel, text_l))
        
       

       # in case you want to run the process n loop, will take a lot of time.
        # result = []
        # i =0

        # for chunk in chunks:
        #     temp = sum_parallel(chunk)
        #     result.append(temp)
        #     i=i+1
        #     print("chunk: ", i, " done")




       
        docs = " ".join([r for r in result])
        print(len(docs), ' done')

prompt = f"""
        Your task is to generate a short summary of given text in atmost 100 words

        Summarize the text below, delimited by triple backticks. Remember to include information related to revenue

        text: ```{docs}```
        """
long_summary = docs
short_summary = sum_parallel(docs)

print("long_summary:- ", long_summary, '\n', "short_summary:- ", short_summary)
finish_time = time.perf_counter()
print(f"Program finished in {finish_time-start_time} seconds")
