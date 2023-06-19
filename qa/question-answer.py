import json
with open('docs_boeing.json','r') as docs_json:
    docs = json.load(docs_json)


import ast
import numpy as np

import os
embeddings = []
for doc in docs:
    embeddings.append(ast.literal_eval(doc['embeddings']))

embeddings = np.array(embeddings)
embeddings = embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True)) # L2 normalize the rows, as is common




from langchain.embeddings import OpenAIEmbeddings
model = OpenAIEmbeddings(openai_api_key="your-api-key")


sentence = "What are future business plans of boeing ?"


query = model.encode(sentence)

query = query/ np.sqrt((query**2).sum())

# Wired: use an SVM


from sklearn import svm

# create the "Dataset"
x = np.concatenate([query[None,...], embeddings]) # x is (num_docs, embeddings) array, with query now as the first row
y = np.zeros(len(x))
y[0] = 1 # we have a single positive example, mark it as such

# train our (Exemplar) SVM
# docs: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.6)
clf.fit(x, y) # train

# infer on whatever data you wish, e.g. the original data
similarities = clf.decision_function(x)
sorted_ix = np.argsort(-similarities)
for k in sorted_ix[:10]:
  print(f"row {k}, similarity {similarities[k]}")





#******************************************************svm approach*************************************************************************


files_string = ''
for k in sorted_ix[1:3]:
    files_string=files_string+'###\n'
    files_string=files_string+docs[k-1]['text_chunk']+'\n'

# comment above part and uncomment below part if you want to just use cosine similarity. 
# cosine similarity is much more scalable while SVM yields better result on diverse data.

# much more context by G.O.A.T. Andrej karpathy in below link.

# https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb


##########################################cosine similarity*********************************************************************


# similarities = embeddings.dot(np.transpose(query))
# sorted_ix = np.argsort(-similarities)


# files_string = ''
# for k in sorted_ix[0:2]:
#     files_string=files_string+'###\n'
#     files_string=files_string+docs[k-1]['text_chunk']+'\n'




prompt =f'''
Human: If I give you some paragraphs, can you use them to answer the question {sentence} ?
Assistant: Sure. I'll keep my answer precise.
Human: Remember to only use information provided in paragraphs. Don't answer questions you are not sure about.
Assistant: Noted. Please provide me the paragraphs.

Human: Here they are {files_string}

Assistant:
'''
#
print(prompt)

#print(sentence, '\n')


from langchain.llms import OpenAI

llm = OpenAI(openai_api_key="...")


print(llm(prompt))
