import pickle
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI()



with open('boeing.pkl', 'rb') as file:
    texts, embeddings = pickle.load(file)

print(len(texts), embeddings.shape)


def get_embed(text):
    return client.embeddings.create(input = [text],  model = 'text-embedding-3-small').data[0].embedding


#question = "What was total revenue of boeing in 2022, 2021, 2020 respectively?"
#question = 'Tell me about top 5 objectives of Boeing in bullet points'
#question = "Tell me about IT infra of boeing"
question = 'Tell me about subsidiaries or segments for  boeing company?'
#question = ' Tell me about revenue of Defense, Space & Security (BDS)'
#question = 'Tell me about backlogs of boeing in 2022 , 2021 , 2020 , respectively'ArithmeticError

query = get_embed(question)
query = np.array(query)
query = query/ np.sqrt((query**2).sum())
similarities = embeddings.dot(np.transpose(query))
sorted_ix = np.argsort(-similarities)


files_string = ''
files_string=files_string+'###\n' + texts[sorted_ix[0]] +'###\n' + texts[sorted_ix[2]] +'###\n' + texts[sorted_ix[4]] + texts[sorted_ix[3]] + texts[sorted_ix[1]]
#print(files_string)


completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant that answers questions based on the context provided."},
    {"role": "user", "content": f'''
    Provide me the answer to the given question: {question} based on the text provided below
    {files_string}.
    Don't give any answer if the given text/context doesn't have necessary information.
    Report numbers if any in millions.
    '''}
  ]
)

print(completion.choices[0].message.content)
























# # Wired: use an SVM


# from sklearn import svm

# # create the "Dataset"
# x = np.concatenate([query[None,...], embeddings]) # x is (num_docs, embeddings) array, with query now as the first row
# y = np.zeros(len(x))
# y[0] = 1 # we have a single positive example, mark it as such

# # train our (Exemplar) SVM
# # docs: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

# clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.6)
# clf.fit(x, y) # train

# # infer on whatever data you wish, e.g. the original data
# similarities = clf.decision_function(x)
# sorted_ix = np.argsort(-similarities)
# for k in sorted_ix[:10]:
#   print(f"row {k}, similarity {similarities[k]}")





# #******************************************************svm approach*************************************************************************


# files_string = ''
# for k in sorted_ix[1:3]:
#     files_string=files_string+'###\n'
#     files_string=files_string+docs[k-1]['text_chunk']+'\n'

# comment above part and uncomment below part if you want to just use cosine similarity. 
# cosine similarity is much more scalable while SVM yields better result on diverse data.

# much more context by G.O.A.T. Andrej karpathy in below link.

# https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb


##########################################cosine similarity*********************************************************************


# similarities = embeddings.dot(np.transpose(query))
# sorted_ix = np.argsort(-similarities)


