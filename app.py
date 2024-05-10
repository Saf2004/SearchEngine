import re
import tensorflow as tf
import tensorflow_hub as hub
import nltk
import pandas as pd
import pyterrier as pt
import numpy as np

from flask import Flask, request, render_template
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from xpmir.models import AutoModel

model = AutoModel.load_from_hf_hub("monobert", as_instance=True)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

words_dict = {}
def stem_text(text):
    tokens = word_tokenize(text)
    stemmed_tokens = []
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        stemmed_tokens.append(stemmed_token)
        if stemmed_token not in words_dict:
            words_dict[stemmed_token] = token
    return ' '.join(stemmed_tokens)


def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if
                       word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


def clean(text):
    text = re.sub(r"http\S+", " ", text)  # remove urls
    text = re.sub(r"RT ", " ", text)  # remove rt
    text = re.sub(r"@[\w]*", " ", text)  # remove handles
    text = re.sub(r"[\.\,\#_\|\(\)\[\]\:\<\>\?\?\*\-\=\+\@\#\$\%\^\\/\=]", " ", text)  # remove special characters
    text = re.sub(r'\t', ' ', text)  # remove tabs
    text = re.sub(r'\n', ' ', text)  # remove line jump
    text = re.sub(r"\s+", " ", text)  # remove extra white space
    text = re.sub(r"^\s+|\s+?$", "", text)  # remove leading and trailing white space
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"<\d+>", " ", text)
    text = text.strip()
    return text


def preprocess(sentence,stem=True,stopwords=True):
    sentence = remove_stopwords(sentence)
    sentence = clean(sentence)
    if stem:
        sentence = stem_text(sentence)
    if stopwords:
        sentence = remove_stopwords(sentence)
    return sentence

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

df = pd.read_csv('data/cisi.csv')
df['processed_text'] = df['Text'].apply(preprocess)

df['docno'] = (df['ID']-1).astype(str)
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
indexer = pt.DFIndexer("./DatasetIndex", overwrite=True)

index_ref = indexer.index(df["processed_text"], df["docno"])
index = pt.IndexFactory.of(index_ref)
tfidf_retr = pt.BatchRetrieve(index, controls={"wmodel": "TF_IDF"}, num_results=10)

print("Loading ELMo model...")
elmo = hub.load("/Users/safey/.cache/kagglehub/models/google/elmo/tensorFlow1/elmo/3")
print("ELMo model loaded successfully")

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def homePage():
    if request.method == 'POST':
        query = request.form.get('query')
        original_query = query
        query = preprocess(query)
        results = tfidf_retr.search(query)


        rm3_expander = pt.rewrite.RM3(index, fb_terms=10, fb_docs=100)

        rm3_qe = tfidf_retr >> rm3_expander
        expanded_query = rm3_qe.search(query).iloc[0]["query"]
        expanded = []
        for s in expanded_query.split()[1:]:
            expanded.append(s.split('^')[0])
        reverse_stemmed_expanded = []
        for word in expanded:
            reverse_stemmed_expanded.append(words_dict[word])
        documents = []
        sentences = []

        results = df.merge(results, on='docno')["score", "processed_text"]
        output = model.rsv(original_query, results["processed_text"].values)

        data = [(list(obj.document.items.values())[0].text, obj.score) for obj in output]
        reviews_result_v2 = pd.DataFrame(data, columns=['document', "score"]).sort_values(by="score", ascending=False)
        reviews_result_v2 = df.merge(reviews_result_v2, left_on='processed_text')
        results = reviews_result_v2
        for i, result in enumerate(results['docno']):
            document = {'content': df.at[df[df['docno'] == result].index[0], 'Text'][:140]+'...', 'title': df.at[df[df['docno'] == result].index[0], 'Text'][:40]+'.....', 'docno': result}
            documents.append(document)
            sentences.append(df.at[df[df['docno'] == result].index[0], 'Text'])


        # embeddings = elmo.signatures["default"](tf.constant(sentences))["elmo"].numpy()
        # query_embedding = elmo.signatures["default"](tf.constant([original_query]))["elmo"].numpy()
        # cosine_similarities = tf.tensordot(query_embedding, tf.transpose(embeddings), 1)
        # cosine_similarities_np = cosine_similarities.numpy()
        # sorted_indices = np.argsort(cosine_similarities_np[0])[::-1]
        # print(sorted_indices.shape)



        return render_template('resultPage.html', documents=documents, expanded=reverse_stemmed_expanded, query=original_query)
    else:
        return render_template('homePage.html', documents=None, expanded=None, query=None)
@app.route('/search/<query>', methods=['GET', 'POST'])
def search(query):
    query = preprocess(query)
    original_query = query
    results = tfidf_retr.search(query)
    print(results)
    documents = []
    for i, result in enumerate(results['docno']):
        document = {'content': df.at[df[df['docno'] == result].index[0], 'Text'],
                    'title': df.at[df[df['docno'] == result].index[0], 'Text'][:40], 'docno': result['docno']}
        documents.append(document)

    rm3_expander = pt.rewrite.RM3(index, fb_terms=10, fb_docs=100)
    rm3_qe = tfidf_retr >> rm3_expander

    expanded_query = rm3_qe.search(preprocess(query)).iloc[0]["query"]
    expanded = [s.split('^')[0] for s in expanded_query.split()[1:]]
    reverse_stemmed_expanded = []
    for word in expanded:
        reverse_stemmed_expanded.append(words_dict[word])
    embeddings = elmo.signatures["default"](tf.constant([documents[i]['content'] for i in range(len(documents))]))[
        "elmo"]
    query_embedding = elmo.signatures["default"](tf.constant(original_query))["elmo"]
    print(query_embedding)
    print(embeddings)
    cosine_similarities = tf.tensordot(query_embedding, embeddings)
    cosine_similarities_np = cosine_similarities.numpy()
    top_indices = np.argsort(cosine_similarities_np)[0][-5:][::-1]
    top_5_similar_words = [documents[i]['original_word'] for i in top_indices]
    reverse_stemmed_expanded = reverse_stemmed_expanded + top_5_similar_words
    return render_template('resultPage.html', documents=documents, expanded=reverse_stemmed_expanded, query=original_query)
@app.route('/results')
def results():
        return render_template('resultPage.html', documents=request.args.get('documents'))

@app.route('/view/<docno>', methods=['GET', 'POST'])
def view(docno):
    document = df[df['docno'] == docno].iloc[0]
    document = {'content': document['Text'], 'title': document['Text'][:40],'docno': docno}
    return render_template('viewPage.html', document=document)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
