import re

import nltk
import pandas as pd
import pyterrier as pt
from flask import Flask, request, render_template, redirect

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

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
    text = re.sub(r"<.*?>", " ", text)  # remove html tags
    text = text.strip()
    return text


def preprocess(sentence):
    sentence = remove_stopwords(sentence)
    sentence = clean(sentence)
    sentence = stem_text(sentence)
    return sentence

df = pd.read_csv('data/rec_autos.csv')
df['processed_text'] = df['processed_text'].apply(preprocess)
df['docno'] = df['docno'].apply(str)
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
indexer = pt.DFIndexer("./DatasetIndex", overwrite=True)

index_ref = indexer.index(df["processed_text"], df["docno"])
index = pt.IndexFactory.of(index_ref)
tfidf_retr = pt.BatchRetrieve(index, controls={"wmodel": "TF_IDF"}, num_results=10)
print(df.columns)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def homePage():
    if request.method == 'POST':
        query = request.form.get('query')
        query = preprocess(query)
        results = tfidf_retr.search(query)
        documents = []
        for i, result in enumerate(results['docno']):
            document = {'content': df.at[df[df['docno'] == result].index[0], 'text'][:140]+'...', 'title': df.at[df[df['docno'] == result].index[0], 'subject']}
            documents.append(document)
        rm3_expander = pt.rewrite.RM3(index, fb_terms=10, fb_docs=100)

        rm3_qe = tfidf_retr >> rm3_expander
        expanded_query = rm3_qe.search(query).iloc[0]["query"]
        expanded = []
        for s in expanded_query.split()[1:]:
            expanded.append(s.split('^')[0])
        reverse_stemmed_expanded = []
        for word in expanded:
            reverse_stemmed_expanded.append(words_dict[word])
        original_query = ''
        for word in query.split():
            original_query = original_query + words_dict[word] + ' '
            print(words_dict)
        return render_template('resultPage.html', documents=documents, expanded=reverse_stemmed_expanded, query=original_query)
    else:
        return render_template('homePage.html', documents=None, expanded=None, query=None)
@app.route('/search/<query>', methods=['GET', 'POST'])
def search(query):
    query = preprocess(query)
    results = tfidf_retr.search(query)
    documents = []
    for i, result in enumerate(results['docno']):
        document = {'content': df.at[df[df['docno'] == result].index[0], 'text'],
                    'title': df.at[df[df['docno'] == result].index[0], 'subject'], 'docno': result}
        documents.append(document)

    rm3_expander = pt.rewrite.RM3(index, fb_terms=10, fb_docs=100)
    rm3_qe = tfidf_retr >> rm3_expander

    expanded_query = rm3_qe.search(preprocess(query)).iloc[0]["query"]
    expanded = [s.split('^')[0] for s in expanded_query.split()[1:]]
    reverse_stemmed_expanded = []
    for word in expanded:
        reverse_stemmed_expanded.append(words_dict[word])
    original_query = ''
    for word in query.split():
        original_query = original_query + words_dict[word] + ' '
    return render_template('resultPage.html', documents=documents, expanded=reverse_stemmed_expanded, query=original_query)
@app.route('/results')
def results():
        return render_template('resultPage.html', documents=request.args.get('documents'))

@app.route('/view/<docno>', methods=['GET', 'POST'])
def view(docno):
    document = df[df['docno'] == docno].iloc[0]
    document = {'content': document['text'], 'title': document['subject'],'docno': docno}
    return render_template('viewPage.html', document=document)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
