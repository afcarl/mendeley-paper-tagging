from flask import Flask, redirect, render_template, request, session
import yaml

from mendeley import Mendeley
from mendeley.session import MendeleySession

from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack

with open('config.yml') as f:
    config = yaml.load(f)

REDIRECT_URI = 'http://localhost:5000/oauth'

app = Flask(__name__)
app.debug = True
app.secret_key = config['clientSecret']

mendeley = Mendeley(config['clientId'], config['clientSecret'], REDIRECT_URI)


@app.route('/')
def home():
    # if 'token' in session:
    #     return redirect('/listDocuments')

    auth = mendeley.start_authorization_code_flow()
    session['state'] = auth.state

    return render_template('home.html', login_url=(auth.get_login_url()))


@app.route('/oauth')
def auth_return():
    auth = mendeley.start_authorization_code_flow(state=session['state'])
    mendeley_session = auth.authenticate(request.url)

    session.clear()
    session['token'] = mendeley_session.token

    return redirect('/listDocuments')


@app.route('/listDocuments')
def list_documents():
    if 'token' not in session:
        return redirect('/')

    mendeley_session = get_session_from_cookies()

    name = mendeley_session.profiles.me.display_name
    # for g in mendeley_session.groups.list().items:
    #     if g.name == "DSOARS":

    dsoars_id = "6eb64e0d-8590-3b35-a8dc-eea72b64dd68"
    docs = []
    for doc in mendeley_session.groups.get(dsoars_id).documents.iter(page_size=200,
                                                                     view="all"):
    # for doc in mendeley_session.groups.get(dsoars_id).documents.list(page_size=20,
    #         view="all").items:
        docs.append(doc)

    # DO THE MACHINE LEARNING HERE
    tags = set()
    for doc in docs:
        doc.suggested_tags = []
        doc.human_tags = []
        doc.labeled = False

        if doc.tags is None:
            continue

        for tag in doc.tags:
            if "suggested::" not in tag:
                tags.add(tag)
                doc.labeled = True
                doc.human_tags.append(tag)

    print(tags)

    author_vectorizer = CountVectorizer(analyzer='char_wb',
                                        strip_accents="unicode",
                                        ngram_range=(1, 5))
    title_vectorizer = CountVectorizer(analyzer='char_wb',
                                       strip_accents="unicode",
                                       ngram_range=(1, 5))
    abstract_vectorizer = CountVectorizer(analyzer='char_wb',
                                          strip_accents="unicode",
                                          ngram_range=(1, 5))
    clf1 = LogisticRegression()
    # clf1 = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    # clf1 = SVC()
    params = {'C': np.logspace(-30, 2, 10000)}
    clf = RandomizedSearchCV(clf1, params, cv=2, n_iter=10)

    for tag in tags:
        X_authors = []
        X_titles = []
        X_abstracts = []
        y_train = []
        for doc in docs:
            if doc.labeled is False:
                continue

            if doc.authors is None:
                authors = ""
            else:
                authors = ",".join([p.last_name for p in doc.authors])
            X_authors.append(authors)
            X_titles.append(doc.title)

            X_abstracts.append("" if doc.abstract is None else doc.abstract)

            if doc.tags is not None and tag in doc.tags:
                y_train.append(1)
            else:
                y_train.append(0)

        if len(set(y_train)) <= 1:
            continue
        if sum(y_train) <= 2 or len(y_train) - sum(y_train) <= 2:
            continue

        print("Training classifier for: %s" % tag)

        author_features = author_vectorizer.fit_transform(X_authors)
        title_features = title_vectorizer.fit_transform(X_titles)
        abstract_features = abstract_vectorizer.fit_transform(X_abstracts)

        X_train = hstack([author_features, title_features, abstract_features],
                         format="csr")

        clf.fit(X_train, y_train)

        print(clf.best_params_)
        print(clf.best_score_)

        # clf.fit(X_train[y_train == 1])

        for doc in docs:
            if doc.labeled is True:
                continue

            if doc.authors is None:
                authors = ""
            else:
                authors = ",".join([p.last_name for p in doc.authors])

            x = hstack([author_vectorizer.transform([authors]),
                        title_vectorizer.transform([doc.title]),
                        abstract_vectorizer.transform(["" if doc.abstract is None else doc.abstract])],
                       format="csr")

            if clf.predict(x)[0] == 1:
                doc.suggested_tags.append(tag)

    for i, doc in enumerate(docs):
        if doc.labeled is True:
            continue

        print("Updating %i of %i" % (i, len(docs)))
        if len(doc.suggested_tags) == 0:
            doc.update(tags=["suggested::Untagged"])
        else:
            doc.update(tags=["suggested::" + tag for tag in doc.suggested_tags])

    # END ML

    # print([(g.name, g.id) for g in mendeley_session.groups.list().items])
    # docs = mendeley_session.documents.list(view='client').items

    return render_template('library.html', name=name, docs=docs)


@app.route('/document')
def get_document():
    if 'token' not in session:
        return redirect('/')

    mendeley_session = get_session_from_cookies()

    document_id = request.args.get('document_id')
    doc = mendeley_session.documents.get(document_id, view="tags")

    return render_template('metadata.html', doc=doc)


@app.route('/metadataLookup')
def metadata_lookup():
    if 'token' not in session:
        return redirect('/')

    mendeley_session = get_session_from_cookies()

    doi = request.args.get('doi')
    doc = mendeley_session.catalog.by_identifier(doi=doi)

    return render_template('metadata.html', doc=doc)


@app.route('/download')
def download():
    if 'token' not in session:
        return redirect('/')

    mendeley_session = get_session_from_cookies()

    document_id = request.args.get('document_id')
    doc = mendeley_session.documents.get(document_id)
    doc_file = doc.files.list().items[0]

    return redirect(doc_file.download_url)


@app.route('/logout')
def logout():
    session.pop('token', None)
    return redirect('/')


def get_session_from_cookies():
    return MendeleySession(mendeley, session['token'])


if __name__ == '__main__':
    app.run()
