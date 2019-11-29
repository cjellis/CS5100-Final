import pandas as pd
import urllib
import re
import string
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7554634'
ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7554637'


def download_file(url, fname):
    urllib.request.urlretrieve(url, fname)


# Borrowed from https://stackoverflow.com/questions/39121104/how-to-add-another-feature-length-of-text-to-current-bag-of-words-classificati
def get_length(texts):
    return np.array([len(t) for t in texts]).reshape(-1, 1)


start = datetime.now()

download_file(ANNOTATED_COMMENTS_URL, 'attack_annotated_comments.tsv')
download_file(ANNOTATIONS_URL, 'attack_annotations.tsv')

comments = pd.read_csv('attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('attack_annotations.tsv',  sep = '\t')

len(annotations['rev_id'].unique())


# labels a comment as an atack if the majority of annoatators did so
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5


# join labels and comments
comments['attack'] = labels

# remove newline, tab tokens, and many forms of punctuation
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("'", ""))
# https://stackoverflow.com/questions/34860982/replace-the-punctuation-with-whitespace
comments['comment'] = comments['comment'].apply(lambda x: x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))))
# Collapse multiple spaces to one
comments['comment'] = comments['comment'].apply(lambda x: re.sub(' +', ' ', x))
# Make every string lowercase
comments['comment'] = comments['comment'].apply(lambda x: x.lower())


comments.query('attack')['comment'].head(10)


# fit a simple text classifier

train_comments = comments.query("split=='train'")
test_comments = comments.query("split=='test'")
dev_comments = comments.query("split=='dev'")
non_dev_comments = comments.query("split=='test' or split=='train'")

if __name__ == '__main__':
    parameters = {
        'clf__alpha': [.00000001, .000001, .0001, .01, 1, 100],
        'clf__loss': ['log', 'modified_huber'],
        'clf__penalty': ['l2', 'l1', 'elasticnet'],
        'clf__max_iter': [1000, 2000],
        'clf__n_iter_no_change': [5, 10],
        'clf__class_weight': ['balanced', None],
    }

    word_and_char = FeatureUnion([
        ('vect_word', CountVectorizer(max_features=10000, analyzer='word', ngram_range=(1, 1))),
        ('vect_char', CountVectorizer(max_features=10000, analyzer='char_wb', ngram_range=(4, 4)))        # Borrowed from https://stackoverflow.com/questions/39121104/how-to-add-another-feature-length-of-text-to-current-bag-of-words-classificati
        # ('length', FunctionTransformer(get_length, validate=False))
    ])

    clf = Pipeline([
        ('all', FeatureUnion([
            ('comments', Pipeline([
                ('extract_field', FunctionTransformer(lambda x: x['comment'], validate=False)),
                ('vects', word_and_char),
                ('tfidf', TfidfTransformer(norm='l2'))
            ])),
            ('login', Pipeline([
                ('extract_field', FunctionTransformer(lambda x: x['logged_in'][:, np.newaxis], validate=False)),
                ('encoder', OneHotEncoder())
            ]))
        ])),
        ('clf', SGDClassifier(alpha=.0001, class_weight=None, loss='modified_huber', max_iter=1000, n_iter_no_change=10, penalty='elasticnet', random_state=5))
        # ('clf', RandomForestClassifier())
        # ('clf', MultinomialNB())
        # ('clf', LogisticRegression()),
        # ('clf', MLPClassifier())
    ])
    search = GridSearchCV(clf, parameters, cv=3, verbose=10)

    search.fit(dev_comments['comment'], dev_comments['attack'])

    print("Best Score: %.3f" %search.best_score_)
    print("Best parameters set:")

    best_parameters = search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    kf = KFold(n_splits=5)

    recalls = []
    precisions = []
    fbetas = []
    roc_aucs = []

    for training_data_indices, test_data_indices in kf.split(non_dev_comments):
        training_data = non_dev_comments.iloc[training_data_indices]
        test_data = non_dev_comments.iloc[test_data_indices]

        clf = clf.fit(training_data, training_data['attack'])
        predictions = clf.predict(test_data)
        (precision, recall, fbeta, support) = precision_recall_fscore_support(test_data['attack'], predictions, average='binary')
        print('Recall: %.3f' %recall)
        print('Precision: %.3f' %precision)
        print('F-Beta: %.3f' %fbeta)
        recalls.append(recall)
        precisions.append(precision)
        fbetas.append(fbeta)
        conf_matrix = confusion_matrix(test_data['attack'], predictions)
        print('Confusion Matrix:\n', conf_matrix)
        auc = roc_auc_score(test_data['attack'], clf.predict_proba(test_data)[:, 1])
        print('Test ROC AUC: %.3f' %auc)
        roc_aucs.append(auc)

    print('Avg. Recall: %.3f' %np.mean(recalls))
    print('Avg. Precision: %.3f' %np.mean(precisions))
    print('Avg. F-Beta: %.3f' %np.mean(fbetas))
    print('Avg. ROC AUC: %.3f' %np.mean(roc_aucs))

end = datetime.now()

time = end - start
print("Time to run: ", time)
