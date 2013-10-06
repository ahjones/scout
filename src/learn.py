import csv
import codecs
import pandas as pd

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def load_data_frame(path):

    def unicode_csv_reader(unicode_csv_data):
        csv_reader = csv.reader(utf_8_encoder(unicode_csv_data))
        for row in csv_reader:
            yield [unicode(cell, 'utf-8') for cell in row]

    def utf_8_encoder(unicode_csv_data):
        for line in unicode_csv_data:
            yield line.encode('utf-8')

    file = unicode_csv_reader(codecs.open(path, 'r', 'utf-8'))

    columns = next(file)
    return pd.DataFrame.from_records([x for x in file], columns=columns)

insults = load_data_frame('../data/train-utf8.csv')
insults['Date'] = pd.to_datetime(insults['Date'])
insults['Insult'] = insults['Insult'].apply(int)

def classifiers():
    v = CountVectorizer(min_df = 1)
    x = v.fit_transform(insults['Comment'])

    #Learn from the fit
    lr = LogisticRegression()
    cv_array = cross_val_score(lr, x, insults.Insult, cv=5, scoring='roc_auc')
    cv_mean = sum(cv_array)/len(cv_array) #0.872

    t = DecisionTreeClassifier()
    t_array = cross_val_score(t, x.todense(), insults.Insult, cv=5, scoring='roc_auc')
    t_mean = sum(t_array)/len(t_array) #0.698

    forest_array = cross_val_score(RandomForestClassifier(), x.todense(), insults.Insult, cv=5, scoring='roc_auc')
    forest_mean = sum(forest_array)/len(forest_array) #0.817

    ada_array = cross_val_score(AdaBoostClassifier(), x.todense(), insults.Insult, cv=5, scoring='roc_auc')
    ada_mean = sum(ada_array)/len(ada_array) #0.842

    #lr.predict(v.transform(['have a nice day']))

#Grid search cross validator to find a good set of parameters
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('lr', LogisticRegression())])
arameters = {'vect__ngram_range': ((1,1), (1,2)), 'vect__max_df': (0.5, 0.75, 1.0)}
gs = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='roc_auc')
gs.fit(insults.Comment, insults.Insult)
gs.best_score_ #0.875
