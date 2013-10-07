import csv
import codecs
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data_frame(path):

    def unicode_csv_reader(unicode_csv_data):
        csv_reader = csv.reader(utf_8_encoder(unicode_csv_data))
        for row in csv_reader:
            yield [unicode(cell, 'utf-8') for cell in row]

    def utf_8_encoder(unicode_csv_data):
        for line in unicode_csv_data:
            yield line.encode('utf-8')

    # open the file with UTF-8 encoding
    file = unicode_csv_reader(codecs.open(path, 'r', 'utf-8'))

    # read in the first like from the file
    columns = next(file)
    # load (and return) the file contents in the file (minus the header which is passed in as columns)
    return pd.DataFrame.from_records([x for x in file], columns=columns)

# call the above function to get the training frame
insults = load_data_frame('../data/train-utf8.csv')
# convert the Date field and put it back in the data frame
insults['Date'] = pd.to_datetime(insults['Date'])
# convert the Insult result field to an int
insults['Insult'] = insults['Insult'].apply(int)

def evaluate(predictor, x, y):
    cv_array = cross_val_score(predictor, x, y, cv=5, scoring='roc_auc')
    cv_mean = sum(cv_array) / len(cv_array)
    return cv_mean


if __name__ == '__main__':
    def classifiers():
        v = CountVectorizer(min_df = 1)
        x = v.fit_transform(insults['Comment'])

        #Learn from the fit
 #       lr = LogisticRegression()
 #       cv_array = cross_val_score(lr, x, insults.Insult, cv=5, scoring='roc_auc')
 #       cv_mean = sum(cv_array)/len(cv_array) #0.872
 #       print "CV Mean\n"
 #       print cv_mean

 #       t = DecisionTreeClassifier()
 #       t_array = cross_val_score(t, x.todense(), insults.Insult, cv=5, scoring='roc_auc')
 #       t_mean = sum(t_array)/len(t_array) #0.698
 #       print "T Mean\n"
 #       print t_mean

#        forest_array = cross_val_score(RandomForestClassifier(), x.todense(), insults.Insult, cv=5, scoring='roc_auc')
#        forest_mean = sum(forest_array)/len(forest_array) #0.817
#        print "Forest Mean\n"
#        print forest_mean

#        ada_array = cross_val_score(AdaBoostClassifier(), x.todense(), insults.Insult, cv=5, scoring='roc_auc')
#        ada_mean = sum(ada_array)/len(ada_array) #0.842
#        print "ADA Mean\n"
#        print ada_mean

        nbClass = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

        nb_array = cross_val_score(nbClass, x.todense(), insults.Insult, cv=5, scoring='roc_auc')
        nb_mean = sum(nb_array)/len(nb_array) #0.8064
        print "NB Mean\n"
        print nb_mean

        #lr.predict(v.transform(['have a nice day']))

        #Grid search cross validator to find a good set of parameters
        from sklearn.pipeline import Pipeline
        from sklearn.grid_search import GridSearchCV
        from sklearn import svm
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression

        anova_filter = SelectKBest(f_regression, k=5)
        clf = svm.SVC(kernel='linear')
        # ('anova', anova_filter), ('svc', clf)
        pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('lr', LogisticRegression())])
        parameters = {
                    'tfidf__norm': ('l1', 'l2'),
                    'tfidf__use_idf': (True, False),
                    'vect__ngram_range': ((1,1), (2,3)),
                    'vect__max_df': (0.5, 0.75, 1.0)}
        gs = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='roc_auc')
        gs.fit(insults.Comment, insults.Insult)
        print "Best Score"
        print gs.best_score_ #0.875
        # print out the params that had the most effect on the score
        print gs.best_params_

        test = load_data_frame('../data/test-utf8.csv')
        estimates = gs.best_estimator_.predict_proba(test.Comment)

        result = open('result', 'w')
        result.write("Id, Insult\n")
        for row in test.iterrows():
            comment = row[1]["Comment"]
            num = row[1]["id"]
            prediction = gs.best_estimator_.predict_proba([comment])[0][1]
#            print ("%s, %f\n" % (num, prediction))
            result.write("%s, %f\n" % (num, prediction))
        result.close()

# run the classifiers
#classifiers()
