import csv
import codecs
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_regression

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

        print "CV Mean"
        print evaluate(LogisticRegression(), x, insults.Insult) #0.872

        print "T Mean"
        print evaluate(DecisionTreeClassifier(), x.todense(), insults.Insult) #0.698

        print "Forest Mean"
        print evaluate(RandomForestClassifier(), x.todense(), insults.Insult) #0.817

        print "ADA Mean"
        print evaluate(AdaBoostClassifier(), x.todense(), insults.Insult) #0.842

        print "NB Mean"
        print evaluate(MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True), x.todense(), insults.Insult) #0.806

        #Grid search cross validator to find a good set of parameters
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

        with open('result', 'w') as result:
            result.write("Id, Insult\n")
            for row in test.iterrows():
                comment = row[1]["Comment"]
                num = row[1]["id"]
                prediction = gs.best_estimator_.predict_proba([comment])[0][1]
                result.write("%s, %f\n" % (num, prediction))

# run the classifiers
classifiers()
