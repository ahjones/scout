import csv
import codecs
import pandas as pd

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
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

v = CountVectorizer(min_df = 1)
x = v.fit_transform(insults['Comment'])

#Learn from the fit
lr = LogisticRegression()
cv_array = cross_val_score(lr, x, insults.Insult, cv=5)
cv_mean = sum(cv_array)/len(cv_array) #0.842

#Try a prediction
#lr.predict(v.transform(['have a nice day']))
