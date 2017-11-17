# author: @Rudrakh97
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import metrics
import pandas as pd
import csv
import sys
import string
import numpy
import codecs

csv.field_size_limit(sys.maxint)
printable = set(string.printable)
texts = []

# number of clusters
num_clusters = int(sys.argv[2])

# number of components to be retained during SVD
# throws error if increases beyond vocabulary size
num_components = 80

labels = []

def csv_dict_reader(file_obj):
    print "Reading file: ...\n"
    reader = csv.reader(file_obj, delimiter=',')
    b = 1
    # change c for controlling number of samples for training dataset
    c = int(sys.argv[1])
    for row in reader:
        s = row[6]
        #s = row[1]
        if(len(s)<500):
            continue
        s = unicode(s, errors='ignore')
        sys.stdout.write("\rNumber of text samples in training list: %i" % b)
        sys.stdout.flush()
        b = b + 1
        filter(lambda x: x in printable, s)
        s = s.lower()
        texts.append(s)
        age = int(row[2])
        if age<18:
            labels.append(0)
        elif age<25:
            labels.append(1)
        elif age<45:
            labels.append(2)
        else:
            labels.append(3)
        if b > c:
            break

file = open("blogtext.csv", "r")
csv_dict_reader(file)
print "\n"

Vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), lowercase=True,  norm=None, encoding='utf-8')
X = Vectorizer.fit_transform(texts)

svd = TruncatedSVD(n_components = num_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(X)

joblib.dump(km,'kmeans.sav')
joblib.dump(Vectorizer,'tfidf.sav')
joblib.dump(lsa,'lsa.sav')

# km.labels_ is the label array the document corpus after clustering
# print km.labels_
print "\n"
# count of documents in each label
unique, counts = numpy.unique(km.labels_, return_counts=True)
print dict(zip(unique, counts))

labels = sorted(labels)
labels1 = sorted(km.labels_)

print "\nEVALUATION SCORES: "
print "====================================================================\n"

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, labels1))
print("Completeness: %0.3f" % metrics.completeness_score(labels, labels1))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, labels1))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, labels1))