# author: @Rudrakh97
import sklearn
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import csv
import sys
import string
import numpy


def get_results(test_file, K):
    texts = []
    csv.field_size_limit(sys.maxint)
    printable = set(string.printable)
    km = joblib.load('kmeans.sav')
    lsa = joblib.load('lsa.sav')
    Vectorizer = joblib.load('tfidf.sav')

    filter(lambda x: x in printable, test_file)
    test_file = test_file.lower()
    test = []
    test.append(test_file)
    Y = Vectorizer.transform(test)
    Y = lsa.transform(Y)
    arr = cosine_similarity(Y,km.cluster_centers_)
    values = arr[0]

    def csv_dict_reader(file_obj):
        print "Reading csv file ..."
        reader = csv.reader(file_obj, delimiter=',')
        b = 1
        # change c for controlling number of samples for training dataset
        c = len(km.labels_)
        for row in reader:
            s = row[6]
            #s = row[1]
            if(len(s)<500):
                continue
            s = unicode(s, errors='ignore')
            sys.stdout.write("\rNumber of text samples added to database: %i" % b)
            sys.stdout.flush()
            b = b + 1
            filter(lambda x: x in printable, s)
            s = s.lower()
            texts.append(s)
            if b > c:
                break


    file = open("blogtext.csv", "r")
    csv_dict_reader(file)
    print ""

    def get_doc_cluster(test_file):
        maxi = values[0]
        ind = 0
        for i in range(0,len(values)):
            if values[i] > maxi:
                maxi = values[i]
                ind = i
        return ind

    # returns cluster number (0,1,2,3,...)
    myCluster = get_doc_cluster(test_file)

    tests = []

    for i in range(0,len(texts)):
        if km.labels_[i]==myCluster and len(texts[i]) >= 500:
            tests.append(texts[i])

    theta = []

    for i in range(0,len(tests)):
        theta.append(i)

    Z = Vectorizer.transform(tests)
    Z = lsa.transform(Z)

    rank = cosine_similarity(Y,Z)[0]

    theta = sorted(theta, key = lambda x: -rank[x])

    j = 0
    ret = ""
    for i in range(0,len(tests)):
        ret += "\n"
        if j >= K:
            break
        ret += "Result "+ str(i+1) +":\tSimilarity score: " + str(rank[theta[i]]) + "\n" + str(tests[theta[i]])
        j = j+1

    if j < K:
        ret += "\nNot enough results in database!"
    return ret

# test file
test_doc = ' '.join(sys.argv[1:])
print get_results(test_doc, 4)