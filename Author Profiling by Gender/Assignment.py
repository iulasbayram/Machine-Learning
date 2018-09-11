import os
import nltk
import csv
import re
import numpy as np
import pandas
from xml.dom import minidom
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

# Below that, There is a function for eliminating unused words in tweets.
def eliminateNonEnglistWords(tags):
    regex = "^.*http.*$|[^a-zA-Z0-9_]"
    temp = []
    for i in range(0,len(tags)):
        x = re.search(regex, tags[i][0])
        if x:
            temp.append(tags[i])

    return list(set(tags) - set(temp))


truth_array = []
path = os.path.join("en","truth.txt")
f = open(path, 'r')

for line in f: #This part is to split truth.txt components.
    temp = line.strip("\n").split(":::")
    truth_array.append(temp)
print(truth_array)

gender_list = ["male","female"]

file1 = open("tweets.csv","w")
writer = csv.writer(file1)
writer.writerow(["gender","determiner" ,"preposition", "pronoun"])
tree=[]
for i in range(0,len(truth_array)):
    general_prep = 0
    general_pron = 0
    general_deter = 0
    path = os.path.join("en",truth_array[i][0]+".xml")
    tree = minidom.parse(path)
    items = tree.getElementsByTagName('document')
    for elem in items:
        prep = 0
        pron = 0
        deter = 0
        temp_list = []
        token = nltk.word_tokenize(elem.firstChild.data) # nltk part to split tweets into words and tokens.
        tagged = nltk.pos_tag(token)
        tagged = list(tagged)
        tagged = eliminateNonEnglistWords(tagged)
        for tag in tagged:
            if tag[1] == "DT" or tag[1] == "WDT" or tag[1] == "PDT": # DT,WDT,PDT --> Determiner
                deter += 1
            elif tag[1] == "IN" or tag[1] == "TO": # IN, TO --> Preposition
                prep += 1
            elif tag[1] == "PRP" or tag[1] == "PRP$" or tag[1] == "WP" or tag[1] == "WP$": # PRP, PRP$, WP, WP$ --> Pronoun
                pron += 1
        general_deter += deter
        general_prep += prep
        general_pron += pron
        if (truth_array[i][1] == gender_list[0]): # If gender is male, I assign number 1 to represent male
            temp_list.append(1)
            temp_list.append(deter)
            temp_list.append(prep)
            temp_list.append(pron)
            writer.writerow(temp_list)
        else: # If gender is female, I assign number 0 to represent female
            temp_list.append(0)
            temp_list.append(deter)
            temp_list.append(prep)
            temp_list.append(pron)
            writer.writerow(temp_list)

file1.close()

file = 'tweets.csv'
df = pandas.read_csv(file)
X = np.array(df[["determiner","preposition","pronoun"]]).astype(float)

Y = np.array(df[["gender"]]).astype(float)
Y.dtype = 'float64'

sc = preprocessing.MinMaxScaler()
X = sc.fit_transform(X) # fitting X train set using linear regression
Y = Y.ravel()
X_train ,Y_train ,X_test ,Y_test = [], [], [], []
kf = KFold(n_splits=10)
for tr_index, ts_index in kf.split(X): # splitting our data train and test sets using kFold function.
    X_train, X_test = X[tr_index], X[ts_index]
    Y_train, Y_test = Y[tr_index], Y[ts_index]

reg = LogisticRegression(random_state=0)
reg.fit(X_train,Y_train)

score = cross_val_score(reg,X,Y,cv=10)
print("Accuracy is -----> " ,np.mean(score))