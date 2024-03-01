import csv
import nltk
import wordninja
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

############################################ Data Preparation #############################################
###########################################################################################################

####### Import data #######
path = 'C:/Users/Administrator/Desktop/RandomedAnnotation03.csv'
comment,sentiment = [],[]
with open(path, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        comment.append(row[0])
        sentiment.append(row[1])
del comment[0],sentiment[0] 

####### Remove empty row ####### 
while len(comment) > 0 and (comment[0] == "" or sentiment[0] == ""):
    del comment[0]
    del sentiment[0]
comment   = comment  [:min(len(comment), len(sentiment))]
sentiment = sentiment[:min(len(comment), len(sentiment))]

####### Remove "The media could not be loaded." in comment ####### 
def delMedia(comment):
    return comment.replace('The media could not be loaded.\n','').strip()    
comment = [delMedia(c) if 'The media could not be loaded.' in c else c for c in comment]
             
      
########################################### Data Pre-Processed ############################################
###########################################################################################################

####### Add whitespace ####### 
def addWS(comment):    
    listWords = wordninja.split(comment)
    comment = " ".join(listWords)    
    return comment
for c in range(len(comment)) : comment[c] = addWS(comment[c])

####### Sentence to Token ####### 
tokenComments = []
for c in comment : tokenComments.append(nltk.word_tokenize(c))  

####### Convert to Lowercase #######
tokenComments = [[token.lower() for token in comment] for comment in tokenComments]

####### Remove all non-alphanumeric tokens #######
tokenComments = [[token for token in comment if token.isalnum()] for comment in tokenComments]

####### Remove Stop Words #######
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenComments = [[token for token in comment if token not in stop_words] for comment in tokenComments]


################################################ Modeling #################################################
###########################################################################################################

####### Feature Extraction : Create Bag Of Words ####### 
vectorizer = CountVectorizer()
BOW = vectorizer.fit_transform([' '.join(comment) for comment in tokenComments]) 

####### Split Data to Train and Test Sets (Train 80% : Test 20%) #######
X,y = BOW,sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


####### Naive Bayes Train Model ####### #######################
clf = MultinomialNB()
clf.fit(X_train, y_train)

def SANB(comment):
    comment = [comment]
    x = vectorizer.transform([' '.join(comment)])
    y = clf.predict(x)    
    return (y[0])

####### Naive Bayes Model Evaluation ####### 
y_pred = clf.predict(X_test)

NBaccuracy  = (y_pred == y_test).mean()
NBprecision = precision_score(y_test, y_pred, pos_label='0')
NBrecall    = recall_score(y_test, y_pred, pos_label='0')
NBf1        = f1_score(y_test, y_pred, pos_label='0')
print("Accuracy:", NBaccuracy)
print("Precision:",NBprecision)
print("Recall:",   NBrecall)
print("F1-score:", NBf1)

d,b = 0,0    
for x in range(len(comment)):    
    if str(SANB(comment[x])) == str(sentiment[x]) : d += 1
    else : b += 1
    
print("Naive Bayes",d,b)


####### SVM Train Model ####### #######################

from sklearn.svm import SVC

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

def SASVM(comment):
    comment = [comment]
    x = vectorizer.transform([' '.join(comment)])
    y = clf.predict(x)    
    return (y[0])

####### SVM Model Evaluation #######

y_pred = svm.predict(X_test)
SVMaccuracy  = accuracy_score(y_test, y_pred)
SVMprecision = precision_score(y_test, y_pred, pos_label='0')
SVMrecall    = recall_score(y_test, y_pred, pos_label='0')
SVMf1        = f1_score(y_test, y_pred, pos_label='0')
print("Accuracy:", SVMaccuracy)
print("Precision:",SVMprecision)
print("Recall:",   SVMrecall)
print("F1-score:", SVMf1)

d,b = 0,0    
for x in range(len(comment)):    
    if str(SASVM(comment[x])) == str(sentiment[x]) : d += 1
    else : b += 1
    
print("SVM",d,b)













        
        

        







    
    







