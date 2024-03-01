import csv
import nltk
import wordninja
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

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

BOW = ([' '.join(comment) for comment in tokenComments]) 

# Create bag of words vectorizer
vectorizer = CountVectorizer()

# Create bag of words matrix
X = vectorizer.fit_transform(BOW).toarray()

# Convert labels to numerical values
Y = sentiment

####### Split Data to Train and Test Sets (Train 80% : Test 20%) #######
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define model architecture
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model
model.fit(X_train, Y_train, epochs=10, batch_size=32)

# Evaluate model
score = model.evaluate(X_test, Y_test)
print("Accuracy:", score[1])






import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

csv = ["RandomedAnnotation01.csv","RandomedAnnotation02.csv","RandomedAnnotation03.csv"]
fold_scores = []
fold_precisions = []
fold_recalls = []
fold_f1_scores = []

avg_fold_scores = []
avg_fold_precisions = []
avg_fold_recalls = []
avg_fold_f1_scores  = []

# โหลดข้อมูล
for i  in csv : 
  print("*"*50 , i , "*"*50)
  data = pd.read_csv(i)

# แบ่งข้อมูลเป็น features และ target variable
  X = data["Tweets"]

  y = data["Annotation"].astype(int)

# One-Hot Encoding กับ features
  ohe = OneHotEncoder(sparse_output=False)
  X_ohe = ohe.fit_transform(X.values.reshape(-1, 1))
# Scaling กับ features
  scaler = StandardScaler(with_mean=False)
  X_scaled = scaler.fit_transform(X_ohe)

# สร้าง model ด้วย Logistic Regression
  model = keras.Sequential([
    keras.layers.Dense(units=64, activation='relu', input_shape=(len(data["Tweets"]),)),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')])

  model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# กำหนด Stratified K-Fold Cross-Validation
  skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

  for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
      X_train, y_train = X_scaled[train_idx], y[train_idx]
      X_test, y_test = X_scaled[test_idx], y[test_idx]

      history = model.fit(X_train,y_train, epochs=10, verbose=1)

      y_test_binary = (y_test >= 0.5).astype(int)
    # แสดงผลของแต่ละ fold
      test_target = []
      predict = []
      print(f"Fold {fold + 1}:")
      print(f"  Train: index={train_idx}")
      print(f"  Test:  index={test_idx}")
      for i in y_test :
        test_target.append(i)
      for i in y_test_binary:
        predict.append(i)
      print("Test : " , test_target)
      print("Predict :  " , predict)
      test_loss, test_acc = model.evaluate(X_test, y_test_binary)
      print('Test accuracy:', test_acc)

      tn, fp, fn, tp = confusion_matrix(y_test_binary,model.predict(X_test).round()).ravel()

      precision_score = tp / (tp+fp)
      recall = tp / (tp+fn)
      F1_score = 2 * (precision_score * recall) / (precision_score + recall)
      Accuracy = (tp +  tn) / (tp+  fp + tn + fn)

      print(f"Accuracy: {Accuracy}")
      print(f"Precision: {precision_score}")
      print(f"Recall: {recall}")
      print(f"F1-Score: {F1_score}")
      print("-" * 20)
      fold_scores.append(Accuracy)
      fold_precisions.append(precision_score)
      fold_recalls.append(recall)
      fold_f1_scores.append(F1_score) 

# แสดงผลเฉลี่ยของทั้ง 10 folds
  print(f"Average Metrics:")
  print(f"Accuracy: {np.mean(fold_scores)}")
  print(f"Precision: {np.mean(fold_precisions)}")
  print(f"Recall: {np.mean(fold_recalls)}")
  print(f"F1-Score: {np.mean(fold_f1_scores)}")
  avg_fold_scores.append(np.mean(fold_scores))
  avg_fold_precisions.append(np.mean(fold_precisions))
  avg_fold_recalls.append(np.mean(fold_recalls))
  avg_fold_f1_scores.append(np.mean(fold_f1_scores))


print("--"*50,"Macro Average Score ","--"*50)


print(avg_fold_scores,"  Macro Average Accuracy = " ,np.mean(avg_fold_scores))
print(avg_fold_precisions," Macro Average Precision Score = " ,np.mean(avg_fold_precisions))
print(avg_fold_recalls," Macro Average Recall = " ,np.mean(avg_fold_recalls))
print(avg_fold_f1_scores," Macro  Average F1 Score = " ,np.mean(avg_fold_f1_scores))


















        
        

        







    
    







