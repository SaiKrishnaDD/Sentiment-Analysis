import pandas as pd import string import nltk from sklearn.model_selection import 
train_test_split from sklearn.feature_extraction.text import TfidfVectorizer from 
sklearn.linear_model import LogisticRegression from sklearn.metrics import 
accuracy_score, confusion_matrix, classification_report import matplotlib.pyplot as 
plt import seaborn as sns 
nltk.download('stopwords') 
nltk.download('punkt') 
nltk.download('wordnet') 
from nltk.corpus import stopwords from 
nltk.tokenize import word_tokenize from 
nltk.stem import WordNetLemmatizer 
# Load dataset 
df  
= pd.read_csv('https://raw.githubusercontent.com/datasets/sentiment-analysis
imdb/master/data/imdb_labelled. 
txt', names=['text', 'label'], sep='\t') 

# Preprocessing function 
def preprocess_text(text): 
text = text.lower()     
word_tokenize(text)     
t.isalpha()]     
tokens = 
tokens = [t for t in tokens if 
stop_words = 
set(stopwords.words('english'))     
tokens = [t for t 
in tokens if t not in stop_words]     lemmatizer = 
WordNetLemmatizer()     tokens = 
[lemmatizer.lemmatize(t) for t in tokens]     
'.join(tokens) 
return ' 
df['clean_text'] = df['text'].apply(preprocess_text) 
# Split data 
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42) 
# TF-IDF Vectorization tfidf = 
TfidfVectorizer(max_features=5000) 
X_train_tfidf = tfidf.fit_transform(X_train) 
X_test_tfidf = tfidf.transform(X_test) 
# Model training model = 
LogisticRegression() 
model.fit(X_train_tfidf, y_train) 
# Predictions y_pred = 
model.predict(X_test_tfidf) # 
Evaluation print("Accuracy:", 
accuracy_score(y_test, y_pred)) 
print("\nClassification Report:\n", 

classification_report(y_test, 
y_pred)) 
# Confusion Matrix cm = confusion_matrix(y_test, 
y_pred) sns.heatmap(cm, annot=True, fmt='d', 
cmap='Blues') plt.title("Confusion Matrix") 
plt.xlabel("Predicted") plt.ylabel("Actual") plt.show()