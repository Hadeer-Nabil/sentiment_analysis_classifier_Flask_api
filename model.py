# importing libs
import os 
import pandas as pd 
import numpy as np
import pickle
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.svm import LinearSVC

def load_train_test_imdb_data(data_dir):

    data = {}
    for split in ["train", "test"]:
        data[split] = []
        for sentiment in ["neg", "pos"]:
            score = 1 if sentiment == "pos" else 0

            path = os.path.join(data_dir, split, sentiment)
            file_names = os.listdir(path)
            for f_name in file_names:
                with open(os.path.join(path, f_name), "r", encoding="utf-8") as f:
                    review = f.read()
                    data[split].append([review, score])

    np.random.shuffle(data["train"])        
    data["train"] = pd.DataFrame(data["train"],
                                 columns=['text', 'sentiment'])

    np.random.shuffle(data["test"])
    data["test"] = pd.DataFrame(data["test"],
                                columns=['text', 'sentiment'])

    return data["train"], data["test"]

def clean_text(text):
    """
        clean data from special characters and html tags
    """
    
    text = text.strip().lower()
    
    # remove tags 
    text = re.sub(r'<.*?>', '', text)
            
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    text = ''.join((filter(lambda x: x not in filters.split(), text)))

    return text

def stem_sentences(sentence):
    stemmer = PorterStemmer()
    tokens = sentence.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


train_data, test_data = load_train_test_imdb_data(
    data_dir="aclImdb/")


train_data['sentiment'].value_counts().plot(kind = 'bar')

vectorizer = TfidfVectorizer(stop_words="english",
                             preprocessor=clean_text,
                             ngram_range=(1, 2))

tfidf = vectorizer.fit(train_data["text"])
training_features = vectorizer.transform(train_data["text"])


#training_features = vectorizer.fit_transform(train_data["text"])    
test_features = vectorizer.transform(test_data["text"])

# Training
model = LinearSVC()
model.fit(training_features, train_data["sentiment"])
y_pred = model.predict(test_features)

# Evaluation

print(confusion_matrix(test_data["sentiment"],y_pred))
print(classification_report(test_data["sentiment"],y_pred))
print("Accuracy: {:.2f}".format(accuracy_score(test_data["sentiment"], y_pred)*100))

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))



# save tfid to disk
pickle.dump(tfidf, open('tfidf.pickle', 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
loaded_vec = pickle.load(open("tfidf.pickle", 'rb'))
result = loaded_model.score(test_features, test_data["sentiment"])

neg = loaded_model.predict(loaded_vec.transform(["this is bad"]))
pos = loaded_model.predict(loaded_vec.transform(["this is good"]))

print(neg)
print(pos)


