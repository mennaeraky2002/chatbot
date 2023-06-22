import nltk
import numpy as np
import pandas as pd
import re
from sklearn.metrics import accuracy_score
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
########## Load the dataset

SMS_SH_df = pd.read_csv("C:/chatbot.csv", names=['Q', 'A'], header=None)
df = SMS_SH_df.copy()
###############

################stemming function

ps=nltk.PorterStemmer()
def stemming(tokenized_text):
    tokens=word_tokenize(tokenized_text)
    text=" ".join([ps.stem(word) for word in tokens])
    return text
df['text_stemmed']=df['Q'].apply(lambda x:stemming(x))

#print(df.head(10))

##################

############## Define a function to preprocess the text
lemmatizer=WordNetLemmatizer()
def preprocess(corpus):
    corpus = corpus.lower()
    corpus = re.sub('[^a-zA-Z]', ' ', corpus)
    corpus = corpus.split()
    corpus = [lemmatizer.lemmatize(word) for word in corpus]
    corpus = ' '.join(corpus)
    return corpus

df['Q'] = df['Q'].apply(preprocess)
df['A'] = df['A'].apply(preprocess)

####################


###################### Define a function to clean_text

stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    text = " ".join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])
    return text

df['cleaned_text'] = df['A'].apply(clean_text)

#print(df.head())

#############################

################Count vectorization

vectorizer = CountVectorizer()
features_cv = vectorizer.fit_transform(df['Q'])
#print(features_cv.shape)
#print('Sparse Matrix :\n', features_cv)
features_cv = pd.DataFrame(features_cv.toarray())
features_cv.columns = vectorizer.get_feature_names_out()
#print(features_cv)

################


#############Vectorizing Data: N-Grams
ngram_vect = CountVectorizer(ngram_range=(1,3))
features_ng = ngram_vect.fit_transform(df['Q'])
#print(features_ng.shape)
#print('Sparse Matrix :\n', features_ng)
features_ng = pd.DataFrame(features_ng.toarray())
features_ng.columns = ngram_vect.get_feature_names_out()
#print(features_ng)
##########


###############Vectorizing Data: TF-IDF

vectorizer = TfidfVectorizer()
features_tfidf = vectorizer.fit_transform(df['cleaned_text'])
#print(features_tfidf.shape)
#print('Sparse Matrix :\n', features_tfidf)
features_tfidf = pd.DataFrame(features_tfidf.toarray())
features_tfidf.columns = vectorizer.get_feature_names_out()
#print(features_tfidf)

##############

#vectorizer = TfidfVectorizer()
# Convert the questions and cleaned answers to vectors
X = vectorizer.fit_transform(df['Q'])
#y = vectorizer.transform(df['cleaned_text'])
############# Define a function to respond to user input

##############Create feature for the message length

df['body_len'] = df['Q'].apply(lambda x: len(x)-x.count(' '))
#print(SMS_SH_df.head())

#################

###############Create feature for percent of text that is punctuation
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    if (len(text) - text.count(" ")) == 0:
        return 0
    return round(count/(len(text) - text.count(" ")),3)*100

df['body_len'] = df['Q'].apply(lambda x: len(x) - x.count(" "))
df['punct%'] = df['Q'].apply(lambda x: count_punct(x))
#print(SMS_SH_df.head())

############

################Create feature for The percent of characters in capital letters
def count_Cap(text):
    count = sum([1 for char in text if char.isupper()])
    if (len(text) - text.count(" ")) == 0:
        return 0
    return round( count/(len(text) - text.count(" ")),3)*100

df['body_len'] = df['Q'].apply(lambda x: len(x) - x.count(" "))
df['cap%'] = df['Q'].apply(lambda x: count_Cap(x))

#print(SMS_SH_df.head(3))

#################

#print(SMS_SH_df.describe())


##################feature engineering
df['word_count'] = df['Q'].apply(lambda x: len(x.split()))
df['char_count'] = df['Q'].apply(len)
############

########### Combine features

Y = pd.concat([df[['body_len','word_count', 'char_count','cap%','punct%']], pd.DataFrame(features_tfidf)], axis=1)

# Save the final dataframe
Y.to_csv('final_data.csv', index=False)
#print(Y.head())

#########


def respond(input_text, threshold=0.2):
    # Preprocess the input text
    input_text = preprocess(input_text)
    # Convert the input text to a vector
    input_text_vector = vectorizer.transform([input_text])
    # Compute the cosine similarity between the input text vector and the question vectors
    sim = cosine_similarity(input_text_vector, X).flatten()
    # Sort the indices of the questions by similarity in descending order
    indices = np.argsort(sim)[::-1]
    # Iterate over the questions and return the answer to the most similar question if the similarity is above the threshold
    for index in indices:
        if sim[index] > threshold:
            return df['A'][index]
    # If no question is similar enough, return a default response
    return "I'm sorry, I don't understand. Can you please rephrase your question?"


############accuracy
y_true = df['A'].values
y_pred = []
for input_text in df['Q']:
    response = respond(input_text)
    y_pred.append(response)

accuracy = accuracy_score(y_true, y_pred)
print('Accuracy:', accuracy)
###################


####### Allow users to interact with the chatbot

print("Bot: Hi, I'm a chatbot. How can I help you today?")
print("Bot: If you want to end the conversation, type 'end' or 'quit'.")
while True:
    input_text = input('You: ')
    if input_text.lower() in ['quit', 'end']:
        print("Bot: Goodbye!")
        break
    answer = respond(input_text)
    print('Bot:', answer)