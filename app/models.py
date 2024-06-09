import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model and vectorizer (assumed to be trained and saved previously)
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

tokenizer = nltk.RegexpTokenizer(r'\w+')
stopword_list = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z0-9\s]' if remove_digits else r'[^a-zA-Z\s]'
    return re.sub(pattern, '', text)

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    return ' '.join(filtered_tokens)

def preprocess_text(text):
    text = denoise_text(text)
    text = remove_special_characters(text)
    text = lemmatize_text(text)
    text = remove_stopwords(text)
    return text

def predict_sentiment(review):
    review = preprocess_text(review)
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    return 'Positive' if prediction == 1 else 'Negative'
    # return review
