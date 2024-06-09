from flask import Blueprint, render_template, request
from .models import predict_sentiment

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    sentiment = predict_sentiment(review)
    return render_template('result.html', sentiment=sentiment)