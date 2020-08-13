from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
import re

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	'''load the bengali stopwords from pickle file'''
	stopword_list = open('book_stopwords.pkl', 'rb')
	stp = pickle.load(stopword_list)

	'''This function is for cleaning the reviews'''
	
	def process_reviews(review):
		review = review.replace('\n', '')  # removing new line
		# removing unnecessary punctuation
		review = re.sub('[^\u0980-\u09FF]', ' ', str(review))
		result = review.split()
		review = [word.strip() for word in result if word not in stp]
		review = " ".join(review)
		return review

	'''load the pickle file of the cleaned data '''
	cleaned_data = open('book_review_data.pkl','rb')
	data = pickle.load(cleaned_data)

	'''Extract TF-IDF for Unigram feature '''
	tfidf = TfidfVectorizer(use_idf=True, tokenizer=lambda x: x.split())
	X = tfidf.fit_transform(data.cleaned)

	'''load the Multinomial Naive bayes model'''
	model = open('book_review_mnb.pkl', 'rb')
	nb = pickle.load(model)

	''' Take the input text and follow the steps to get a prediction
		and pass this values into the template file
	'''
	if request.method == 'POST':
		comment = request.form['comment']
		review = process_reviews(comment)
		vect = tfidf.transform([review]).toarray()
		my_prediction = nb.predict(vect)
		prediction_score = nb.predict_proba(vect)
		score = round(max(prediction_score.reshape(-1)), 2) * 100

	return render_template('sent_prediction.html',value = comment,sentiment = my_prediction,prob = score )



if __name__ == '__main__':
	app.run(debug=True)