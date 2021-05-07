#Importing the Libraries
import numpy as np
from flask import Flask, request,render_template
from flask_cors import CORS
import os
import joblib
import pickle
import flask
import os
import newspaper
from newspaper import Article
import urllib
import pandas as pd
import re
import string

#Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')

with open('C:/Users/Radley Roy/OneDrive/Desktop/College/Fake_news/LR_model.sav', 'rb') as handle:
	LR = joblib.load(handle)
with open('C:/Users/Radley Roy/OneDrive/Desktop/College/Fake_news/DT_model.sav', 'rb') as handle:
	DT = joblib.load(handle)
with open('C:/Users/Radley Roy/OneDrive/Desktop/College/Fake_news/GBC_model.sav', 'rb') as handle:
	GBC = joblib.load(handle)
with open('C:/Users/Radley Roy/OneDrive/Desktop/College/Fake_news/RFC_model.sav', 'rb') as handle:
	RFC = joblib.load(handle)
with open('C:/Users/Radley Roy/OneDrive/Desktop/College/Fake_news/tfidf_model.sav', 'rb') as handle:
	vectorization = joblib.load(handle)

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    prob_LR = round(LR.predict_proba(new_xv_test)[0][0]*100,2)
    prob_DT = round(DT.predict_proba(new_xv_test)[0][0]*100,2)
    prob_GBC = round(GBC.predict_proba(new_xv_test)[0][0]*100,2)
    prob_RFC = round(RFC.predict_proba(new_xv_test)[0][0]*100,2)
    check_LR = False
    check_RFC = False
    check = True
    if prob_LR <= 80:
        check_LR = True
    if prob_RFC <= 70:
        check_RFC = True
    return render_template('main.html', check=check, check_LR = check_LR, check_RFC = check_RFC, LR = prob_LR, DT = prob_DT, GBC = prob_GBC, RFC = prob_RFC)
    
if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)