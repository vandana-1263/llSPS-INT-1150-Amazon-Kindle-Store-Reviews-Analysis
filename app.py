from keras.models import load_model
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import tensorflow as tf


#graph = tf.get_default_graph()
#cv = pickle.load(open('cv.pkl', 'rb'))


#model=load_model('model.h5')
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/y_predict',methods=['POST'])
def y_predict():
    """
        For rendering results on HTML GUI
    """
    #model.predict(cv.transform(['I\'m So Happy']))
    #x_test=[request.form['Sentence']]
    #print(x_test)
    #return render_template('home.html',prediction_text=x_test)
    #x_test = cv.transform([request.form['Sentence']])
    #return render_template('home.html',prediction_text=str(x_test))
    with open('cv.pkl','rb') as file:
        cv=pickle.load(file)    
        model=load_model('model.h5')
        prediction = model.predict(cv.transform([request.form['Sentence']]))
        output=prediction[0]
        if(output<0.5):
            return render_template('home.html',prediction_text="Negative Review")
        else:
            return render_template('home.html',prediction_text="Positive Review")
if(__name__=="__main__"):
    app.run(debug=True)
