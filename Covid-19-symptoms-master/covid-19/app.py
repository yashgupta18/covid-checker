import numpy as np
from flask import Flask, request, jsonify,render_template, url_for, redirect
import pickle
import os

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    if output==0:
      return render_template('home.html', prediction_text='You have a low respiratory illness. Based on your inputs, we advise you to stay home and social distance. Follow the precautions and instructions given by the health agency.Monitor your symptoms and get medical attention if your situation worsens\n')
    elif output==1:
      return render_template('home.html', prediction_text='You have a moderate respiratory illness. 1) start home isolation and social distancing\n 2)Monitor your symptoms and get medical attention if your situation worsens 3)If your situation worsens, do consult you physician\n 4)If you go outside, wear a face mask.\n')
    elif output==2:
      return render_template('home.html', prediction_text='You have a high respiratory illness. \n1)Start home isolation immediately \n2)Please visit a physician as there may be a requirement for further care, COVID 19 testing may be needed at your physician’s advise \n3)Monitor your symptoms and isolate yourself \n4)You can CONSULT ONLINE to speak to a physician immediately \n5)Wear a face mask at all times when near someone else and outside.')

if __name__ == "__main__":
    # app.run(debug=True,use_reloader=False)
    port = int(os.environ.get('PORT', 5023))
    #run the app locally on the givn port
    app.run(debug=True, host='0.0.0.0', port=port)
    #app.run(debug=True)
