#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle
#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of web-app
@app.route('/')
def home():
    return render_template('index.html')

#For actual prediction
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template('index.html', prediction_text='It is possible that your prostate tumour is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
