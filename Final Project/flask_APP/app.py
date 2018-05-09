from flask import Flask,render_template,url_for,request
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)
clf_rf_2 = joblib.load('project2.pkl') 
xgc = joblib.load('project1.pkl') 


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/project1')
def project1():
    return render_template('project1.html')


@app.route('/project1/predict',methods=['POST'])
def predict1():
    data =request.form
    index = data['test0']
    class_label = ['ARCH', 'ASYM', 'CALC', 'CIRC', 'MISC', 'NORM', 'SPIC']
    data = np.load('test'+str(index)+'.npy')
    file_name = "/static/test"+str(index)+".png"
    prediction = xgc.predict(data.reshape(1,-1))
    label = class_label[prediction[0]]
    return render_template('result1.html',prediction = label, file_path = file_name)





@app.route('/project2')
def project2():
    return render_template('project2.html')

@app.route('/project2/predict',methods=['POST'])
def predict2():
    data =request.form
    area_mean = data['area_mean']
    area_se = data['area_se']
    texture_mean = data['texture_mean']
    concavity_worst = data['concavity_worst']
    concavity_mean = data['concavity_mean']
    data = np.array([area_mean, area_se, texture_mean, concavity_worst,concavity_mean]).reshape(1,-1)
    prediction = clf_rf_2.predict(data)
    return render_template('result2.html',prediction = prediction[0])



if __name__ == '__main__':
    app.run()