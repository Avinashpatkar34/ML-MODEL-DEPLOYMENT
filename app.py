import numpy as np 
from flask import Flask ,request, render_template
import pickle

# create flask app
app =Flask(__name__)


# load pickle model
model =pickle.load(open("pickle.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    float_feature =[float(x) for x in request.form.values()]
    feartures =[np.array(float_feature)]
    prediction = model.predict(feartures)
    return render_template("index.html",prediction_text ="the flower species is  bro{}.".format(prediction))

if __name__ =="__main__":
    app.run(debug=True)