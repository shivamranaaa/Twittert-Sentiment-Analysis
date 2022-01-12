from flask import Flask,render_template,request
import pickle

app=Flask(__name__)

classifier=pickle.load(open("classifier.pkl","rb"))
tfidf_v=pickle.load(open("tfidf_v.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    message=request.form['message']
    vect = tfidf_v.transform([message]).toarray()
    my_prediction = classifier.predict(vect)[0]
    
    if my_prediction==1:
        my_prediction="The tweet is racist/sexist"
    else:
        my_prediction="The tweet is not racist/sexist"
        
    return render_template("index.html",output=my_prediction)
if __name__=="__main__":
    app.run()