from flask import Flask , render_template , request , jsonify
import pickle 
import sklearn 

tokenizer=pickle.load(open('models/cv.pkl' , 'rb'))
model=pickle.load(open('models/clf.pkl' , 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict' , methods=['POST'])
def predict() :
    if request.method == 'POST' :
        email=request.form.get('content')
    tokenized_email=tokenizer.transform([email])
    prediction=model.predict(tokenized_email)[0]
    prediction=1 if prediction==1 else -1 
    return render_template('index.html' , prediction=prediction , email=email)

@app.route('/api/predict' , methods=['POST'])
def api_predict() :
    data=request.get_json(force=True)
    email=data['content']
    tokenized_email=tokenizer.transform([email])
    prediction=model.predict(tokenized_email)[0]
    prediction=1 if prediction==1 else -1
    return jsonify({'prediction' : prediction , 'email':email})


if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0' , port=8080)
