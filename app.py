import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
# city_mapping = pickle.load(open('city_mapping.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=["GET", "POST"])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    city = request.form['city']
    team1 = request.form['team1']
    team2 = request.form['team2']
    tosswin = request.form['TossWinner']
    decide = request.form['TossDecision']
    
    with open('city_mapping.pkl', 'rb') as f:
        vocab = pickle.load(f)
    city1 = vocab[city]
    with open('team1_mapping.pkl', 'rb') as f:
        vocab = pickle.load(f)
    cteam1 = vocab[team1]
    cteam2 = vocab[team2]
    cwin  = vocab[tosswin]
    with open('toss_mapping.pkl', 'rb') as f:
        vocab = pickle.load(f)
    cdecide=vocab[decide]
    # city1 = tuple(city1)
    # cteam1 = tuple(cteam1)
    # cteam2 = tuple(cteam2)
    # cwin = tuple(cwin)
    # cdecide = tuple(cdecide)


    # lst = np.array([city1, cteam1, cteam2, cwin, cdecide], dtype='int32').reshape(1,-1)
    data = [[city1, cteam1, cteam2, cwin, cdecide]]
    # columns = ['City', 'Team1', 'Team2', 'TossWinner', 'TossDecision']
    # df = pd.Series(data, index=columns).to_frame().T

    input_df = pd.DataFrame(data, columns=['City', 'Team1', 'Team2', 'TossWinner', 'TossDecision'])
    # input_df = pd.DataFrame({'City':city1,'Team1':cteam1,'Team2':cteam2,'TossWinner':cwin,'TossDecision':cdecide})
    # final_features = np.array(input_df)
    # final_features=np.array(final_features)
    # print(final_features)
    prediction = model.predict(input_df)
    if prediction==cteam1:
        prediction=team1
    if prediction==cteam2:
        prediction=team2

    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='{}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)