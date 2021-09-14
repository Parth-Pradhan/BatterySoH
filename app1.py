
from flask import Flask, jsonify, render_template, request, redirect, url_for

app1 = Flask(__name__)

@app1.route('/')
def home():
    return render_template('index.html')

@app1.route('/predict', methods = ['POST'])
def predict():
    '''
    for rendering results on HTML GUI
    '''
    Cycle = request.form["Enter current discharge cycle"]
    Pred_hor = request.form["Enter prediction horizon"]
    Battery_name = request.form.get('Battery name')

    with open('nopol.txt', 'w') as f:
        f.write(Battery_name + '\n')
        f.write(Cycle + '\n')
        f.write(Pred_hor)
    return render_template('thankyou.html')

if __name__ == "__main__":
    app1.run(host ='127.0.0.1', port = '5000', debug = True)




