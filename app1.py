from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

def load_model():
    with open('D:/rutvi/ml/flask/linear_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

def validate_input(data):
    ranges = {
        'Age': (0, 100),
        'Gender': (0, 1),
        'Degree': (0, 20),
        'Job_Title': (0, 2000),
        'Experience_Years': (0, 100),
    }
    errors = []
    for feature, value in data.items():
        if feature in ranges:
            min_val, max_val = ranges[feature]
            if value < min_val or value > max_val:
                errors.append(f"{feature}: Value must be between {min_val} and {max_val}")
    return errors

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Age = int(request.form['a'])
        Gender = int(request.form['b'])
        Degree = float(request.form['c'])
        Job_Title = float(request.form['d'])
        Experience_Years = float(request.form['e'])

        # Prepare the input data for prediction
        input_data = {
            'Age': Age,
            'Gender': Gender,
            'Degree': Degree,
            'Job_Title': Job_Title,
            'Experience_Years': Experience_Years,
        }

        # Validate the input data
        errors = validate_input(input_data)

        if errors:
            return render_template('home.html', errors=errors)

        # Convert the input data to a numpy array
        input_data = np.array(list(input_data.values())).reshape(1, -1)
        output = None

        # Use the loaded model to make the prediction
        prediction = model.predict(input_data)[0]

        output = round(prediction[0],2)

    return render_template('home.html', prediction_text="Employee Salary should be ${}".format(output))

        # Redirect to the result page with the prediction
        #return redirect(url_for('after', prediction=float(prediction)))

    # If it's a GET request, render the index page
   # return render_template('home.html')

#@app.route('/after')
#def result():
    #prediction = request.args.get('prediction')
    #return render_template('after.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
