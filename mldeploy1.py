import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = "./best_model_linear.pkl"
model = pickle.load(open(model_path, "rb"))

estimator_path = "./best_estimator.pkl"
estimator = pickle.load(open(estimator_path, "rb"))

@app.route("/", methods=['POST', 'GET'])
def predict():
    print('server is running')
    if request.method == 'POST':
        try:
            # Extract form data
            input_data = {
                'anxiety_level': float(request.form['anxiety_level']),
                'self_esteem': float(request.form['self_esteem']),
                'mental_health_history': float(request.form['mental_health_history']),
                'depression': float(request.form['depression']),
                'headache': float(request.form['headache']),
                'blood_pressure': float(request.form['blood_pressure']),
                'sleep_quality': float(request.form['sleep_quality']),
                'noise_level': float(request.form['noise_level']),
                'safety': float(request.form['safety']),
                'basic_needs': float(request.form['basic_needs']),
                'teacher_student_relationship': float(request.form['teacher_student_relationship']),
                'future_career_concerns': float(request.form['future_career_concerns']),
                'social_support': float(request.form['social_support']),
                'peer_pressure': float(request.form['peer_pressure']),
                'extracurricular_activities': float(request.form['extracurricular_activities']),
                'bullying': float(request.form['bullying'])
            }
            top_16_features = ['anxiety_level', 'self_esteem', 'mental_health_history', 'depression', 'headache', 'blood_pressure', 'sleep_quality', 'noise_level','safety', 'basic_needs', 'teacher_student_relationship', 'future_career_concerns', 'social_support', 'peer_pressure', 'extracurricular_activities','bullying']
            # Select the relevant features for prediction
            X_user = np.array([input_data[feature] for feature in top_16_features]).reshape(1, -1)

            # Predict academic performance using the best linear regression model
            academic_performance_prediction = model.predict(X_user)
            print(academic_performance_prediction)

            # Predict stress level using the best logistic regression model
            stress_level_prediction = estimator.predict(X_user)
            print(stress_level_prediction)

            # Display the predicted academic performance and stress level to the user
            academic_performance_result = academic_performance_prediction[0] if len(academic_performance_prediction) > 0 else None
            stress_level_result = stress_level_prediction[0] if len(stress_level_prediction) > 0 else None

            print("\nPredicted Academic Performance:", academic_performance_result)
            print("Predicted Stress Level:", stress_level_result)
            
            return render_template('studperf.html', academic_performance_prediction=academic_performance_result, stress_level_prediction=stress_level_result)
        
        except Exception as e:
            error_message = str(e)
            return render_template('error.html', error_message=error_message)
    
    else:
        return render_template('studperf.html')


if __name__ == "__main__":
    app.run(debug=True)
