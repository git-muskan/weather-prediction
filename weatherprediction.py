#Import necessary libs

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import datetime
import streamlit as st

# Load the datasets
dataset_url1 = 'https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/2/station/71420/period/corrected-archive/data.csv'
dataset_url2 = 'https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/2/station/71420/period/latest-months/data.csv'

data1 = pd.read_csv(dataset_url1, sep=';', skiprows=13, names=[
    'Fran Datum Tid (UTC)', 'till', 'day', 'temperature', 'Kvalitet', 'Tidsutsnitt:', 'Unnamed: 5'
])
data2 = pd.read_csv(dataset_url2, sep=';', skiprows=14, names=[
    'Fran Datum Tid (UTC)', 'till', 'day', 'temperature', 'Kvalitet', 'Tidsutsnitt:', 'Unnamed: 5'
])

def preprocess_data(data):
    data['day'] = pd.to_datetime(data['day'], errors='coerce')
    data['till'] = pd.to_datetime(data['till'], errors='coerce')
    data['day_of_year'] = data['day'].dt.dayofyear
    data['month'] = data['day'].dt.month
    data['weekday'] = data['day'].dt.weekday
    data = data.drop(columns=['Fran Datum Tid (UTC)', 'Kvalitet', 'Tidsutsnitt:', 'Unnamed: 5'])
    return data.dropna()

def train_data():
    data1_processed = preprocess_data(data1)
    data2_processed = preprocess_data(data2)

    X = data1_processed.drop(columns=['temperature', 'day', 'till'])
    y = data1_processed['temperature']
    X2 = data2_processed.drop(columns=['temperature', 'day', 'till'])
    y2 = data2_processed['temperature']

    X = pd.concat([X, X2])
    y = pd.concat([y, y2])
    
    X = X.astype(int)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

    # Standardizing the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest Model
    model = RandomForestRegressor(n_estimators=100, random_state=123)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)

    # Evaluating the model
    print(f'R^2 Score: {r2_score(y_test, pred)}')
    print(f'Mean Squared Error: {mean_squared_error(y_test, pred)}')

    # Saving the model and scaler
    joblib.dump(model, 'weather_predictor.pkl')
    joblib.dump(scaler, 'scaler.pkl')

def get_the_weather(date):
    weather = data1['day']
    temp = data1['temperature']

    for i in range(len(weather)):
        if weather.iloc[i] == date:
            return temp.iloc[i]
    return None

def predict_weather():
    model = joblib.load('weather_predictor.pkl')
    scaler = joblib.load('scaler.pkl')
    
    st.write("Enter the details of the date you would like to predict")
    
    year = st.text_input("Year", "2024")
    month = st.text_input("Month number (00)", "06")
    day = st.text_input("Day number (00)", "12")

    if st.button("Predict Weather"):
        try:
            date_str = f"{year}-{month}-{day}"
            day = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            day_of_year = day.timetuple().tm_yday
            month = day.month
            weekday = day.weekday()

            X = np.array([[day_of_year, month, weekday]])
            X_scaled = scaler.transform(X)

            predicted_temp = model.predict(X_scaled)[0]
            actual_temp = get_the_weather(day)

            st.write(f"The temperature is predicted to be: {predicted_temp}")
            st.write(f"The temperature was actually: {actual_temp if actual_temp is not None else 'Data not available'}")
        except ValueError as e:
            st.write(f"Invalid date: {e}")

def lookup_weather():
    st.write("Enter the details of the date you would like to look up")
    
    year = st.text_input("Year", "2024")
    month = st.text_input("Month number (00)", "06")
    day = st.text_input("Day number (00)", "12")

    if st.button("Look Up Weather"):
        try:
            date_str = f"{year}-{month}-{day}"
            day = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            actual_temp = get_the_weather(day)

            st.write(f"The temperature on {date_str} was: {actual_temp if actual_temp is not None else 'Data not available'}")
        except ValueError as e:
            st.write(f"Invalid date: {e}")

# Main function to run Streamlit app
def main():
    st.title("Weather Forecasting App")

    menu = ["Look up the weather on a specific day", "Predict the weather on a specific day", "Exit"]
    choice = st.selectbox("Menu", menu)

    if choice == "Look up the weather on a specific day":
        lookup_weather()
    elif choice == "Predict the weather on a specific day":
        predict_weather()
    elif choice == "Exit":
        st.write("Exiting...")

if __name__ == "__main__":
    train_data()
    main()
