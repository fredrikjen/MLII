import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the preprocessed dataset
titanic_data = pd.read_pickle("Titanic.pkl")

# Feature engineering
titanic_data['SurvivalPrediction'] = 0
titanic_data.loc[(titanic_data['Pclass'] == 1) & (titanic_data['Sex'] == 'female'), 'SurvivalPrediction'] = 1
titanic_data.loc[(titanic_data['Pclass'] == 2) & (titanic_data['Sex'] == 'female'), 'SurvivalPrediction'] = 1
titanic_data.loc[(titanic_data['Pclass'] == 3) & (titanic_data['Sex'] == 'female'), 'SurvivalPrediction'] = 0
titanic_data.loc[(titanic_data['Pclass'] == 1) & (titanic_data['Sex'] == 'male'), 'SurvivalPrediction'] = 0
titanic_data.loc[(titanic_data['Pclass'] == 2) & (titanic_data['Sex'] == 'male'), 'SurvivalPrediction'] = 0
titanic_data.loc[(titanic_data['Pclass'] == 3) & (titanic_data['Sex'] == 'male'), 'SurvivalPrediction'] = 0

# Data splitting
X = titanic_data.drop(['Survived', 'Name', 'SurvivalPrediction'], axis=1)
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# Feature scaling and preprocessing
numerical_cols = ['Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
categorical_cols = ['Sex', 'Pclass']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse=False)

# Handling missing values for numerical columns
numerical_imputer = SimpleImputer(strategy='mean')
numerical_cols_transformer = Pipeline(steps=[
    ('imputer', numerical_imputer),
    ('scaler', numerical_transformer)
])

# Handling missing values for categorical columns
categorical_imputer = SimpleImputer(strategy='constant', fill_value=-1)  # Use any valid numerical fill value
categorical_cols_transformer = Pipeline(steps=[
    ('imputer', categorical_imputer),
    ('encoder', categorical_transformer)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_cols_transformer, numerical_cols),
        ('cat', categorical_cols_transformer, categorical_cols)
    ])

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Model training
svm_classifier = SVC(class_weight='balanced')
svm_classifier.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = svm_classifier.predict(X_test_scaled)
classification_result = classification_report(y_test, y_pred)

# Streamlit UI
st.title('Titanic Survival Prediction App')

# Display the dataset
st.subheader('Titanic Dataset')
st.dataframe(titanic_data)

# Display the classification result
st.subheader('Classification Result')
st.text(classification_result)

# Input elements for the user to choose Pclass and Sex
pclass_input = st.selectbox('Choose Pclass:', [1, 2, 3])
sex_input = st.radio('Choose Sex:', ['male', 'female'])

# Prepare user input for prediction
user_input = pd.DataFrame({
    'Pclass': [pclass_input],
    'Sex': [sex_input],
    'Age': [30],  # Set default age value
    'Siblings/Spouses Aboard': [0],  # Set default value
    'Parents/Children Aboard': [0],  # Set default value
    'Fare': [10]  # Set default fare value
})

# Ensure that 'Pclass' and 'Sex' columns are included in the user input
user_input_transformed = preprocessor.transform(user_input)

# Make prediction
prediction = svm_classifier.predict(user_input_transformed)

# Display the prediction
st.subheader('Prediction')
if prediction[0] == 0:
    st.write('Not Survived')
else:
    st.write('Survived')