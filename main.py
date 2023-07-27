
# Importer les librairies nécessaires

import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Chargement des données dépuis sklearn

iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Faisons la prévision

clf = RandomForestClassifier()
clf.fit(X, Y)

# Création de l'application avec streamlit

st.title("Iris Flower Classification")
st.header("Enter the measurements of the Iris flower:")

# Ajoutons les paramètres

sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Définissons un bouton de prédiction

predict_button = st.button("Predict")

# Affichage de la prédiction du type de fleur d'iris

if predict_button:
    prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    target_names = iris.target_names
    st.write(f"Predicted Iris Flower Type: {target_names[prediction[0]]}")

