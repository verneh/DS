import streamlit as st
import os

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')# To Prevent Errors
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import joblib

def get_value(val,my_dict):
	for key ,value in my_dict.items():
		if val == key:
			return value

def main():

	st.title("Churn Predictor")
	st.subheader("Predicting Churn with Machine Learning and Streamlit")

	# Load and drop first column
	df = pd.read_csv("churn.csv")
	df = df.drop("Unnamed: 0", axis=1)

	# Prediction
	st.write("Select attributes: ")

	# Features
	active = {"Inactive":0,"Active":1}
	choice_active = st.radio("Active Account?",tuple(active.keys()))
	result_active = get_value(choice_active,active)
	st.markdown("---")
	age = st.slider("Age",18,92)
	st.markdown("---")
	balance = st.slider("Balance",0,251000)
	st.markdown("---")
	has_cc = {"No":0,"Yes":1}
	choice_cc = st.radio("Credit Card?",tuple(has_cc.keys()))
	result_cc = get_value(choice_cc,has_cc)
	st.markdown("---")
	creditscore = st.slider("Credit Score",350,850)
	st.markdown("---")
	gender = {"Female":0,"Male":1}
	choice_gender = st.radio("Gender",tuple(gender.keys()))
	result_gender = get_value(choice_gender,gender)
	st.markdown("---")
	numofproducts = st.slider("Number of Products",1,4)
	st.markdown("---")
	salary = st.slider("Salary (Estimated)",11,200000)
	st.markdown("---")
	tenure = st.slider("Tenure",0,10)
	st.markdown("---")
	france = {"No":0,"Yes":1}
	choice_france = st.radio("Resident of France?",tuple(france.keys()))
	result_france = get_value(choice_france,france)
	st.markdown("---")
	germany = {"No":0,"Yes":1}
	choice_germany = st.radio("Resident of Germany?",tuple(germany.keys()))
	result_germany = get_value(choice_germany,germany)
	st.markdown("---")
	spain = {"No":0,"Yes":1}
	choice_spain = st.radio("Resident of Spain?",tuple(spain.keys()))
	result_spain = get_value(choice_spain,spain)


	results = [result_active, age, balance, result_cc, creditscore, result_gender, numofproducts, salary, tenure, result_france, result_germany, result_spain]
	displayed_results = [choice_active, age, balance, choice_cc, creditscore, choice_gender, numofproducts, salary, tenure, choice_france, choice_germany, choice_spain]

	prettified_result = {"result_active":choice_active, "age":age, "balance":balance, "result_cc":choice_cc,
	"creditscore":creditscore, "result_gender":choice_gender, "numofproducts":numofproducts, "salary":salary,
	"tenure":tenure, "result_france":choice_france, "result_germany":choice_germany, "result_spain":choice_spain}

	sample_data = np.array(results).reshape(1, -1)

	st.markdown(" ")

	st.sidebar.title("Ahoy Mate!")
	st.sidebar.markdown("So this is a continuation of my original churn modelling that i put up on github.")

	st.sidebar.markdown("Decided to upload this as an app on AWS, instead of the entire notebook. Feel free \
	to play around and see if the person is still with the bank or not.")
	st.sidebar.subheader("Your Input Summary")
	st.sidebar.json(prettified_result)

	# st.text("Vectorized as ::{}".format(results))

	st.subheader("Prediction?")

	def get_key(val,my_dict):
				for key ,value in my_dict.items():
					if val == value:
						return key

	prediction_label = {"Still a customer!": 0, "Not a customer.": 1}

	if st.button("Yes, Sir!"):

		loaded_model = joblib.load(open("rfc_model.pkl","rb"))
		prediction = loaded_model.predict(sample_data)
		# st.write(prediction)
		final_result = get_key(prediction, prediction_label)
		st.success(final_result)
