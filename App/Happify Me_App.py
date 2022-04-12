# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 


# Utils
import joblib 
pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_.pkl","rb"))




# Fxn
def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


# Main Application
def main():
	st.title("Happify Me")
	menu = ["Home","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	
	
	if choice == "Home":
		
		st.subheader("Let us check your emotions")

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("How am I feeling today?")
			submit_text = st.form_submit_button(label='What is my emotion?')

		if submit_text:
			col1,col2  = st.columns(2)

			# Apply Fxn Here
			prediction = predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)
			
			
			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction,emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))



			with col2:
				st.success("Prediction Probability")
				# st.write(probability)
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				# st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)



	elif choice == "About":
		st.title("Happify Me checks your emotions and helps to prevent Mental Health Issues")






if __name__ == '__main__':
	main()
