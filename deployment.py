from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import joblib
import streamlit as st
# import import_ipynb
# import ipynb
# from hell2 import preprocessing
from preprocessinFunction import preprocessing2

def output(answer):
    if answer[0] == 0:
        return "negative"
    elif answer[0] == 2:
        return "positive"
    else:
        return "neutral"

model = joblib.load("sentiment_analysis_model2.pkl")
x = model.predict("I bad")
print(output(x))

st.title("Sentiment Analyzer")
x = st.text_input("Input a Sentence")
if x == "":
    st.write("Sentiment: ")
else:
    prediction = output(model.predict(x))
    if prediction == "negative":
        st.write(f"Sentiment: :red[{prediction}]")
    elif prediction ==  "positive":
        st.write(f"Sentiment: :green[{prediction}]")
    else:
        st.write(f"Sentiment: :blue[{prediction}]")


page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://i.im.ge/2024/05/07/ZW4j4P.WhatsApp-Image-2023-12-16-at-17-49-52-8a255953.jpeg");
  background-size: cover;
}
[data-testid="stHeader"]{
  background-color: rgba(0,0,0,0);
}
</style>
"""

st.markdown(page_element, unsafe_allow_html=True)
