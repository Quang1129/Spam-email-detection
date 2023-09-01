import pickle
import streamlit as st

model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))

def main():
    st.title("Email Spam Dectection")
    st.subheader("Build with Streamlit and Python")
    message = st.text_input("Enter a text: ")
    if st.button("Predict"):
        df = [message]
        vect = cv.transform(df).toarray()
        prediction = model.predict(vect)
        result = prediction[0]
        if result == 0:
            st.success('This is good email')
        else:
            st.error('This is spam email')
            


main()