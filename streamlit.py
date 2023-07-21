import streamlit as st
import joblib
from tqdm import tqdm
tqdm.pandas()
import re
from sklearn.feature_extraction.text import TfidfVectorizer
tfv=TfidfVectorizer()
from nltk.stem import WordNetLemmatizer
import unidecode
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
model =joblib.load('cuisine_models.joblib')
vectorizer=joblib.load('cuisine_vectorizers.joblib')

#preprocess

def preprocess(ingredietns):
    lemmatizer=WordNetLemmatizer()
    text=' '.join(ingredietns)
    text=text.lower()
    text=text.replace('-',' ')
    words=[]
    for word in text.split():
        word=re.sub("[0-9]"," ",word)
        word=re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', word)
        if len(word)<=2: continue
        word=unidecode.unidecode(word)
        word=lemmatizer.lemmatize(word)
        if len(word)>0:words.append(word)
    return ' '.join(words)  
   
reversed_dict={0: 'Greek', 1: 'Southern US', 2: 'Filipino', 3: 'Indian', 4: 'Jamaican', 5: 'Spanish', 6: 'Italian', 7: 'Mexican', 8: 'Chinese', 9: 'British', 10: 'Thai', 11: 'Vietnamese', 12: 'Cajun Creole', 13: 'Brazilian', 14: 'French', 15: 'Japanese', 16: 'Irish', 17: 'Korean', 18: 'Moroccan', 19: 'Russian'}

# Streamlit app
def main():
     st.markdown("<p style='text-align: center;'>Cuisine Guesser: Predict the Cultural Origin of Your Recipe</p>", unsafe_allow_html=True)
     st.title=("Cuisine Classification App")
     image_path = "istockphoto-545286388-612x612.jpg"  # Replace "path_to_your_image.jpg" with the actual file path
     st.image(image_path, use_column_width=True)
     try:
            # User input
            user_text = st.text_input("Enter the ingredietns",'     ')
            user_text = eval(user_text)
     except Exception:
        pass 
     
    # Classify the text when the user clicks the 'Check' button
     if st.button("Predict your Cuisine"):
        st.write("Your list of items:")
        st.write(user_text)
        user_text=preprocess(user_text)
        x=vectorizer.transform([user_text])
        prediction = model.predict(x)[0]
        st.write("Your Predicted Cuisine is ",reversed_dict[prediction])
        #st.write(prediction[0])
#streamlit run streamlit.py
# Run the app
 
if __name__ == '__main__':
     main()
     