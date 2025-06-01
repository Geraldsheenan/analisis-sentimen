# main.py
import streamlit as st

# This MUST be the first Streamlit command
st.set_page_config(page_title="SVM Review Classifier", layout="centered")

# Now import other libraries
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import os

# Configure NLTK data path
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data with better error handling
required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']

for resource in required_nltk_data:
    try:
        nltk.download(resource, download_dir=nltk_data_path, quiet=True)
    except Exception as e:
        st.warning(f"Failed to download NLTK resource '{resource}': {str(e)}")

# ==== 1. Enhanced Dataset (20 samples) ====
data = {
    'text': [
        'This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.',
        'A decent film with some good moments, though it dragged in the middle.',
        'The cinematography was beautiful but the story was confusing and hard to follow.',
        'One of the worst movies I have ever seen. Terrible acting and a nonsensical plot.',
        'The lead actors had great chemistry, making their relationship believable and compelling.',
        'Mediocre at best. I expected more from such a talented director.',
        'A masterpiece of modern cinema. Every frame was perfect.',
        'Not terrible, but not great either. Just okay for a one-time watch.',
        'The special effects were amazing, though the dialogue felt forced at times.',
        'Boring from start to finish. I nearly fell asleep multiple times.',
        'The soundtrack alone makes this movie worth watching. Simply breathtaking!',
        'An uneven experience - some scenes were brilliant while others fell completely flat.',
        'The humor landed perfectly and I laughed throughout the entire film.',
        'Visually stunning but emotionally empty. All style, no substance.',
        'A heartwarming story with genuine emotional depth. Highly recommended!',
        'The pacing was terrible - too slow in the beginning and too rushed at the end.',
        'Fresh and original take on the genre. Refreshing to see something new!',
        'The plot twists were predictable and the characters were one-dimensional.',
        'Perfect balance of action, drama, and comedy. Something for everyone!',
        'Technically impressive but failed to connect with me on an emotional level.'
    ],
    'label': [
        'Positif', 'Netral', 'Netral', 'Netral', 'Positif',
        'Netral', 'Positif', 'Netral', 'Netral', 'Netral',
        'Positif', 'Netral', 'Positif', 'Netral', 'Positif',
        'Netral', 'Positif', 'Netral', 'Positif', 'Netral'
    ]
}
df = pd.DataFrame(data)

# ==== 2. Preprocessing Function ====
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'@\w+|\#', '', text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenization with fallback
        try:
            tokens = nltk.word_tokenize(text)
        except:
            tokens = text.split()  # Basic fallback
            
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    except Exception as e:
        st.warning(f"Text processing warning: {str(e)}")
        return text.lower()  # Minimal processing fallback

df['clean_text'] = df['text'].apply(preprocess_text)

# ==== 3. Model Training ====
X = df['clean_text']
y = df['label']

# TF-IDF + SVM Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(kernel='linear', probability=True))
])

model.fit(X, y)

# ==== 4. Streamlit App ====
st.title("📊 Pengecekan Review Positif atau Netral")
st.write("Masukkan review produk lalu sistem akan mengklasifikasikan apakah review bersifat **Positif** atau **Netral**.")

# Input dari pengguna
user_input = st.text_area("Masukkan review Anda di sini:", height=150)

if st.button("🔍 Cek Sentimen"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan kalimat terlebih dahulu.")
    else:
        clean_input = preprocess_text(user_input)
        prediction = model.predict([clean_input])[0]
        probas = model.predict_proba([clean_input])[0]
        proba_dict = dict(zip(model.classes_, probas))
        
        # Display results with better formatting
        st.markdown("### 🔎 Hasil Prediksi:")
        
        # Color coding for prediction
        if prediction == 'Positif':
            st.success(f"✅ **Positif** ({(proba_dict['Positif']*100):.1f}% confidence)")
        else:
            st.info(f"🟢 **Netral** ({(proba_dict['Netral']*100):.1f}% confidence)")
        
        # Probability visualization
        st.markdown("### 📊 Probabilitas Klasifikasi:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Positif", value=f"{proba_dict['Positif']*100:.1f}%")
        with col2:
            st.metric(label="Netral", value=f"{proba_dict['Netral']*100:.1f}%")
        
        # Analyze positive and neutral words
        st.markdown("### 🔍 Kata Kunci yang Terdeteksi")
        
        # Define word lists (expanded based on your training data)
        positive_words = ['fantastic', 'superb', 'great', 'perfect', 'amazing', 
                         'breathtaking', 'brilliant', 'heartwarming', 'recommended',
                         'refreshing', 'compelling', 'believable', 'masterpiece',
                         'excellent', 'wonderful', 'enjoyable', 'impressive',
                         'engaging', 'outstanding', 'splendid', 'remarkable']
        
        neutral_words = ['decent', 'mediocre', 'okay', 'average', 'confusing',
                        'predictable', 'uneven', 'boring', 'forced', 'rushed',
                        'dragged', 'nonsensical', 'empty', 'acceptable', 'standard',
                        'ordinary', 'typical', 'moderate', 'adequate', 'passable']
        
        # Tokenize and find matches
        tokens = clean_input.split()
        found_positive = [word for word in tokens if word in positive_words]
        found_neutral = [word for word in tokens if word in neutral_words]
        
        # Display found words with consistent formatting
        col1, col2 = st.columns(2)
        with col1:
            if found_positive:
                st.markdown(f"**Kata Positif:** {', '.join(found_positive)}")
            else:
                st.markdown("**Kata Positif:** Tidak terdeteksi")
        
        with col2:
            if found_neutral:
                st.markdown(f"**Kata Netral:** {', '.join(found_neutral)}")
            else:
                st.markdown("**Kata Netral:** Tidak terdeteksi")
        
        # Show cleaned text for debugging
        st.markdown("### 🧹 Teks yang Diproses:")
        st.code(clean_input)
        
# ==== 5. Show training data ====
with st.expander("📋 Data Latih yang Digunakan (20 Contoh)"):
    st.dataframe(df[['text', 'label']], height=400)
    st.write(f"Distribusi Label: {df['label'].value_counts().to_dict()}")

# ==== 6. Test examples ====
st.markdown("### 🧪 Contoh Uji Coba")
test_examples = [
    "This is the worst product I've ever bought!",
    "I'm so happy with my purchase!",
    "It's acceptable but could be better",
    "Absolutely brilliant!",
    "Meh, it's just okay I guess"
]

cols = st.columns(len(test_examples))
for col, example in zip(cols, test_examples):
    with col:
        if st.button(example[:15]+"..."):
            clean_input = preprocess_text(example)
            prediction = model.predict([clean_input])[0]
            probas = model.predict_proba([clean_input])[0]
            proba_dict = dict(zip(model.classes_, probas))
            
            st.write(f"**Prediksi:** {prediction}")
            st.write(f"Positif: {proba_dict['Positif']*100:.1f}%")
            st.write(f"Netral: {proba_dict['Netral']*100:.1f}%")
