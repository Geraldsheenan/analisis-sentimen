# main.py
import streamlit as st
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

# Download required NLTK data with progress feedback
try:
    if not os.path.exists(os.path.join(nltk_data_path, "tokenizers/punkt")):
        nltk.download('punkt', download_dir=nltk_data_path, quiet=False)
    if not os.path.exists(os.path.join(nltk_data_path, "corpora/stopwords")):
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=False)
    if not os.path.exists(os.path.join(nltk_data_path, "corpora/wordnet")):
        nltk.download('wordnet', download_dir=nltk_data_path, quiet=False)
except Exception as e:
    import streamlit as st
    st.error(f"NLTK data download failed: {str(e)}")
# ==== 1. Enhanced Dataset (20 samples) ====
data = {
    'text': [
        'I love this product, it is amazing!',
        'Quite good, I like the experience.',
        'It was okay, nothing special.',
        'I am satisfied with the service.',
        'The experience is neutral, not bad not good.',
        'Good value for money.',
        'It works fine, I think it\'s okay.',
        'Really pleasant and enjoyable!',
        'Just average, neither good nor bad.',
        'Not bad, just neutral.',
        'This is absolutely fantastic! Worth every penny.',
        'The quality exceeded my expectations.',
        'Mediocre at best, I expected more.',
        'Completely disappointed with my purchase.',
        'Works as described, no complaints.',
        'The best product in this category!',
        'Not what I hoped for, but acceptable.',
        'Terrible experience, would not recommend.',
        'Perfect in every way!',
        'It\'s fine, does the job but nothing extraordinary.'
    ],
    'label': [
        'Positif', 'Positif', 'Netral', 'Positif', 'Netral',
        'Positif', 'Netral', 'Positif', 'Netral', 'Netral',
        'Positif', 'Positif', 'Netral', 'Netral', 'Netral',
        'Positif', 'Netral', 'Netral', 'Positif', 'Netral'
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
        
        # Robust tokenization with fallback
        try:
            tokens = nltk.word_tokenize(text)
        except LookupError:
            nltk.download('punkt', download_dir=nltk_data_path)
            tokens = nltk.word_tokenize(text)
        except:
            tokens = text.split()  # Basic fallback
            
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    except Exception as e:
        import streamlit as st
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
st.set_page_config(page_title="SVM Review Classifier", layout="centered")

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
        
        # Show cleaned text for debugging
        st.markdown("### 🧹 Teks yang Diproses:")
        st.code(clean_input)

# ==== 5. Show training data ====
with st.expander("📋 Data Latih yang Digunakan (20 Contoh)"):
    st.dataframe(df[['text', 'label']], height=400)
    st.write(f"Distribusi Label: {df['label'].value_counts().to_dict()}")

# ==== 6. Test examples ====
st.markdown("### � Contoh Uji Coba")
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
