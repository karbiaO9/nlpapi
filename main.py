# -*- coding: utf-8 -*-
import json
import pandas as pd
import re
import os
import nltk
from pathlib import Path
from fastapi import FastAPI, HTTPException
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ========================================
# Data Preparation with Preprocessing
# ========================================
print("Téléchargement des ressources NLTK...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("✓ Ressources NLTK téléchargées")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

DATA_PATH = "data/articles.json"
os.makedirs("data", exist_ok=True)
OUT_PATH = "data/articles_clean.csv"

def preprocess_text(text):
    """
    Prétraitement du texte :
    - Conversion en minuscules
    - Suppression de la ponctuation et caractères spéciaux
    - Tokenization
    - Suppression des stopwords
    - Lemmatisation
    """
    if not isinstance(text, str):
        return ""

    # Conversion en minuscules
    text = text.lower()

    # Suppression des URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Suppression des caractères spéciaux et ponctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Suppression des stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def preprocess_main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    records = []
    for a in articles:
        title = a.get("Title", "")
        body = a.get("Body", "")

        # Prétraitement du CORPS uniquement (plus de contenu)
        body_clean = preprocess_text(body)

        # Vérification que le body n'est pas vide
        if not body_clean or len(body_clean.split()) < 5:
            print(f"⚠ Article ID {a['ID']} a un corps trop court, ajout du titre")
            title_clean = preprocess_text(title)
            body_clean = f"{title_clean} {body_clean}"

        records.append({
            "ID": a["ID"],
            "Title": a.get("Title", ""),      # Titre original
            "Body_Clean": body_clean,          # Corps prétraité
            "Tokens": body_clean.split()       # Tokens pour Word2Vec
        })

    df = pd.DataFrame(records)

    # Statistiques de prétraitement
    df['num_tokens'] = df['Tokens'].apply(len)
    print(f"\n✓ Saved {len(df)} articles → {OUT_PATH}")
    print(f"✓ Statistiques des tokens:")
    print(f"  - Moyenne: {df['num_tokens'].mean():.1f} tokens/article")
    print(f"  - Min: {df['num_tokens'].min()} tokens")
    print(f"  - Max: {df['num_tokens'].max()}")
    print(f"\n✓ Exemple de prétraitement (Article 1):")
    print(f"  Titre: {df.iloc[0]['Title'][:80]}...")
    print(f"  Texte nettoyé: {df.iloc[0]['Body_Clean'][:150]}...")
    print(f"  Nombre de tokens: {df.iloc[0]['num_tokens']}")

    # Sauvegarder
    df_save = df[['ID', 'Title', 'Body_Clean']].copy()
    df_save.to_csv(OUT_PATH, index=False)

    return df

# Exécuter
df = preprocess_main()

# ========================================
# Word2Vec Embeddings
# ========================================
def create_document_embedding(tokens, model, vector_size):
    """
    Créer un embedding de document en moyennant les embeddings Word2Vec des mots.
    Utilise TF-IDF weighting pour donner plus d'importance aux mots rares.
    """
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        # Si aucun mot n'est dans le vocabulaire, retourner un vecteur zéro
        return np.zeros(vector_size)

    # Moyenne des vecteurs de mots
    return np.mean(vectors, axis=0)

def train_w2v_main():
    # Charger les données
    df = pd.read_csv("data/articles_clean.csv")

    # Re-tokeniser le texte nettoyé
    df['Tokens'] = df['Body_Clean'].fillna("").apply(lambda x: x.split())

    # Préparer le corpus pour Word2Vec
    corpus = df['Tokens'].tolist()

    print(f"✓ Corpus préparé: {len(corpus)} documents")
    print(f"✓ Exemple de tokens (Article 1): {corpus[0][:15]}")
    print(f"✓ Taille du vocabulaire total: {len(set([w for doc in corpus for w in doc]))} mots uniques")

    # Entraîner le modèle Word2Vec avec de meilleurs paramètres
    print("\n⏳ Entraînement du modèle Word2Vec...")
    vector_size = 100  # Dimension des embeddings
    window = 10        # Fenêtre de contexte plus large
    min_count = 2      # Au moins 2 occurrences
    workers = 4        # Nombre de threads
    sg = 1            # Skip-gram (meilleur pour petits corpus)
    epochs = 20       # Plus d'époques

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
        negative=5,      # Negative sampling
        ns_exponent=0.75 # Exposant pour negative sampling
    )

    vocab_size = len(model.wv)
    print(f"✓ Modèle entraîné avec {vocab_size} mots dans le vocabulaire")

    if vocab_size < 50:
        print("⚠ ATTENTION: Vocabulaire très petit! Vérifiez le prétraitement.")

    # Créer les embeddings de documents
    print("\n⏳ Création des embeddings de documents...")
    embeddings = []
    empty_count = 0

    for i, tokens in enumerate(corpus):
        emb = create_document_embedding(tokens, model, vector_size)
        if np.all(emb == 0):
            empty_count += 1
            print(f"⚠ Article {i+1} (ID={df.iloc[i]['ID']}): embedding vide!")
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    if empty_count > 0:
        print(f"⚠ {empty_count}/{len(corpus)} articles ont des embeddings vides")

    # Vérifier la variance des embeddings
    variance = np.var(embeddings, axis=0).mean()
    print(f"\n✓ Variance moyenne des embeddings: {variance:.6f}")
    if variance < 0.001:
        print("⚠ ATTENTION: Variance très faible - les embeddings sont trop similaires!")

    # Normalisation L2 pour la similarité cosinus
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Éviter la division par zéro
    embeddings_normalized = embeddings / norms

    # Vérifier la diversité des similarités
    sim_matrix = cosine_similarity(embeddings_normalized)
    np.fill_diagonal(sim_matrix, 0)
    avg_sim = sim_matrix.mean()
    print(f"✓ Similarité moyenne entre documents: {avg_sim:.4f}")
    print(f"✓ Similarité min: {sim_matrix.min():.4f}, max: {sim_matrix.max():.4f}")

    if avg_sim > 0.95:
        print("⚠ ATTENTION: Similarités trop élevées - possible problème d'embedding!")

    # Sauvegarder
    np.save("data/embeddings_w2v.npy", embeddings_normalized)
    model.save("data/word2vec_model.bin")

    print(f"\n✓ Embeddings sauvegardés → data/embeddings_w2v.npy")
    print(f"✓ Forme des embeddings: {embeddings_normalized.shape}")
    print(f"✓ Modèle Word2Vec sauvegardé → data/word2vec_model.bin")


# ========================================
# FastAPI REST API
# ========================================
app = FastAPI(title="Content-Based Recommender API")

# Chargement des données au démarrage
df = pd.read_csv("data/articles_clean.csv")
embeddings = np.load("data/embeddings_w2v.npy")
df["_index"] = range(len(df))
df = df.set_index("ID")

@app.get("/")
def root():
    return {"status": "API running"}

@app.get("/recommend/{article_id}")
def recommend(article_id: int, n: int = 5):
    if article_id not in df.index:
        raise HTTPException(status_code=404, detail="Article not found")

    idx = df.loc[article_id, "_index"]
    query_vec = embeddings[idx].reshape(1, -1)
    scores = cosine_similarity(query_vec, embeddings)[0]

    ranked = scores.argsort()[::-1]
    results = []

    for i in ranked:
        aid = df[df["_index"] == i].index[0]
        if aid == article_id:
            continue

        results.append({
            "ID": int(aid),
            "Title": df.loc[aid, "Title"],
            "Score": float(scores[i])
        })

        if len(results) >= n:
            break

    return {
        "query_title": df.loc[article_id, "Title"],
        "recommendations": results
    }

if __name__ == "__main__":
    preprocess_main()
    train_w2v_main()
