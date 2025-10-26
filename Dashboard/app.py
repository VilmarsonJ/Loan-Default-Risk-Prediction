# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# 1️⃣ PAGE CONFIG
st.set_page_config(
    page_title="Prédiction du Risque de Crédit",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Prédiction du Risque de Défaut de Prêt")
st.markdown("""
Bienvenue sur le dashboard de prédiction du risque de crédit.  
Estimez le **risque de défaut de prêt** pour un nouveau client de manière rapide et fiable.
""")

# 2️⃣ CHEMINS ROBUSTES (utilise Pathlib pour éviter les erreurs)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"
DATA_DIR = BASE_DIR / "Data" / "processed"

# Vérification des chemins
st.sidebar.write(f"📁 Dossier modèles : `{MODELS_DIR}`")

# 3️⃣ CHARGER LES MODÈLES
try:
    preprocessor = joblib.load(MODELS_DIR / "preprocessor_final.pkl")
    model_final = joblib.load(MODELS_DIR / "model_final.joblib")
    st.sidebar.success("✅ Modèles chargés avec succès !")
except FileNotFoundError:
    st.error("❌ Impossible de charger les fichiers du modèle. Vérifie que `Models/` est au même niveau que `app.py`.")
    st.stop()

# 4️⃣ ENTRÉE UTILISATEUR
st.sidebar.header("🧍 Informations du Client")
age = st.sidebar.number_input("Âge", 18, 100, 30)
income = st.sidebar.number_input("Revenu annuel (USD)", 0, step=1000, value=50000)
loan_amount = st.sidebar.number_input("Montant du prêt (USD)", 0, step=500, value=10000)
credit_score = st.sidebar.number_input("Score de crédit", 300, 850, 600)
months_employed = st.sidebar.number_input("Durée d'emploi (mois)", 0, 600, 12)
num_credit_lines = st.sidebar.number_input("Nombre de lignes de crédit", 0, 20, 3)
interest_rate = st.sidebar.number_input("Taux d'intérêt (%)", 0.0, 50.0, 10.0)
loan_term = st.sidebar.number_input("Durée du prêt (mois)", 1, 360, 36)
dti_ratio = st.sidebar.number_input("Ratio dette/revenu (DTI)", 0.0, 5.0, 0.4)
education = st.sidebar.selectbox("Niveau d'éducation", ["High School", "Bachelor", "Master", "PhD"])
employment_type = st.sidebar.selectbox("Type d'emploi", ["Temps plein", "Temps partiel", "Indépendant", "Sans emploi"])
marital_status = st.sidebar.selectbox("Situation matrimoniale", ["Célibataire", "Marié", "Divorcé", "Veuf"])
has_mortgage = st.sidebar.selectbox("Possède une hypothèque ?", ["Oui", "Non"])
has_dependents = st.sidebar.selectbox("A des personnes à charge ?", ["Oui", "Non"])
loan_purpose = st.sidebar.selectbox("Objet du prêt", ["Autre", "Éducation", "Voiture", "Entreprise", "Maison"])
has_cosigner = st.sidebar.selectbox("A un co-emprunteur ?", ["Oui", "Non"])

# 5️⃣ DATAFRAME CLIENT
client_data = pd.DataFrame([{
    "Age": age,
    "Income": income,
    "LoanAmount": loan_amount,
    "CreditScore": credit_score,
    "MonthsEmployed": months_employed,
    "NumCreditLines": num_credit_lines,
    "InterestRate": interest_rate,
    "LoanTerm": loan_term,
    "DTIRatio": dti_ratio,
    "Education": education,
    "EmploymentType": employment_type,
    "MaritalStatus": marital_status,
    "HasMortgage": has_mortgage,
    "HasDependents": has_dependents,
    "LoanPurpose": loan_purpose,
    "HasCoSigner": has_cosigner
}])

# 6️⃣ PRÉPROCESSING
try:
    client_processed = preprocessor.transform(client_data)
except Exception as e:
    st.error(f"Erreur lors du preprocessing : {e}")
    st.stop()

# 7️⃣ PRÉDICTION
pred_prob = model_final.predict_proba(client_processed)[:, 1][0]
pred_label = model_final.predict(client_processed)[0]
pred_label_fr = "Remboursé" if pred_label == 0 else "Défaut"

# 8️⃣ AFFICHAGE RÉSULTAT
st.subheader("🎯 Résultat de la Prédiction")
st.metric(label="Étiquette de risque", value=pred_label_fr, delta=f"{pred_prob:.2%} probabilité de défaut")

# 9️⃣ COMPARAISON CONTEXTUELLE
try:
    loan_data = pd.read_csv(DATA_DIR / "Loan_clean.csv")

    st.write("### 📊 Comparaison du Score de Crédit")
    st.bar_chart(loan_data["CreditScore"].append(pd.Series([credit_score])).value_counts().sort_index())

    st.write("### 📉 Comparaison du Ratio DTI")
    st.bar_chart(loan_data["DTIRatio"].append(pd.Series([dti_ratio])).value_counts().sort_index())

except FileNotFoundError:
    st.warning("⚠️ Le fichier `Loan_clean.csv` est introuvable dans `Data/processed`.")    

# 🔚 FOOTER
st.markdown("""
---
💡 *Ce dashboard permet aux institutions financières d'évaluer rapidement le risque de défaut d’un client.*  
📊 *Les résultats sont générés à partir d’un modèle de machine learning Random Forest optimisé.*
""")
