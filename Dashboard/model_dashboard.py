# model_dashboard.py
# --------------------------
# Dashboard interactif pour la visualisation des prédictions de défaut de prêt
# Créé pour le projet Loan Default Risk Prediction
# Auteur : Vilmarson
# Objectif : Fournir aux décideurs financiers une interface claire pour interpréter
#            le risque de défaut selon les modèles Random Forest et Logistic Regression.

import pandas as pd
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
from joblib import load

# --------------------------
# Charger les données
# --------------------------
data_path = "../Data/processed/predictions.csv"
df = pd.read_csv(data_path)

# --------------------------
# Initialiser l'application Dash
# --------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard Prêt - Risk Analysis"

# --------------------------
# KPI Cards
# --------------------------
total_loans = len(df)
default_rate_rf = round(df['RF_Pred'].mean() * 100, 2)
default_rate_logit = round(df['Logit_Pred'].mean() * 100, 2)
avg_rf_prob = round(df['RF_Prob'].mean() * 100, 2)

cards = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H5("Total de prêts analysés", className="card-title"),
            html.H2(f"{total_loans}", className="card-text")
        ])
    ]), width=3),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H5("Taux de défaut prédit - RF", className="card-title"),
            html.H2(f"{default_rate_rf} %", className="card-text")
        ])
    ]), width=3),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H5("Taux de défaut prédit - Logit", className="card-title"),
            html.H2(f"{default_rate_logit} %", className="card-text")
        ])
    ]), width=3),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H5("Probabilité moyenne de défaut - RF", className="card-title"),
            html.H2(f"{avg_rf_prob} %", className="card-text")
        ])
    ]), width=3)
])

# --------------------------
# Graphiques
# --------------------------
# Histogramme des probabilités RF
fig_rf_prob = px.histogram(df, x="RF_Prob", nbins=30,
                           title="Distribution des probabilités de défaut (Random Forest)",
                           labels={"RF_Prob": "Probabilité de défaut"},
                           color_discrete_sequence=["#EF553B"])

# Comparaison des prédictions Logit vs RF
fig_compare = px.scatter(df, x="Logit_Pred", y="RF_Pred",
                         title="Comparaison des prédictions Logit vs Random Forest",
                         labels={"Logit_Pred": "Logit Pred", "RF_Pred": "RF Pred"},
                         color="Default_label_fr",
                         color_discrete_map={"Remboursé à temps (Prêt sain)":"green",
                                             "Défaut de paiement (Prêt risqué)":"red"})

# Scatter plot Age vs DTI
fig_age_dti = px.scatter(df, x="Age", y="DTIRatio",
                         color="RF_Pred",
                         size="LoanAmount",
                         hover_data=["LoanID", "Income", "CreditScore"],
                         title="Age vs DTIRatio selon le risque prédit")

# --------------------------
# DataTable interactive
# --------------------------
table = dash_table.DataTable(
    id='loan_table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
    filter_action="native",
    sort_action="native",
    page_size=10,
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'left', 'padding': '5px'},
    style_header={'backgroundColor': 'lightblue', 'fontWeight': 'bold'}
)

# --------------------------
# Layout de l'application
# --------------------------
app.layout = dbc.Container([
    html.H1("Dashboard d'analyse du risque de défaut", className="text-center my-4"),
    html.P(
        "Ce dashboard permet d'explorer les prédictions de défaut de prêt selon les modèles "
        "Random Forest et Logistic Regression. Les décideurs peuvent visualiser le taux de défaut, "
        "les probabilités moyennes et filtrer les données par caractéristiques des emprunteurs.",
        className="text-center"
    ),
    cards,
    html.Hr(),
    dcc.Graph(figure=fig_rf_prob),
    dcc.Graph(figure=fig_compare),
    dcc.Graph(figure=fig_age_dti),
    html.Hr(),
    html.H3("Tableau interactif des prêts", className="my-3"),
    table
], fluid=True)

# --------------------------
# Lancer le serveur
# --------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
