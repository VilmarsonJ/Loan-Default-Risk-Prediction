# **Prédiction du Risque de Défaut de Prêt à l’aide du Machine Learning**
---
### ***Une étude basée sur les données pour évaluer le risque de crédit et optimiser les décisions de financement***

![photo](../Images/image12.webp)

---
> ### **Prepared and presented by Data Scientists :**
> #### **Vilmarson JULES** & **Rodolphe CHARLES**  
>**LinkedIn:** [Vilmarson JULES](https://www.linkedin.com/in/jules-vilmarson-2a68a5294/) | [Rodolphe CHARLES](https://www.linkedin.com/in/rodolphe-charles-81b75924b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
> #### **Akademi Education – Octorbre 2025**

## **Brève Description**
---

![photo](../Images/image19.jpg)

Les défauts de remboursement constituent un **défi majeur pour les banques**, affectant **la rentabilité**, **la stabilité financière** et **l’accès au crédit**. Évaluer efficacement quels candidats sont susceptibles de faire défaut est essentiel pour **réduire les risques** et **prendre des décisions de prêt éclairées**. Cette étude utilise des **modèles de machine learning** pour prédire le risque de défaut de remboursement des prêts. 

En analysant les **demandes de prêt historiques** et les **profils des clients**, les modèles permettent d’identifier les **emprunteurs potentiellement à haut risque**, offrant ainsi aux institutions financières la possibilité de **prendre des décisions de crédit basées sur les données**, réduire les pertes financières et améliorer la stabilité des portefeuilles**, tant au niveau mondial que dans des contextes comme Haïti, où l’accès au crédit fiable est limité et les défauts peuvent fortement impacter la stabilité financière.

- ### ***Les objectifs principaux de ce projet sont :***

#### **1. Prédire le risque de défaut de prêt** afin de gérer de manière proactive les emprunteurs à haut risque.  
#### **2. Soutenir les décisions stratégiques de prêt** grâce à des **insights exploitables**.  
#### **3. Réduire les prêts non performants (NPL)** et améliorer la **stabilité financière globale**.  
#### **4. Améliorer les processus d’évaluation du crédit** par une analyse **basée sur les données**.

Les insights issus de cette analyse contribuent à **réduire les prêts non performants**, **optimiser les stratégies de prêt** et **renforcer la stabilité financière globale**, démontrant l’impact concret de l’analytics prédictive dans le secteur bancaire.


## **Business Problem (Problématique du Crédit)**
---


![photo](../Images/image04.jpeg)


Le défaut de remboursement des prêts reste l’un des défis les plus critiques pour les institutions financières dans le monde. Lorsqu’un emprunteur ne rembourse pas son prêt, les banques subissent des **pertes financières directes**, et peuvent faire face à des **risques de liquidité**.  

De plus, un taux élevé de défaut peut **saper la confiance dans le secteur bancaire**, limiter **l’accès au crédit pour d’autres clients**, et compliquer la **conformité réglementaire**. Ce problème est particulièrement marqué dans les marchés émergents, où l’accès au crédit fiable est limité et l’écosystème financier est plus sensible aux défauts.  

**Problème central : Comment les institutions financières peuvent-elles identifier et analyser de manière proactive les emprunteurs présentant un risque élevé de défaut ?**  

Résoudre ce problème permet aux banques et prêteurs de :  
- Prendre des **décisions de prêt basées sur les données**.  
- Concevoir des **interventions ciblées** pour les emprunteurs à risque.  
- Réduire les **pertes financières** et **stabiliser les portefeuilles de prêts**.  
- Soutenir les initiatives de **gestion stratégique et réglementaire du risque**.  

Ce projet aborde cette problématique en exploitant des **données historiques sur les prêts et les clients** avec des **modèles de machine learning**, transformant des patterns complexes en insights exploitables pour **prévenir les défauts avant qu’ils ne surviennent**.

## **The Data**
---

Le jeu de données utilisé dans ce projet provient du **[Loan Default Prediction Challenge](https://www.kaggle.com/datasets/nikhil1e9/loan-default?)** sur Kaggle, basé à l’origine sur l’étude de cas **Loan Default Prediction de Coursera**. Il représente des **données financières réelles de prêts** provenant de banques et d’institutions de crédit accordant des prêts aux particuliers et aux entreprises.  

![photo](../Images/image01.png)

Ces données contiennent des informations essentielles sur les emprunteurs et les prêts, incluant **les revenus, les types d’emploi, les montants de prêt, l’historique de crédit et les résultats de remboursement**, offrant ainsi une vision complète des facteurs influençant le **risque de défaut**.  


## **Methods**

---

Dans ce projet, nous adoptons une approche de **Predictive Analytics** centrée sur la **classification binaire** pour évaluer le risque de défaut de prêt.

***Nous nous appuyons sur un écosystème Python robuste pour la data science, incluant :***  
- **Data manipulation & analysis :** `pandas`, `numpy`  
- **Visualization :** `matplotlib`, `seaborn`  
- **Machine learning :** `scikit-learn`, `xgboost`, `lightgbm`  
- **Environment & version control :** Git, GitHub, Jupyter Notebook  

***Nous avons poursuivi ces etapes qui ci-dessous pour realiser cette etude :***
### I – Data Understanding
### II – Data Cleaning
### III – Exploratory Data Analysis (EDA)
### IV – Data Preprocessing & Feature Engineering
### V – Modeling
### VII – Insights & Interpretation


## **Business Understanding**
---

![photo](../Images/image20.jpg)


Les défauts de paiement représentent un défi majeur pour les institutions financières dans le mondes, en particulier celles des haitiens, impactant la **rentabilité, le risque de crédit et la stabilité financière**.


Cette étude se concentre sur le contexte des ***services bancaires et financiers***, avec des applications dans **l’évaluation du risque de crédit, la stratégie de prêt et la conformité réglementaire**.

#### ***Le public cible principal comprend :***  
- **Banques et institutions financières :** optimiser les prêts et réduire les pertes.  
- **Banques centrales et régulateurs :** surveiller et gérer le risque systémique.  
- **Fintech et prêteurs :** mettre en œuvre des processus d’approbation de crédit basés sur les données.  

En appliquant des **modèles de machine learning avancés**, cette étude traduit des données complexes en **indicateurs quantitatifs de risque**. Cela permet aux parties prenantes de **détecter proactivement les emprunteurs à haut risque**, de cibler plus efficacement les interventions et de prendre des **décisions stratégiques basées sur des preuves**.

#### ***L’impact concret de ce projet est significatif :***  
**1.** **Réduire les prêts non performants (NPLs)** et minimiser les pertes financières.  
**2.** Fournir des **scores de risque pour chaque emprunteur** afin de guider les décisions de prêt.  
**3.** Permettre des **interventions ciblées** pour les emprunteurs à risque de défaut.  
**4.** Soutenir les **rapports réglementaires** et renforcer la **stabilité financière globale**.

>La **motivation** de ce projet est de montrer comment la **data science peut transformer les pratiques de prêt traditionnelles**, améliorer la qualité des décisions de crédit et apporter une **valeur business tangible** aux institutions financières, régulateurs et prêteurs.

## **III – Exploratory Data Analysis (EDA)**
---

1. **Performance Globale des Prêts (Diagramme circulaire)**  
   Le diagramme circulaire montre qu’environ **88% des emprunteurs remboursent leurs prêts à temps**, tandis que **12% font défaut**. Même si la majorité des clients sont fiables, ce **segment minoritaire mais critique** représente un **risque financier important** pour les banques. Identifier proactivement ces emprunteurs peut **prévenir des pertes** et stabiliser le portefeuille de prêts.
la majorité des prêts (**88,4 %**) sont **remboursés à temps**, ce qui indique que le portefeuille global est relativement sain.  

Cependant, un **segment critique de 11,8 %** correspond à des **prêts à risque (défauts de remboursement)**.  
Cette information est essentielle pour les institutions financières, y compris en Haïti, afin de :  
- **Identifier les risques globaux du portefeuille**,  
- **Prioriser les actions de suivi et de prévention**,  
- **Allouer efficacement les ressources pour gérer les défauts potentiels**.




 ## **V -  Modeling (Modélisation) and Evaluation**
Nous avons construit et évalue des **modèles de machine learning** pour prédire le risque de défaut de remboursement des prêts.

### **1️. Logistic Regression (Baseline)**
### ***2 Optimized Logistic Regression***
#### ***3 Optimized Random Forest***

## **Remarks and Models comparason**

Après avoir construit et évalué nos modèles principaux , **Régression Logistique Optimisée** et **Random Forest Optimisé** , nous pouvons comparer leurs performances et comprendre leurs rôles spécifiques dans la prédiction du risque de défaut.

| Modèle | Accuracy | Precision | Recall | F1-score | AUC |
|:-------|:---------:|:----------:|:--------:|:----------:|:----:|
| Régression Logistique Optimisée (identique à la baseline) | 0.69 | 0.23 | 0.70 | 0.34 | 0.76 |
| Random Forest Optimisé | 0.73 | 0.25 | 0.63 | 0.35 | 0.75 |

- Le **Random Forest Optimisé** présente la meilleure précision globale et un équilibre correct entre rappel et F1-score. Il est particulièrement utile pour identifier les clients à risque et limiter les pertes potentielles dans le portefeuille de prêts d’une banque haïtienne.  
- La **Régression Logistique Optimisée** n’améliore pas la baseline (c’est exactement le même modèle), mais elle reste **interprétable et rapide à déployer**, idéale pour les environnements où la transparence des décisions est cruciale, comme dans les banques ou coopératives de microfinance à Port-au-Prince ou Jacmel.  
- Comme les performances sont proches, nous avons choisi de **sauvegarder les deux modèles** pour garantir la **reproductibilité et la flexibilité future**.  
  - **Régression Logistique :** `../Models/logit_optimized_20251023_1225.joblib` 
  - **Random Forest :** `../Models/rf_optimized_20251023_1225.joblib`  
- Cela permet à toute institution souhaitant reproduire nos analyses ou tester le modèle sur de nouvelles données de **choisir le modèle le plus adapté à leur contexte opérationnel**.


## Business Recommendations
---

## Conclusion
---

**Key findings include:**


## Next Steps
---
​

## Full Analysis & Contact
---
Explore the complete analytical workflow in the [Jupyter Notebook](./churn_index.ipynb) or review the [presentation slides](./ChurnInsight_Presentation.pdf) for a high-level summary key findings and actionable recommendations.

For professional inquiries, collaboration opportunities, or discussions about the methodology and insights, reach out to:

**Vilmarson JULES & Rodolphe CHARLES**  
Data Scientists 
📧 [vilmarsonjules22@gmail.com](mailto:vilmarsonjules22@gmail.com)  
![Vilmarson JULES – Data Science & AI Student](![photo](../Images/image26.jpg))

## Repository Structure

