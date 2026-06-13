# 🏦 Digitalni kreditni službenik

Sistem za procjenu rizika bankrota kompanije koristeći **Altman Z-Score** i **XGBoost AI** model.

## 📌 O projektu

Ovaj alat omogućava kreditnim analitičarima i portfolio menadžerima da brzo procijene rizik od bankrota kompanije. 
Kombinuje klasičnu Altman Z-Score formulu sa naprednim XGBoost mašinskim učenjem.

## 🚀 Linkovi

- **Web aplikacija:** [https://huggingface.co/spaces/cvorovic03/kreditni-scoring](https://huggingface.co/spaces/cvorovic03/kreditni-scoring)
- **GitHub repozitorijum:** [https://github.com/cvorovic03/kreditni-scoring](https://github.com/cvorovic03/kreditni-scoring)

## ✨ Funkcionalnosti

| Funkcija | Opis |
|----------|------|
| 📐 Altman Z-Score | Računanje originalne formule sa klasifikacijom (Safe/Grey/Distress) |
| 🤖 XGBoost AI | Hibridni model koji unapređuje predikciju |
| 🔍 SHAP objašnjenja | Objašnjenje zašto je kompanija rizična |
| 📂 CSV upload | Analiza više kompanija odjednom |
| 🔥 Heatmap | Vizuelni prikaz rizika sa bojama |
| 📊 Poređenje modela | Uporedni prikaz Altman vs XGBoost |

## 🛠️ Tehnologije

- **Python 3.9**
- **Streamlit** – web interfejs
- **XGBoost** – mašinsko učenje
- **SHAP** – objašnjivost modela
- **Pandas, NumPy** – obrada podataka
- **Plotly** – vizualizacije

## 📊 Altman Z-Score formula

Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

**Tumačenje:**
- Z > 2.9 → Safe (nizak rizik) 🟢
- 1.8 < Z < 2.9 → Grey zone (umjeren rizik) 🟠
- Z < 1.8 → Distress (visok rizik) 🔴
