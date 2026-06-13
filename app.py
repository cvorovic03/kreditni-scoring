import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Ucitavanje ML modela
ml_model_available = os.path.exists('xgboost_model.pkl')

if ml_model_available:
    with open('xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

st.set_page_config(page_title="Digitalni kreditni sluzbenik", layout="wide")

def izracunaj_altman_zscore(X1, X2, X3, X4, X5):
    return (1.2 * X1) + (1.4 * X2) + (3.3 * X3) + (0.6 * X4) + (1.0 * X5)

def klasifikuj_altman(z_score):
    if z_score > 2.9:
        return "Safe", "🟢", "#2ecc71"
    elif z_score > 1.8:
        return "Grey zone", "🟠", "#f39c12"
    else:
        return "Distress", "🔴", "#e74c3c"

def predikcija_xgboost(X1, X2, X3, X4, X5):
    if not ml_model_available:
        return "Nije dostupno", "❌", "#95a5a6"
    features = np.array([[X1, X2, X3, X4, X5]])
    features_scaled = scaler.transform(features)
    pred = xgb_model.predict(features_scaled)[0]
    if pred == 2:
        return "Safe", "🟢", "#2ecc71"
    elif pred == 1:
        return "Grey zone", "🟠", "#f39c12"
    else:
        return "Distress", "🔴", "#e74c3c"

st.title("🏦 Digitalni kreditni sluzbenik")
st.markdown("**Uporedi Altman Z-Score vs XGBoost AI model**")

# Kreiranje tabova
tab1, tab2, tab3 = st.tabs(["📝 Pojedinačni unos", "📂 Upload CSV fajla", "🤖 Poređenje modela"])

# ========== TAB 1: Pojedinačni unos ==========
with tab1:
    st.subheader("📊 Unesi finansijske podatke kompanije")
    
    col1, col2 = st.columns(2)
    
    with col1:
        X1 = st.number_input("X1 = Radni kapital / Ukupna aktiva", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        X2 = st.number_input("X2 = Zadrzana dobit / Ukupna aktiva", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        X3 = st.number_input("X3 = EBIT / Ukupna aktiva", min_value=0.0, max_value=0.5, value=0.15, step=0.01)
    
    with col2:
        X4 = st.number_input("X4 = Trzisna vrijednost / Ukupne obaveze", min_value=0.0, max_value=3.0, value=0.8, step=0.1)
        X5 = st.number_input("X5 = Prihod / Ukupna aktiva", min_value=0.0, max_value=2.0, value=1.2, step=0.1)
    
    # Izracun
    z_score = izracunaj_altman_zscore(X1, X2, X3, X4, X5)
    altman_kategorija, altman_emoji, altman_boja = klasifikuj_altman(z_score)
    xgb_kategorija, xgb_emoji, xgb_boja = predikcija_xgboost(X1, X2, X3, X4, X5)
    
    st.markdown("---")
    st.subheader("📊 REZULTATI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: {altman_boja}; padding: 20px; border-radius: 15px; text-align: center; color: white;">
            <span style="font-size: 40px;">{altman_emoji}</span>
            <h2 style="margin: 0; color: white;">📐 Altman Z-Score</h2>
            <h1 style="margin: 0; color: white; font-size: 48px;">{z_score:.2f}</h1>
            <h3 style="margin: 0; color: white;">{altman_kategorija}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: {xgb_boja}; padding: 20px; border-radius: 15px; text-align: center; color: white;">
            <span style="font-size: 40px;">{xgb_emoji}</span>
            <h2 style="margin: 0; color: white;">🤖 XGBoost AI</h2>
            <h1 style="margin: 0; color: white; font-size: 48px;">{xgb_kategorija}</h1>
            <p style="margin: 10px 0 0 0;">(predikcija na osnovu istih podataka)</p>
        </div>
        """, unsafe_allow_html=True)
    
    if ml_model_available:
        if altman_kategorija == xgb_kategorija:
            st.success(f"✅ Modeli se slažu! Oba predviđaju: {altman_kategorija}")
        else:
            st.warning(f"⚠️ Modeli se ne slažu! Altman kaže {altman_kategorija}, XGBoost kaže {xgb_kategorija}")
    
    st.subheader("📊 Altman Z-Score vizuelni prikaz")
    progress_value = min(z_score / 4.0, 1.0)
    st.progress(progress_value)
    st.caption(f"0 (Distress) {' ' * 25} 1.8 {' ' * 20} 2.9 {' ' * 15} 4.0 (Safe)")

# ========== TAB 2: Upload CSV fajla ==========
with tab2:
    st.subheader("Upload CSV fajla sa više kompanija")
    
    st.info("""
    **Potrebne kolone u CSV fajlu:**
    - `naziv`, `radni_kapital_aktiva`, `zadrzana_dobit_aktiva`, `ebit_aktiva`, `trzisna_vrijednost_obaveze`, `prihod_aktiva`
    """)
    
    uploaded_file = st.file_uploader("Izaberi CSV fajl", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📄 Učitani podaci:")
        st.dataframe(df, use_container_width=True)
        
        # Altman Z-Score
        df['Z_Score'] = df.apply(lambda row: izracunaj_altman_zscore(
            row['radni_kapital_aktiva'],
            row['zadrzana_dobit_aktiva'],
            row['ebit_aktiva'],
            row['trzisna_vrijednost_obaveze'],
            row['prihod_aktiva']
        ), axis=1)
        
        df['Rizik_Altman'] = df['Z_Score'].apply(lambda z: klasifikuj_altman(z)[0])
        
        # XGBoost predikcija (ako postoji)
        if ml_model_available:
            df['Rizik_XGBoost'] = df.apply(lambda row: predikcija_xgboost(
                row['radni_kapital_aktiva'],
                row['zadrzana_dobit_aktiva'],
                row['ebit_aktiva'],
                row['trzisna_vrijednost_obaveze'],
                row['prihod_aktiva']
            )[0], axis=1)
        
        st.subheader("📊 Rezultati:")
        if ml_model_available:
            st.dataframe(df[['naziv', 'Z_Score', 'Rizik_Altman', 'Rizik_XGBoost']], use_container_width=True)
        else:
            st.dataframe(df[['naziv', 'Z_Score', 'Rizik_Altman']], use_container_width=True)
        
        st.subheader("📈 Grafik Z-Score po kompanijama")
        st.bar_chart(df.set_index('naziv')['Z_Score'])
        
        # Heatmap
        st.subheader("🔥 Heatmap rizika (Altman Z-Score)")
        for i, row in df.iterrows():
            z = row['Z_Score']
            if z > 2.9:
                boja = "#2ecc71"
                emoji = "🟢"
            elif z > 1.8:
                boja = "#f39c12"
                emoji = "🟠"
            else:
                boja = "#e74c3c"
                emoji = "🔴"
            
            if ml_model_available:
                xgb_status = f" | XGBoost: {row['Rizik_XGBoost']}"
            else:
                xgb_status = ""
            
            st.markdown(f"""
            <div style="background-color: {boja}; padding: 12px; border-radius: 8px; margin: 8px 0; color: white;">
                {emoji} <b>{row['naziv']}</b> | Z-Score: {z:.2f} | {row['Rizik_Altman']}{xgb_status}
            </div>
            """, unsafe_allow_html=True)

# ========== TAB 3: Poređenje modela ==========
with tab3:
    st.subheader("🤖 Poređenje Altman Z-Score i XGBoost AI modela")
    
    st.markdown("""
    | Karakteristika | Altman Z-Score | XGBoost AI |
    |----------------|----------------|-------------|
    | Princip rada | Fiksna formula (1968) | Mašinsko učenje na podacima |
    | Potrebni podaci | 5 finansijskih pokazatelja | Isti pokazatelji |
    | Objašnjivost | Jasna formula, svaka varijabla ima težinu | SHAP vrijednosti (koji faktor je važan) |
    | Prednost | Jednostavan, transparentan, bez potrebe za treningom | Može uhvatiti nerealne obrasce, prilagođava se podacima |
    | Nedostatak | Fiksan, ne uči se | Potreban za treniranje, manje transparentan |
    """)
    
    st.info("💡 **Preporuka:** Koristite oba modela. Altman daje dobru osnovu, a XGBoost može detektovati dodatne obrasce koje formula ne vidi.")
    
    if ml_model_available:
        st.success("✅ XGBoost model je uspješno učitan i spreman za korištenje.")
    else:
        st.warning("⚠️ XGBoost model nije učitan. Pokreni 'python train_model.py' da ga kreiraš.")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ O aplikaciji")
st.sidebar.markdown("Ova aplikacija poredi klasični Altman Z-Score model sa XGBoost AI modelom za predikciju rizika od bankrota.")
if ml_model_available:
    st.sidebar.success("✅ XGBoost model: AKTIVAN")
else:
    st.sidebar.error("❌ XGBoost model: NIJE AKTIVAN")