import pandas as pd
import numpy as np

def izracunaj_altman_zscore(finansijski_podaci):
    """
    finansijski_podaci: dict sa ključevima:
    X1 = radni_kapital / ukupna_aktiva
    X2 = zadrzana_dobit / ukupna_aktiva
    X3 = dobit_prije_kamata_i_poreza / ukupna_aktiva
    X4 = trzisna_vrijednost_kapitala / ukupne_obaveze
    X5 = prihod / ukupna_aktiva
    """
    X1 = finansijski_podaci['X1']
    X2 = finansijski_podaci['X2']
    X3 = finansijski_podaci['X3']
    X4 = finansijski_podaci['X4']
    X5 = finansijski_podaci['X5']
    
    z_score = (1.2 * X1) + (1.4 * X2) + (3.3 * X3) + (0.6 * X4) + (1.0 * X5)
    return z_score

def klasifikuj_rizik(z_score):
    if z_score > 2.9:
        return "Safe (nizak rizik)"
    elif z_score > 1.8:
        return "Grey zone (umjeren rizik)"
    else:
        return "Distress (visok rizik)"

# Test primjer (kasnije ćeš ovo zamijeniti sa stvarnim podacima)
if __name__ == "__main__":
    testni_podaci = {
        'X1': 0.2,   # radni kapital / aktiva
        'X2': 0.1,   # zadrzana dobit / aktiva
        'X3': 0.15,  # EBIT / aktiva
        'X4': 0.8,   # trzisna vrijednost / obaveze
        'X5': 1.2    # prihod / aktiva
    }
    z = izracunaj_altman_zscore(testni_podaci)
    rizik = klasifikuj_rizik(z)
    print(f"Z-Score: {z:.2f}")
    print(f"Rizik: {rizik}")