import numpy as np
import entropy.entropy as ee

# --- 1. Inspecter la signature réelle ---
print("-" * 30)
print("DOCSTRING DE LA FONCTION :")
# Ceci va nous dire EXACTEMENT quels arguments sont attendus
print(ee.compute_entropy.__doc__) 
print("-" * 30)

# --- 2. Préparation des données (avec Sécurité Anti-Crash) ---
def get_safe_data(n_samples=1000):
    # On génère des données
    x = np.random.randn(n_samples, 1)
    
    # ASTUCE PRO : On ajoute un bruit minuscule (jitter)
    # Cela empêche log(0) si jamais deux valeurs sont identiques
    x += np.random.normal(0, 1e-10, x.shape)
    
    # Transpose (Features, Samples) + Contigu
    x_c = np.ascontiguousarray(x.T, dtype=np.float64)
    return x_c

data = get_safe_data()
print(f"Données : forme {data.shape}, type {data.dtype}")

# --- 3. Augmenter la verbosité au maximum ---
# Cela forcera la lib C à dire POURQUOI elle plante
try:
    ee.set_verbosity(2) # Ou 3 si possible
    print("Verbosité augmentée.")
except Exception:
    pass

# --- 4. Test progressif ---
print("\n--- Tentative 1 : Standard ---")
try:
    # k=3 est souvent plus sûr que k=5 pour commencer
    res = ee.compute_entropy(data, 3)
    print(f"Résultat : {res}")
except Exception as e:
    print(f"Erreur Python : {e}")

print("\n--- Tentative 2 : Avec sélection d'algo (si dispo) ---")
# Parfois il faut dire quel algo utiliser avant
if hasattr(ee, 'choose_algorithm'):
    try:
        # Essayons d'autres types (souvent 0, 1 ou 2)
        # Le log disait "type 4", essayons de changer ça
        print("Changement d'algorithme vers type 1...")
        ee.choose_algorithm(1) 
        res = ee.compute_entropy(data, 3)
        print(f"Résultat Algo 1 : {res}")
    except Exception as e:
        print(f"Erreur Algo : {e}")