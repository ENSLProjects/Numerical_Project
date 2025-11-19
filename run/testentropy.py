import numpy as np
from bnn_package import prepare_data

# --- PATCH DE COMPATIBILITÉ (Indispensable pour Numpy récent) ---
if not hasattr(np, 'int'):
    setattr(np, 'int', int)
if not hasattr(np, 'float'):
    setattr(np, 'float', float)

import entropy.entropy as ee

# --- PRÉPARATION DES DONNÉES ---
N = 1000
# Génération de X
x_raw = np.random.randn(N)
# Génération de Y (corrélé à X pour avoir une MI non nulle)
y_raw = 0.8 * x_raw + 0.1 * np.random.randn(N)

# Formatage strict
x = prepare_data(x_raw)
y = prepare_data(y_raw)

print(f"Données prêtes : forme {x.shape} (doit être (1, {N}))")

# --- CONFIGURATION ---
ee.set_verbosity(1)

# --- CALCUL ENTROPIE (H) ---
print("-" * 30)
try:
    h = ee.compute_entropy(
        x=x, 
        n_embed=1,   # Dimension de plongement (1 = brut)
        stride=1,    # Pas de décalage
        k=3,         # 3 plus proches voisins
        N_eff=-1     # CRITIQUE : utilise tous les points dispos (évite le crash)
    )
    print(f"Entropie H(X) : {h}")
except Exception as e:
    print(f"Erreur Entropie : {e}")

# --- CALCUL INFO MUTUELLE (MI) ---
print("-" * 30)
try:
    # Elle compare deux signaux bruts X et Y.
    mi_result = ee.compute_MI(
        x=x, 
        y=y, 
        k=3, 
        N_eff=-1     # CRITIQUE ici aussi
    )

    mi = mi_result[0]

    print(f"Info Mutuelle I(X;Y) : {mi}")
    
except TypeError as e:
    print(f"Erreur d'arguments : {e}")
    print("Essai de secours (arguments positionnels)...")
    # Si les noms changent, on tente l'ordre standard : x, y, k, N_eff
    try:
        mi = ee.compute_MI(x, y, 3, -1)
        print(f"Info Mutuelle (Positionnel) : {mi}")
    except Exception as e2:
        print(f"Echec total MI : {e2}")

except Exception as e:
    print(f"Erreur MI Générale : {e}")
print("-" * 30)