import os
import numpy as np
import torch
from recbole.quick_start.quick_start import load_data_and_model
from recbole.data.interaction import Interaction

# Détection automatique du périphérique
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Chemins des fichiers et paramètres
MODEL_PATH = "/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/checkpoints/SASRecF_ml-1m/SASRecF-Jan-18-2026_15-00-53.pth"
OUTPUT_FILE = "predictions.txt"
TOP_K = 10  # Nombre de recommandations à générer

# Chargement du modèle et des données
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(MODEL_PATH)
model = model.to(device)  # Déplacer le modèle sur le bon périphérique
print(f"Taille de item_embedding.weight: {model.item_embedding.weight.size()}")# Limiter à 5 exemples de test
test_examples = []
for i, batch in enumerate(test_data):
    if i >= 5:  # Limiter à 5 exemples
        break
    interaction = batch[0]  # Supposons que les interactions soient le premier élément
    test_examples.append(interaction)
    # Prédiction pour chaque exemple
predictions = []
for i, interaction in enumerate(test_examples):
    # Déplacer l'interaction sur le bon appareil (CPU/GPU)
    interaction = interaction.to(device)

    # Générer les scores pour tous les items
    scores = model.full_sort_predict(interaction)
    scores = scores.view(-1)

    # Obtenir les indices des TOP_K items
    top_k_items = torch.topk(scores, TOP_K).indices.cpu().numpy()
    predictions.append(top_k_items)

# Écriture des prédictions dans un fichier texte
# Mapping interne → réel
# Créer un mapping interne → réel pour les items
item_feat = dataset.item_feat  # Accéder aux données des items
internal_to_real = {i: item_feat['item_id'][i] for i in range(len(item_feat))}
# Vérifiez la taille de item_feat
print(f"Nombre d'items dans item_feat: {len(item_feat)}")

# Vérifiez les identifiants internes générés
for i, top_k in enumerate(predictions):
    print(f"Identifiants internes générés pour Exemple {i + 1}: {top_k}")
# Écriture des prédictions dans un fichier texte avec conversion
# Écriture des prédictions dans un fichier texte avec conversion
# Écriture des prédictions dans un fichier texte avec conversion
with open(OUTPUT_FILE, 'w') as f:
    for i, top_k in enumerate(predictions):
        # Filtrer les identifiants internes pour s'assurer qu'ils sont valides
        valid_top_k = [item for item in top_k if item < len(item_feat)]
        
        # Convertir les identifiants internes valides en identifiants réels
        real_ids = [internal_to_real[item] for item in valid_top_k]
        
        # Ajouter un message pour les identifiants invalides (facultatif)
        if len(valid_top_k) < len(top_k):
            real_ids.append(f"Invalid IDs: {len(top_k) - len(valid_top_k)}")

        f.write(f"Exemple {i + 1}: {', '.join(map(str, real_ids))}\n")


print(f"Prédictions écrites dans {OUTPUT_FILE}")


