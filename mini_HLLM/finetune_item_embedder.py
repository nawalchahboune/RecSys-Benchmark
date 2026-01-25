from __future__ import annotations
import random
from pathlib import Path
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from item_encoder import ClueWebSeqDataset

from helpers import load_seen_items, load_sequences_from_dataset, load_item_text_simple

# On veut que l’embedding du contexte soit plus proche de l’embedding 
# de la vraie page suivante que de l’embedding de la page random.
def build_training_examples(
    sequences: List[List[int]],
    item_text: Dict[int, str],
    seen_items: List[int],
    n_context: int = 5,
    max_examples: int = 5000,
) -> List[InputExample]:
    """
    Triplets: (anchor=context_text, positive=next_item_text, negative=random_item_text)
    """
    examples = []
    seen_set = set(seen_items)

    for seq in sequences:
        # keep only items we have text for
        seq = [x for x in seq if x in seen_set and x in item_text]
        if len(seq) < 2:
            continue

        for t in range(1, len(seq)):
            context_ids = seq[max(0, t - n_context):t]
            pos_id = seq[t]

            # build anchor text: concatenate last n_context item texts
            anchor_text = " [SEP] ".join(item_text[i] for i in context_ids if i in item_text)
            pos_text = item_text[pos_id]

            # random negative
            neg_id = random.choice(seen_items)
            while neg_id == pos_id or neg_id not in item_text:
                neg_id = random.choice(seen_items)
            neg_text = item_text[neg_id]

            examples.append(InputExample(texts=[anchor_text, pos_text, neg_text]))
            if len(examples) >= max_examples:
                return examples

    return examples


def main():
    random.seed(0)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    data_dir = Path("../data/ClueWeb-Reco/ordered_id_splits")
    
    seen_items = load_seen_items(data_dir / "seen_item_ids.txt")
    print(f"Loaded {len(seen_items)} seen items")
    
    dataset = ClueWebSeqDataset(
        input_path=str(data_dir / "valid_input.tsv"),
        target_path=str(data_dir / "valid_target.tsv"),
        max_seq_len=50,
    )
    sequences_train = load_sequences_from_dataset(dataset)
    print(f"Loaded {len(sequences_train)} sequences using ClueWebSeqDataset")
    
    item_text = load_item_text_simple(seen_items)
    print(f"Loaded text for {len(item_text)} items")

    train_examples = build_training_examples(
        sequences=sequences_train,
        item_text=item_text,
        seen_items=seen_items,
        n_context=5,
        max_examples=20000,
    )

    train_loader = DataLoader(train_examples, batch_size=32, shuffle=True)

    train_loss = losses.TripletLoss(model=model)

    from sentence_transformers import evaluation
    eval_examples = train_examples[:min(500, len(train_examples))]
    evaluator = evaluation.TripletEvaluator.from_input_examples(eval_examples, name='eval')

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,  
        epochs=30,  
        evaluation_steps=500,  
        warmup_steps=100,
        show_progress_bar=True,
        output_path="./checkpoints/item_embedder",  
        save_best_model=True,  
        checkpoint_save_steps=500, 
        checkpoint_save_total_limit=2,  
    )
    
    print(f"\n✓ MEILLEUR modèle sauvegardé dans: ./checkpoints/item_embedder")
    print("✓ Derniers checkpoints dans: ./checkpoints/item_embedder/checkpoint-XXX")
    print("Vous pouvez arrêter (Ctrl+C) à tout moment!")

if __name__ == "__main__":
    main()
