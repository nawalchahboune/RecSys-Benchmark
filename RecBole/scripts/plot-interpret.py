#!/usr/bin/env python3
"""
SASRecF Training Log Parser & Real-time Tracker
Analyse les logs RecBole et affiche des visualisations en temps réel
"""
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

class SASRecFLogParser:
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.metrics = defaultdict(list)
        self.debug_stats = defaultdict(lambda: defaultdict(list))
        self.epochs_data = []
        
    def parse_log_file(self, log_file):
        """Parse un fichier de logs complet"""
        with open(log_file, 'r') as f:
            content = f.read()
        return self.parse_log_content(content)
    
    def parse_log_content(self, content):
        """Parse le contenu des logs (string)"""
        lines = content.split('\n')
        
        current_epoch = None
        
        for line in lines:
            # === EPOCH INFO ===
            # Détecte: "epoch 0 training [time: 169.81s, train loss: 3567.8742]"
            epoch_train = re.search(r'epoch\s+(\d+)\s+training.*train loss:\s+([\d.]+)', line)
            if epoch_train:
                epoch_num = int(epoch_train.group(1))
                train_loss = float(epoch_train.group(2))
                current_epoch = epoch_num
                
                # Stocker
                self.metrics['epoch'].append(epoch_num)
                self.metrics['train_loss'].append(train_loss)
            
            # Détecte: "epoch 0 evaluating [time: 1.25s, valid_score: 0.024100]"
            epoch_eval = re.search(r'epoch\s+(\d+)\s+evaluating.*valid_score:\s+([\d.]+)', line)
            if epoch_eval:
                valid_score = float(epoch_eval.group(2))
                self.metrics['valid_score'].append(valid_score)
            
            # === VALIDATION METRICS ===
            # Détecte: "recall@10 : 0.0503"
            if 'valid result:' in line:
                # La ligne suivante contient toutes les métriques
                continue
            
            metric_match = re.findall(r'(recall|ndcg|mrr|hit|precision)@(\d+)\s*:\s*([\d.]+)', line)
            for metric_name, k, value in metric_match:
                key = f'{metric_name}@{k}'
                val = float(value)
                
                # Associer au dernier epoch connu
                if current_epoch is not None:
                    if key not in self.metrics:
                        self.metrics[key] = []
                    
                    # Pad avec None si nécessaire
                    while len(self.metrics[key]) < current_epoch:
                        self.metrics[key].append(None)
                    
                    self.metrics[key].append(val)
            
            # === DEBUG STATS (embeddings, gates, etc.) ===
            # Détecte: "[SASRecF] item_emb: {'shape': (2048, 50, 256), 'mean': -0.0002, 'std': 0.0147, ...}"
            debug_match = re.search(r'\[SASRecF\]\s+(\w+):\s+\{(.+)\}', line)
            if debug_match:
                tensor_name = debug_match.group(1)
                stats_str = debug_match.group(2)
                
                # Parse les stats
                stats = {}
                for stat in ['mean', 'std', 'min', 'max']:
                    stat_match = re.search(rf"'{stat}':\s*([\d.e-]+)", stats_str)
                    if stat_match:
                        try:
                            stats[stat] = float(stat_match.group(1))
                        except:
                            stats[stat] = 0.0
                
                # Stocker
                for stat, val in stats.items():
                    self.debug_stats[tensor_name][stat].append(val)
        
        return self.metrics, self.debug_stats
    
    def print_summary(self):
        """Affiche un résumé des métriques"""
        print("\n" + "="*70)
        print(" RÉSUMÉ DE L'ENTRAÎNEMENT")
        print("="*70)
        
        if 'epoch' in self.metrics and len(self.metrics['epoch']) > 0:
            n_epochs = max(self.metrics['epoch']) + 1
            print(f"\n Epochs complétés: {n_epochs}")
            
            # Train Loss
            if 'train_loss' in self.metrics:
                losses = self.metrics['train_loss']
                print(f"\n Train Loss:")
                print(f"   Epoch 0:     {losses[0]:.4f}")
                if len(losses) > 1:
                    print(f"   Epoch {len(losses)-1}:     {losses[-1]:.4f}")
                    print(f"   Amélioration: {losses[0] - losses[-1]:.4f} ({100*(losses[0]-losses[-1])/losses[0]:.1f}%)")
            
            # Validation Metrics
            print(f"\n Métriques de Validation (dernier epoch):")
            key_metrics = ['recall@10', 'ndcg@10', 'mrr@10', 'hit@10']
            for metric in key_metrics:
                if metric in self.metrics and len(self.metrics[metric]) > 0:
                    val = self.metrics[metric][-1]
                    print(f"   {metric:12s}: {val:.4f}")
        
        # Debug Stats Summary
        if self.debug_stats:
            print(f"\n🔍 Debug Stats (dernières valeurs):")
            for tensor in ['item_emb', 'feat_emb', 'gid', 'gfeat', 'gretr']:
                if tensor in self.debug_stats:
                    stats = self.debug_stats[tensor]
                    if 'mean' in stats and len(stats['mean']) > 0:
                        mean = stats['mean'][-1]
                        std = stats['std'][-1] if 'std' in stats else 0
                        
                        # Diagnostic
                        status = "success"
                        warning = ""
                        
                        if 'emb' in tensor:  # embeddings
                            if std < 0.02:
                                status = "error"
                                warning = "TROP BAS !"
                            elif std < 0.1:
                                status = "attention"
                                warning = "faible"
                        
                        elif tensor.startswith('g'):  # gates
                            if std < 0.01:
                                status = "error"
                                warning = "pas de variabilité"
                            elif std < 0.05:
                                status = "attention"
                                warning = "faible sélectivité"
                        
                        print(f"   {tensor:10s}: mean={mean:7.4f}, std={std:7.4f} {status} {warning}")
        
        print("="*70 + "\n")
    
    def plot_training_curves(self, save_path='training_analysis.png'):
        """Génère des visualisations complètes"""
        
        n_plots = 3  # Loss, Metrics, Debug
        fig, axes = plt.subplots(1, n_plots, figsize=(18, 5))
        
        # === PLOT 1: LOSS ===
        if 'train_loss' in self.metrics:
            epochs = range(len(self.metrics['train_loss']))
            axes[0].plot(epochs, self.metrics['train_loss'], 'b-o', label='Train Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title(' Training Loss', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Ajouter ligne de référence
            if len(self.metrics['train_loss']) > 0:
                initial_loss = self.metrics['train_loss'][0]
                axes[0].axhline(y=initial_loss/2, color='r', linestyle='--', 
                               label=f'Target (50% init)', alpha=0.5)
        
        # === PLOT 2: VALIDATION METRICS ===
        key_metrics = ['recall@10', 'ndcg@10', 'mrr@10']
        colors = ['green', 'orange', 'purple']
        
        for metric, color in zip(key_metrics, colors):
            if metric in self.metrics and len(self.metrics[metric]) > 0:
                # Filtrer les None
                epochs = [i for i, v in enumerate(self.metrics[metric]) if v is not None]
                values = [v for v in self.metrics[metric] if v is not None]
                
                axes[1].plot(epochs, values, marker='o', label=metric, 
                           color=color, linewidth=2)
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title(' Validation Metrics', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_ylim(bottom=0)
        
        # === PLOT 3: DEBUG STATS (Gates) ===
        gate_names = ['gid', 'gfeat', 'gretr']
        gate_colors = ['blue', 'green', 'red']
        
        for gate, color in zip(gate_names, gate_colors):
            if gate in self.debug_stats and 'mean' in self.debug_stats[gate]:
                means = self.debug_stats[gate]['mean']
                iterations = range(len(means))
                axes[2].plot(iterations, means, label=f'{gate} mean', 
                           color=color, alpha=0.7)
        
        axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral (0.5)')
        axes[2].set_xlabel('Iteration (logs)', fontsize=12)
        axes[2].set_ylabel('Gate Value', fontsize=12)
        axes[2].set_title(' Gates Evolution', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_ylim([0.3, 0.7])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n Visualisations sauvegardées: {save_path}")
        plt.show()
    
    def plot_embeddings_health(self, save_path='embeddings_health.png'):
        """Visualise la santé des embeddings"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # === STD Evolution ===
        for emb_name in ['item_emb', 'feat_emb', 'fused', 'seq_output']:
            if emb_name in self.debug_stats and 'std' in self.debug_stats[emb_name]:
                stds = self.debug_stats[emb_name]['std']
                iterations = range(len(stds))
                axes[0].plot(iterations, stds, label=emb_name, linewidth=2)
        
        # Zone saine pour embeddings
        axes[0].axhspan(0.3, 0.6, alpha=0.2, color='green', label='Healthy Range')
        axes[0].axhspan(0.0, 0.1, alpha=0.2, color='red', label='Critical')
        
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('Std Dev', fontsize=12)
        axes[0].set_title(' Embeddings Standard Deviation', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)
        
        # === Loss Evolution (per batch) ===
        if 'loss' in self.debug_stats and 'mean' in self.debug_stats['loss']:
            losses = self.debug_stats['loss']['mean']
            iterations = range(len(losses))
            axes[1].plot(iterations, losses, 'r-', linewidth=2, label='Batch Loss')
            
            # Smooth curve (moving average)
            if len(losses) > 10:
                window = 10
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                axes[1].plot(range(window-1, len(losses)), smoothed, 'b-', 
                           linewidth=2, label='Smoothed', alpha=0.7)
        
        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title(' Batch Loss Evolution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Health check sauvegardé: {save_path}")
        plt.show()
    
    def export_to_csv(self, output_file='training_metrics.csv'):
        """Exporte les métriques en CSV"""
        import csv
        
        # Trouver toutes les clés
        all_keys = set(self.metrics.keys())
        
        # Déterminer la longueur max
        max_len = max(len(v) for v in self.metrics.values()) if self.metrics else 0
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            
            for i in range(max_len):
                row = {}
                for key in all_keys:
                    if i < len(self.metrics[key]):
                        row[key] = self.metrics[key][i]
                    else:
                        row[key] = None
                writer.writerow(row)
        
        print(f" Métriques exportées: {output_file}")


if __name__ == "__main__":
    import sys
    
    # Option 1: Parser un fichier de logs
    parser = SASRecFLogParser()
    
    # Détecte automatiquement les fichiers .err ou .log
    import glob
    
    # Chercher les fichiers de logs possibles
    log_patterns = [
        "*.err",           # Fichiers stderr du cluster
        "*.log",           # Fichiers log classiques
        "training*.log",   # Logs d'entraînement
        "sasrecf*.err",    # Tes fichiers spécifiques
    ]
    
    log_files = []
    for pattern in log_patterns:
        log_files.extend(glob.glob(pattern))
    
    LOG_FILE = "outputs_sasrecf_on_ml-1m/sasrecf_ml1m_train-27753.err"
    if LOG_FILE and os.path.exists(LOG_FILE):
        try:
            print(f"\n Parsing {LOG_FILE}...")
            parser.parse_log_file(LOG_FILE)
            parser.print_summary()
            
            # Générer les plots
            base_name = LOG_FILE.rsplit('.', 1)[0]
            parser.plot_training_curves(f'{base_name}_training.png')
            parser.plot_embeddings_health(f'{base_name}_health.png')
            parser.export_to_csv(f'{base_name}_metrics.csv')
            
            print("\n Analyse terminée !")
            
        except Exception as e:
            print(f" Erreur lors du parsing: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f" Aucun fichier de logs trouvé.")
        print("\n Utilisation:")
        print("  python log_parser.py                          # Auto-détecte les fichiers")
        print("  python log_parser.py sasrecf_ml1m_train-28019.err  # Spécifie un fichier")
        print("\n Formats supportés: .err, .log")
        
        # Option 2: Parser du contenu directement (exemple)
        print("\n" + "="*70)
        print("EXEMPLE avec des logs de démonstration:")
        print("="*70)
        
        sample_logs = """
epoch 0 training [time: 169.81s, train loss: 3567.8742]
epoch 0 evaluating [time: 1.25s, valid_score: 0.024100]
valid result: 
recall@1 : 0.0061    recall@10 : 0.0503    recall@20 : 0.0912    ndcg@10 : 0.0241
[SASRecF] item_emb: {'mean': -0.0002, 'std': 0.0147}
[SASRecF] feat_emb: {'mean': -0.0001, 'std': 0.0072}
[SASRecF] gid: {'mean': 0.5013, 'std': 0.0034}
[SASRecF] gfeat: {'mean': 0.5016, 'std': 0.0014}
[SASRecF] loss: {'mean': 6.948}
        """
        
        parser.parse_log_content(sample_logs)
        parser.print_summary()