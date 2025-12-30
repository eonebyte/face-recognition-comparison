import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve)
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# KONFIGURASI SESUAI BAB 3 & TABEL 3.2.1
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = 'dataset/UTKFace'
OUTPUT_DIR = 'output/v8'
MAX_SAMPLES = 50  # Testing: 1000, Final: 5000-10000
EPOCHS = 125
BATCH_SIZE = 64
EMBEDDING_DIM = 128

# Tabel 3.2.1 Configuration
SPLIT_RATIOS = {
    '90-10': (0.9, 0.1),
    '80-20': (0.8, 0.2)
}

LEARNING_RATES = {
    '0.0001': 1e-4,
    '0.001': 1e-3
}

OPTIMIZERS = ['Adam', 'AdaGrad']

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')
print(f"Device: {DEVICE}")


# ==========================================
# TRIPLET LOSS (BAB 3.1.1.1.6)
# ==========================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        neg_dist = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
        losses = torch.nn.functional.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


class TripletDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = labels
        self.label_to_indices = {}
        
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        anchor_label = self.labels[idx]
        pos_indices = self.label_to_indices[anchor_label]
        pos_idx = np.random.choice([i for i in pos_indices if i != idx] or pos_indices)
        
        neg_label = np.random.choice([l for l in self.label_to_indices.keys() if l != anchor_label])
        neg_idx = np.random.choice(self.label_to_indices[neg_label])
        
        return (self.embeddings[idx], 
                self.embeddings[pos_idx], 
                self.embeddings[neg_idx],
                anchor_label)


class EmbeddingRefiner(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super(EmbeddingRefiner, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return torch.nn.functional.normalize(self.fc(x), p=2, dim=1)


# ==========================================
# ENGINE UTAMA
# ==========================================
class TrainingCurvesEngine:
    
    def __init__(self):
        print(f"\n{'='*80}")
        print("INITIALIZING TRAINING CURVES ENGINE (TABLE 3.2.1)")
        print(f"{'='*80}\n")
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        print("Loading Pre-trained Models...")
        
        self.detector = MTCNN(keep_all=False, device=DEVICE, post_process=False)
        self.facenet = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(DEVICE)
        
        mnet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.arc2face_backbone = torch.nn.Sequential(*list(mnet.children())[:-1]).eval().to(DEVICE)
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.retinaface_backbone = torch.nn.Sequential(*list(resnet.children())[:-1]).eval().to(DEVICE)
        
        print("All models loaded!\n")
    
    def load_dataset(self):
        print(f"\n{'='*80}")
        print(f"STEP 1: LOADING DATASET")
        print(f"{'='*80}\n")
        
        files = [f for f in os.listdir(DATASET_PATH) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if MAX_SAMPLES:
            files = files[:MAX_SAMPLES]
        
        paths = [os.path.join(DATASET_PATH, f) for f in files]
        labels = []
        
        for f in files:
            try:
                parts = f.split('_')
                label = f"{parts[1]}_{parts[2]}"
                labels.append(label)
            except:
                labels.append("unknown")
        
        print(f"   Total files: {len(files)}")
        print(f"   Unique identities: {len(np.unique(labels))}\n")
        
        return paths, np.array(labels)
    
    def extract_embeddings(self, paths, labels):
        print(f"\n{'='*80}")
        print(f"STEP 2: EXTRACTING EMBEDDINGS")
        print(f"{'='*80}\n")
        
        facenet_emb, arc2face_emb, retinaface_emb, valid_labels = [], [], [], []
        
        for i, path in enumerate(tqdm(paths, desc="Extracting")):
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_tensor = self.detector(img_rgb)
                
                if face_tensor is None:
                    continue
                
                face_tensor = face_tensor.to(DEVICE)
                
                with torch.no_grad():
                    fn_emb = self.facenet(face_tensor.unsqueeze(0)).cpu().numpy().flatten()
                    fn_emb = fn_emb / (np.linalg.norm(fn_emb) + 1e-8)
                    
                    face_224 = torch.nn.functional.interpolate(
                        face_tensor.unsqueeze(0), size=(224, 224), mode='bilinear'
                    )
                    arc_emb = torch.flatten(self.arc2face_backbone(face_224)).cpu().numpy()
                    arc_emb = arc_emb / (np.linalg.norm(arc_emb) + 1e-8)
                    
                    ret_emb = torch.flatten(self.retinaface_backbone(face_224)).cpu().numpy()
                    ret_emb = ret_emb / (np.linalg.norm(ret_emb) + 1e-8)
                
                facenet_emb.append(fn_emb)
                arc2face_emb.append(arc_emb)
                retinaface_emb.append(ret_emb)
                valid_labels.append(labels[i])
                
            except Exception as e:
                continue
        
        print(f"\nExtracted: {len(valid_labels)} faces\n")
        
        return (np.array(facenet_emb), np.array(arc2face_emb), 
                np.array(retinaface_emb), np.array(valid_labels))
    
    def train_with_curves(self, X, y, method_name, split_name, lr_name, opt_name):
        """
        Training dengan recording metrics per epoch untuk plotting curves
        """
        config_name = f"{split_name}_{lr_name}_{opt_name}"
        print(f"\n>> {method_name} | Config: {config_name}")
        
        # Get configuration
        train_ratio, test_ratio = SPLIT_RATIOS[split_name]
        learning_rate = LEARNING_RATES[lr_name]
        
        # Encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        # Filter classes
        unique, counts = np.unique(y_enc, return_counts=True)
        valid_classes = unique[counts >= 4]
        mask = np.isin(y_enc, valid_classes)
        X_filt, y_filt = X[mask], y_enc[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_filt, y_filt, test_size=test_ratio, 
            stratify=y_filt, random_state=42
        )
        
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Initialize model
        refiner = EmbeddingRefiner(X.shape[1], EMBEDDING_DIM).to(DEVICE)
        triplet_loss = TripletLoss(margin=0.2)
        
        if opt_name == 'Adam':
            optimizer = optim.Adam(refiner.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adagrad(refiner.parameters(), lr=learning_rate)
        
        # Prepare dataset
        dataset = TripletDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Training history
        history = {
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        # Training loop
        print(f"  Training for {EPOCHS} epochs...")
        
        from sklearn.neighbors import KNeighborsClassifier
        
        for epoch in tqdm(range(EPOCHS), desc=f"  {config_name}"):
            # Training step
            refiner.train()
            for anchor, pos, neg, _ in dataloader:
                anchor, pos, neg = anchor.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
                
                optimizer.zero_grad()
                anchor_emb = refiner(anchor)
                pos_emb = refiner(pos)
                neg_emb = refiner(neg)
                
                loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
                loss.backward()
                optimizer.step()
            
            # Evaluation setiap epoch
            refiner.eval()
            with torch.no_grad():
                X_train_ref = refiner(torch.FloatTensor(X_train).to(DEVICE)).cpu().numpy()
                X_test_ref = refiner(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()
            
            # Train KNN dan evaluate
            knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
            knn.fit(X_train_ref, y_train)
            y_pred = knn.predict(X_test_ref)
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # AUC
            try:
                y_test_bin = (y_test == 0).astype(int)
                y_prob = knn.predict_proba(X_test_ref)[:, 0]
                auc = roc_auc_score(y_test_bin, y_prob)
            except:
                auc = 0.5
            
            # Store
            history['precision'].append(precision)
            history['recall'].append(recall)
            history['f1'].append(f1)
            history['auc'].append(auc)
        
        # Final metrics
        final_metrics = {
            'precision': history['precision'][-1],
            'recall': history['recall'][-1],
            'f1': history['f1'][-1],
            'auc': history['auc'][-1]
        }
        
        print(f"  Final: P={final_metrics['precision']:.3f}, "
              f"R={final_metrics['recall']:.3f}, F1={final_metrics['f1']:.3f}, "
              f"AUC={final_metrics['auc']:.3f}\n")
        
        return history, final_metrics
    
    def visualize_training_curves(self, history, method_name, split_name, lr_name, opt_name):
        """
        Visualisasi 4 kurva training (SEPERTI GAMBAR ANDA)
        """
        config_title = f"Method: {method_name} | Opt: {opt_name} | LR: {lr_name}"
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 9))
        fig.patch.set_facecolor('#E8E8E8')
        
        epochs = np.arange(len(history['precision']))
        
        # Plot configurations
        plots = [
            ('precision', f'Precision {split_name}', '#1f77b4', axs[0, 0]),
            ('recall', f'Recall {split_name}', '#ff7f0e', axs[0, 1]),
            ('f1', f'F1-Score {split_name}', '#2ca02c', axs[1, 0]),
            ('auc', f'ROC-AUC {split_name}', '#d62728', axs[1, 1])
        ]
        
        for metric, title, color, ax in plots:
            data = np.array(history[metric])
            
            # Plot kurva
            ax.plot(epochs, data, color=color, linewidth=2.5)
            
            # Styling
            ax.set_title(title, fontweight='bold', fontsize=11)
            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel('Score', fontsize=9)
            ax.grid(True, which='both', color='white', linestyle='-', linewidth=0.8)
            ax.set_facecolor('#F0F0F0')
            ax.set_axisbelow(True)
            
            # Y-axis limits
            y_min = max(0, data.min() - 0.05)
            y_max = min(1, data.max() + 0.05)
            ax.set_ylim([y_min, y_max])
            
            # X-axis
            ax.set_xlim([0, EPOCHS])
        
        plt.suptitle(config_title, fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save
        filename = f"{OUTPUT_DIR}/{method_name}_{split_name}_{lr_name}_{opt_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def run(self):
        """Main execution - Generate 24 grafik"""
        start_time = time.time()
        
        print(f"\n{'#'*80}")
        print(f"  TABLE 3.2.1 EXPERIMENT - TRAINING CURVES")
        print(f"  Total: 24 grafik (3 methods × 8 configs)")
        print(f"{'#'*80}\n")
        
        # Step 1: Load dataset
        paths, labels = self.load_dataset()
        
        # Step 2: Extract embeddings
        facenet_emb, arc2face_emb, retinaface_emb, valid_labels = \
            self.extract_embeddings(paths, labels)
        
        # Step 3: Run experiments
        methods = [
            (facenet_emb, "FaceNet"),
            (arc2face_emb, "Arc2Face"),
            (retinaface_emb, "RetinaFace")
        ]
        
        all_results = []
        generated_files = []
        
        for embeddings, method_name in methods:
            print(f"\n{'='*80}")
            print(f"PROCESSING: {method_name}")
            print(f"{'='*80}")
            
            for split_name in SPLIT_RATIOS.keys():
                for lr_name in LEARNING_RATES.keys():
                    for opt_name in OPTIMIZERS:
                        
                        # Train and get history
                        history, final_metrics = self.train_with_curves(
                            embeddings, valid_labels,
                            method_name, split_name, lr_name, opt_name
                        )
                        
                        # Visualize
                        filename = self.visualize_training_curves(
                            history, method_name, split_name, lr_name, opt_name
                        )
                        
                        generated_files.append(filename)
                        
                        # Store results
                        all_results.append({
                            'Method': method_name,
                            'Split': split_name,
                            'LR': lr_name,
                            'Optimizer': opt_name,
                            'Final_Precision': final_metrics['precision'],
                            'Final_Recall': final_metrics['recall'],
                            'Final_F1': final_metrics['f1'],
                            'Final_AUC': final_metrics['auc']
                        })
        
        # Save summary
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f"{OUTPUT_DIR}/all_results_summary.csv", index=False)
        
        # Print summary
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETED!")
        print(f"{'='*80}")
        print(f"   Time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hours)")
        print(f"   Generated: {len(generated_files)} grafik")
        print(f"   Output: {OUTPUT_DIR}/")
        print(f"\n   Files generated:")
        for i, f in enumerate(generated_files, 1):
            print(f"   [{i:2d}] {os.path.basename(f)}")
        print(f"\n   Summary: all_results_summary.csv")
        print(f"{'='*80}\n")
        
        # Print final results table
        print("\nFINAL RESULTS SUMMARY:")
        print(df_results.to_string(index=False))


if __name__ == "__main__":
    engine = TrainingCurvesEngine()
    engine.run()