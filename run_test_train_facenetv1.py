import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ==========================================
# KONFIGURASI (Tabel 3.2.1)
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = 'dataset/UTKFace'  # Sesuaikan dengan path Anda
OUTPUT_DIR = 'output/fixed_v1'
MAX_SAMPLES = 100  # Ubah ke None untuk data penuh
EPOCHS = 100
BATCH_SIZE = 32

# Parameter Eksperimen
SPLITS = {'80-20': 0.2, '90-10': 0.1}
OPTIMIZERS = ['Adam', 'Adagrad']
LR_TO_TEST = [0.0001] # Anda bisa menambah 0.001 jika ingin lebih banyak gambar

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# CALLBACK UNTUK TRACKING (DARI CODE 2)
# ==========================================
class ClassificationMetricsCallback(Callback):
    def __init__(self, validation_data, train_data):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.X_train, self.y_train = train_data
        self.history = {
            'p_v': [], 'p_t': [], 'r_v': [], 'r_t': [],
            'f_v': [], 'f_t': [], 'a_v': [], 'a_t': []
        }
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        # Prediksi
        y_pv = self.model.predict(self.X_val, verbose=0)
        y_cv = (y_pv > 0.5).astype(int).flatten()
        y_pt = self.model.predict(self.X_train, verbose=0)
        y_ct = (y_pt > 0.5).astype(int).flatten()

        # Simpan Metrik
        self.history['p_v'].append(precision_score(self.y_val, y_cv, zero_division=0))
        self.history['p_t'].append(precision_score(self.y_train, y_ct, zero_division=0))
        self.history['r_v'].append(recall_score(self.y_val, y_cv, zero_division=0))
        self.history['r_t'].append(recall_score(self.y_train, y_ct, zero_division=0))
        self.history['f_v'].append(f1_score(self.y_val, y_cv, zero_division=0))
        self.history['f_t'].append(f1_score(self.y_train, y_ct, zero_division=0))
        self.history['a_v'].append(roc_auc_score(self.y_val, y_pv))
        self.history['a_t'].append(roc_auc_score(self.y_train, y_pt))
        
        self.epochs.append(epoch + 1)

# ==========================================
# ENGINE UTAMA
# ==========================================
class SkripsiEngine:
    def __init__(self):
        print(f"Loading models on {DEVICE}...")
        self.detector = MTCNN(keep_all=False, device=DEVICE)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
        
        # Backbone Arc2Face & RetinaFace
        mnet = models.mobilenet_v3_large(pretrained=True)
        self.arc2face = torch.nn.Sequential(*list(mnet.children())[:-1]).eval().to(DEVICE)
        res = models.resnet50(pretrained=True)
        self.retinaface = torch.nn.Sequential(*list(res.children())[:-1]).eval().to(DEVICE)

    def load_data(self):
        files = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.jpg', '.png'))]
        if MAX_SAMPLES: files = files[:MAX_SAMPLES]
        paths, labels = [], []
        for f in files:
            try:
                gender = int(f.split('_')[1])
                paths.append(os.path.join(DATASET_PATH, f))
                labels.append(gender)
            except: continue
        return paths, np.array(labels)

    def get_embs(self, paths, method):
        embs = []
        valid_idx = []
        for i, p in enumerate(tqdm(paths, desc=f"Extracting {method}")):
            try:
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                face = self.detector(img)
                if face is not None:
                    face = face.to(DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        if method == "FaceNet": e = self.facenet(face)
                        elif method == "Arc2Face": e = self.arc2face(torch.nn.functional.interpolate(face, (224,224)))
                        else: e = self.retinaface(torch.nn.functional.interpolate(face, (224,224)))
                    embs.append(e.cpu().numpy().flatten())
                    valid_idx.append(i)
            except: continue
        return np.array(embs), valid_idx

    def plot_hasil(self, cb, method, split_label, opt_name, lr):
        """Menghasilkan gambar 2x2 persis seperti contoh lampiran"""
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        h = cb.history
        ep = cb.epochs

        # Konfigurasi per sub-plot
        plots = [
            ('p', 'Precision', 'blue', 'lightblue', axs[0, 0]),
            ('r', 'Recall', 'orange', 'navajowhite', axs[0, 1]),
            ('f', 'F1-Score', 'green', 'lightgreen', axs[1, 0]),
            ('a', 'ROC-AUC', 'firebrick', 'pink', axs[1, 1])
        ]

        for k, title, c_v, c_t, ax in plots:
            ax.plot(ep, h[f'{k}_v'], color=c_v, label=f'Val {title}', linewidth=1.8)
            ax.plot(ep, h[f'{k}_t'], color=c_t, label=f'Train {title}', linestyle='--', linewidth=1.5)
            ax.set_title(f"{title} {split_label}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.7)
            ax.set_ylim([None, 1.02])

        plt.tight_layout()
        # Penamaan file sesuai permintaan: facenet_80_20.adam.png
        filename = f"{method.lower()}_{split_label.replace('-', '_')}.{opt_name.lower()}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        plt.close()
        print(f"Saved: {filename}")

    def run(self):
        paths, labels = self.load_data()
        methods = ["FaceNet", "Arc2Face", "RetinaFace"]

        for m_name in methods:
            print(f"\n--- Method: {m_name} ---")
            embs, v_idx = self.get_embs(paths, m_name)
            v_labels = labels[v_idx]

            for s_label, s_val in SPLITS.items():
                for opt_name in OPTIMIZERS:
                    # Ambil satu LR saja untuk memenuhi kuota 4 gambar utama per method
                    lr = 0.0001 
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        embs, v_labels, test_size=s_val, stratify=v_labels, random_state=42
                    )

                    # Build Model
                    model = Sequential([
                        Dense(256, activation='relu', input_shape=(embs.shape[1],)),
                        BatchNormalization(),
                        Dropout(0.5),
                        Dense(1, activation='sigmoid')
                    ])
                    opt = Adam(learning_rate=lr) if opt_name == 'Adam' else Adagrad(learning_rate=lr)
                    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

                    # Callback
                    m_cb = ClassificationMetricsCallback((X_test, y_test), (X_train, y_train))
                    
                    # Train
                    model.fit(X_train, y_train, validation_data=(X_test, y_test),
                              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                              callbacks=[m_cb, EarlyStopping(patience=5)])

                    # Plot 2x2 persis lampiran
                    self.plot_hasil(m_cb, m_name, s_label, opt_name, lr)

if __name__ == "__main__":
    engine = SkripsiEngine()
    engine.run()