# pyright: ignore[reportAttributeAccessIssue]
# pyright: ignore[reportMissingImports]
# pyright: reportGeneralTypeIssues=false

#run_test_train.py 
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import tensorflow as tf
#tf.compat.v1.disable_v2_behavior()
import numpy as np 
import keras
from keras.callbacks import EarlyStopping

from keras.callbacks import Callback



from facenet.src.align import detect_face
import cv2
from facenet.src import facenet

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from keras.optimizers import Adam, Adagrad



print("\n\n\n===== ENVIRONMENT CHECK =====")
try:
    print("TensorFlow version:", tf.__version__)
except AttributeError:
    print("TensorFlow version: unknown (TF1 legacy)")
print("Keras version:", keras.__version__)
print("NumPy version:", np.__version__)
print("=============================\n")

dataset_path = os.path.dirname(os.path.abspath(__file__))
#BASE_DIR = os.path.dirname(os.path.abspath(BASE_DIR))
dataset_path = os.path.join(os.path.dirname(dataset_path), "Nabor Danych Lic 2\\")



print("===== DATASET PATH CHECK =====")
print("Dataset path:", dataset_path)
print("Exists:", os.path.exists(dataset_path))
print("==============================\n")

# Список всех файлов
all_images = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]


# Извлекаем все метки: возраст, пол, раса


labels = []
valid_images = []  # создаём пустой список
for f in all_images:
    parts = f.split('_')
    if len(parts) < 4:  # проверяем, что файл имеет все 4 части
        print("Skipping some files:", f)
        continue

    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
    except ValueError:
        print("Error transforming file, slipping:", f)
        continue

    labels.append((age, gender, race))
    valid_images.append(f)  # добавляем только корректный файл

print("===== LABEL PARSING =====")
print("Valid images:", len(valid_images))
print("First 5 valid images:", valid_images[:5])
print("First 5 labels (age, gender, race):", labels[:5])
print("=========================\n")





# Для stratify: кортежи строкам преобразуем 
labels_str = [f"{age}_{gender}_{race}" for age, gender, race in labels]
# Считаем, сколько раз встречается каждая комбинация
counter = Counter(labels_str)


print("===== STRATIFY STATS =====")
print("Unique (age_gender_race) combinations:", len(counter))
print("Top 5 combinations:", counter.most_common(5))
print("==========================\n")


filtered_images = []
filtered_labels_str = []

for img, lbl in zip(valid_images, labels_str):
    if counter[lbl] >= 2:
        filtered_images.append(img)
        filtered_labels_str.append(lbl)

train_90, test_10 = train_test_split(
    filtered_images,
    test_size=0.1,
    random_state=42,
    stratify=filtered_labels_str
)

train_80, test_20 = train_test_split(
    filtered_images,
    test_size=0.2,
    random_state=42,
    stratify=filtered_labels_str
)
print("\n===== DATA SPLITS =====")
print(f"Train 80%: {len(train_80)} | Val 20%: {len(test_20)}")
print(f"Train 90%: {len(train_90)} | Val 10%: {len(test_10)}")
print("=======================\n")


#======================= ДЛЯ ГРАФИКА ПОСТРОЕНИЯ ==============


class ClassificationMetricsCallback(Callback):
    def __init__(self, validation_data, train_data=None):
        super().__init__()
        self.has_train = (train_data is not None)
        if self.has_train:
            self.X_train, self.y_train = train_data
        else:
            self.X_train = self.y_train = None

        self.X_val, self.y_val = validation_data
        self.X_train, self.y_train = train_data  # ← Добавили!
        self.precision_val = []
        self.recall_val = []
        self.f1_val = []
        self.auc_val = []
        self.precision_train = []  # ← Новые списки для трейна
        self.recall_train = []
        self.f1_train = []
        self.auc_train = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        # --- ВАЛИДАЦИЯ ---
        y_pred_proba_val = self.model.predict(self.X_val, verbose=0)
        y_pred_val = (y_pred_proba_val > 0.5).astype(int).flatten()

        prec_val = precision_score(self.y_val, y_pred_val, zero_division=0)
        rec_val = recall_score(self.y_val, y_pred_val, zero_division=0)
        f1_val = f1_score(self.y_val, y_pred_val, zero_division=0)
        auc_val = roc_auc_score(self.y_val, y_pred_proba_val)

        self.precision_val.append(prec_val)
        self.recall_val.append(rec_val)
        self.f1_val.append(f1_val)
        self.auc_val.append(auc_val)

        # --- ТРЕНИРОВКА (если есть данные) ---
        has_train = (self.X_train is not None and self.y_train is not None)
        if has_train:
            y_pred_proba_train = self.model.predict(self.X_train, verbose=0)
            y_pred_train = (y_pred_proba_train > 0.5).astype(int).flatten()


            prec_train = precision_score(self.y_train, y_pred_train, zero_division=0)
            rec_train = recall_score(self.y_train, y_pred_train, zero_division=0)
            f1_train = f1_score(self.y_train, y_pred_train, zero_division=0)
            auc_train = roc_auc_score(self.y_train, y_pred_proba_train)

            self.precision_train.append(prec_train)
            self.recall_train.append(rec_train)
            self.f1_train.append(f1_train)
            self.auc_train.append(auc_train)
        else:
            self.precision_train.append(0)
            self.recall_train.append(0)
            self.f1_train.append(0)
            self.auc_train.append(0)


        self.epochs.append(epoch + 1)

        # Печатаем обе метрики
        if has_train:
             print(f" - val_precision: {prec_val:.4f} | train_precision: {prec_train:.4f}")
             print(f" - val_recall: {rec_val:.4f} | train_recall: {rec_train:.4f}")
             print(f" - val_f1: {f1_val:.4f} | train_f1: {f1_train:.4f}")
             print(f" - val_auc: {auc_val:.4f} | train_auc: {auc_train:.4f}")
        else:
             print(f" - val_precision: {prec_val:.4f}")
             print(f" - val_recall: {rec_val:.4f}")
             print(f" - val_f1: {f1_val:.4f}")
             print(f" - val_auc: {auc_val:.4f}")

        # Сохраняем в logs
        if logs is not None:
            logs['val_precision'] = prec_val
            logs['val_recall'] = rec_val
            logs['val_f1'] = f1_val
            logs['val_auc'] = auc_val

    def plot_metrics(self, title_suffix=""):
        plt.figure(figsize=(12, 8))
        epochs = self.epochs
        has_train = len(self.precision_train) > 0 and self.precision_train[-1] != 0  # или лучше: проверить через атрибут

        # Precision
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.precision_val, label='Val Precision', color='blue')
        if len(self.precision_train) > 0:
            plt.plot(epochs, self.precision_train, label='Train Precision', color='lightblue', linestyle='--')
        plt.title(f'Precision {title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)

        # Recall
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.recall_val, label='Val Recall', color='orange')
        if len(self.recall_train) > 0:
            plt.plot(epochs, self.recall_train, label='Train Recall', color='peachpuff', linestyle='--')
        plt.title(f'Recall {title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)

        # F1-Score
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.f1_val, label='Val F1', color='green')
        if len(self.f1_train) > 0:
            plt.plot(epochs, self.f1_train, label='Train F1', color='lightgreen', linestyle='--')
        plt.title(f'F1-Score {title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('F1')
        plt.legend()
        plt.grid(True)

        # ROC-AUC
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.auc_val, label='Val AUC', color='red')
        if len(self.auc_train) > 0:
            plt.plot(epochs, self.auc_train, label='Train AUC', color='pink', linestyle='--')
        plt.title(f'ROC-AUC {title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'classification_metrics_{title_suffix.replace(" ", "_").lower()}.png')
        plt.show()

#===============МОДЕЛЯ FACENET'A ОПРЕДЕЛЕНИЕ====================

import sys
FACENET_SRC = r"D:\Documents\KULIAH SEMESTER 11\FaceNet\facenet\src"
sys.path.append(FACENET_SRC)


model_path = "D:\\Documents\\KULIAH SEMESTER 11\\FaceNet\\models\\20180408-102900.pb"


#==========ИЗ DAVIDSANDBERG'A БЕРУЩЕЙСЯ АРХИТЕКТУРЫ FACENET'A ПОСТРОЕНИЕ ===========================
# FACENET НАЗНАЧИТЬ (ЗНАКОМ РАВЕНСТВА) НЕ НАДОБИТСЯ, ПОСКОЛЬКУ FACENET, get_default_graph() уже осуществил , после "facenet.load_model()"а назначения 
with tf.Graph().as_default():
    sess = tf.Session()
    facenet.load_model(model_path)
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    images_ph = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train = tf.get_default_graph().get_tensor_by_name("phase_train:0")


print("===== FACENET LOAD =====")
print("FaceNet src path:", FACENET_SRC)
print("Model path:", model_path)
print("Model exists:", os.path.exists(model_path))
print("========================")


print("FaceNet model loaded successfully.")

print("Tensor names in graph:")
for t in tf.get_default_graph().get_operations()[:10]:
    print(" ", t.name)
print("...")


print("===== FACENET TENSORS =====")
print("Input tensor:", images_ph)
print("Embeddings tensor:", embeddings)
print("Phase_train tensor:", phase_train)
print("===========================\n")


# ЛИЦА КАДРИРОВАТЬ, ВЫРАВНИВАТЬ , ОТОЖДЕСТВЛЯТЬ ПОСРЕДСТВОМ MTCNN'A НАДОБИТСЯ 



def mtcnn_align_face(img, image_size=160, margin=44):
    """
    img: numpy array (H, W, 3)
    return: aligned face or None
    """
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    bounding_boxes, _ = detect_face.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor
    )

    if len(bounding_boxes) == 0:
        return None

    # берём самое большое лицо
    det = bounding_boxes[0][0:4].astype(int)

    h, w, _ = img.shape
    x1 = max(det[0] - margin // 2, 0)
    y1 = max(det[1] - margin // 2, 0)
    x2 = min(det[2] + margin // 2, w)
    y2 = min(det[3] + margin // 2, h)

    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (image_size, image_size))
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0  # FaceNet normalization

    return face


# ====================ВСТРАИВАЦИИ ПОЛУЧЕНИЕ====================


def get_embeddings(image_paths, batch_size=32):
    print(f"\nExtracting embeddings for {len(image_paths)} images...")
    embs = []

    for i in range(0, len(image_paths), batch_size):
        print(f" Batch {i} : {min(i+batch_size, len(image_paths))}")
        batch = load_images(image_paths[i:i+batch_size])

        feed = {
            images_ph: batch,
            phase_train: False
        }

        emb = sess.run(embeddings, feed_dict=feed)
        print("  \nEmbeddings batch shape:", emb.shape)
        embs.append(emb)

    embs = np.vstack(embs)
    print("\nFinal embeddings shape:", embs.shape)
    return embs

#===============ИЗОБРАЖЕНИЯ ЗАГРУЖЕНИЯ КОД========================


def load_images(file_list):
    imgs = []
    print(f"\nLoading {len(file_list)} images with MTCNN...")

    for f in file_list:
        path = os.path.join(dataset_path, f)
        if not os.path.exists(path):
            print("WARNING: missing file:", path)
            continue
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face = mtcnn_align_face(img)

        if face is None:
            print("No face detected, skipping:", f)
            continue

        imgs.append(face)

    imgs = np.array(imgs)
    print("Loaded aligned faces shape:", imgs.shape)
    return imgs


facenet_model = {
    "input": images_ph,
    "output": embeddings,
    "phase_train": phase_train,
    "session": sess
}

# ================== ГИПЕРПАРАМЕТРВО ПОДСТРОЕНИЕ==================
from keras.models import Sequential 
from keras.layers import Dense, Dropout, BatchNormalization

classifier = Sequential()
classifier.add(Dense(256, activation='relu', input_shape=(512,)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))
classifier.add(Dense(64, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(1, activation='sigmoid'))

optimizer = Adam(lr = 0.0001) # ПОДСТРАИВАТЬ ОБЯЗАНО 
classifier.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy'] 
)



#==============early stopping callbacks ОСУЩЕСТВЛЕНИЕ===================

class AccuracyThresholdCallback(Callback):
    def __init__(self, threshold=0.95):
        super().__init__()
        self.threshold = threshold
        self.metric_key = None

    def on_epoch_end(self, epoch, logs=None):
        if self.metric_key is None:
            # Определяем, какое имя использует Keras
            if 'val_accuracy' in logs:
                self.metric_key = 'val_accuracy'
            elif 'val_acc' in logs:
                self.metric_key = 'val_acc'
            else:
                print("⚠️ Не найдена метрика валидационной точности!")
                return

        val_acc = logs[self.metric_key]
        if val_acc >= self.threshold:
            print(f"\n Целевая точность {self.threshold} достигнута: {val_acc:.4f}. Остановка.\n")
            self.model.stop_training = True



early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)






znaczenie_epocha = 130

#===============определение обучения 80 - испытание 20 (ПОДСТРАИВАТЬ НУЖНО )=================

X_train_emb80 = get_embeddings(train_80)
X_val_emb20   = get_embeddings(test_20)

y_train80 = np.array([int(f.split('_')[1]) for f in train_80])
y_val20   = np.array([int(f.split('_')[1]) for f in test_20])


# ГРАФИКА ПОСТРОЕНИЕ 1
metrics_callback1 = ClassificationMetricsCallback(
    validation_data=(X_val_emb20, y_val20),
    train_data=(X_train_emb80, y_train80)
)


# CALLBACKS1
callbacks1 = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    AccuracyThresholdCallback(threshold=0.95),
    metrics_callback1
]

print("\n===== LABELS CHECK (80/20) =====\n")
print("y_train80 shape:", y_train80.shape)
print("Unique labels train:", np.unique(y_train80, return_counts=True))
print("y_val20 shape:", y_val20.shape)
print("Unique labels val:", np.unique(y_val20, return_counts=True))
print("===============================\n")



history1 = classifier.fit(
    X_train_emb80, y_train80,
    validation_data=(X_val_emb20, y_val20),
    epochs=znaczenie_epocha,
    batch_size=32,
    callbacks=callbacks1
)


last_val_acc = history1.history['val_acc'][-1] if 'val_acc' in history1.history else history1.history['val_accuracy'][-1]
print(f"\nПоследняя val_acc из истории (80 - 20): {last_val_acc:.4f}")


#-----Предсказание на валидации-----
y_pred_proba = classifier.predict(X_val_emb20)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Расчёт метрик
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_val20, y_pred, zero_division=0)
recall = recall_score(y_val20, y_pred, zero_division=0)
f1 = f1_score(y_val20, y_pred, zero_division=0)

# Вывод
print("\n===== FINAL METRICS (80/20) =====")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1-Score  = {f1:.4f}")
print("===============================\n")


#===============определение обучения 90 - испытание 10 (ПОДСТРАИВАТЬ НУЖНО )=================

X_train_emb90 = get_embeddings(train_90)
X_val_emb10   = get_embeddings(test_10)

y_train90 = np.array([int(f.split('_')[1]) for f in train_90])
y_val10   = np.array([int(f.split('_')[1]) for f in test_10])


# ГРАФИКА ПОСТРОЕНИЕ 2
metrics_callback2 = ClassificationMetricsCallback(
    validation_data=(X_val_emb10, y_val10),
    train_data=(X_train_emb90, y_train90) 
)




# CALLBACKS2
callbacks2 = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    AccuracyThresholdCallback(threshold=0.95),
    metrics_callback2
]

print("\n===== LABELS CHECK (90/10) =====\n")
print("y_train90 shape:", y_train90.shape)
print("y_val10 shape:", y_val10.shape)
print("===============================\n")

history2 = classifier.fit(
    X_train_emb90, y_train90,
    validation_data=(X_val_emb10, y_val10),
    epochs=znaczenie_epocha,
    batch_size=32,
    callbacks=callbacks2
)


last_val_acc = history2.history['val_acc'][-1] if 'val_acc' in history2.history else history2.history['val_accuracy'][-1]
print(f"\nПоследняя val_acc из истории (90 - 10): {last_val_acc:.4f}")


#------Предсказание на валидации---------
y_pred_proba = classifier.predict(X_val_emb10)  # ← classifier2, если вы создали новую модель!
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

precision = precision_score(y_val10, y_pred, zero_division=0)
recall = recall_score(y_val10, y_pred, zero_division=0)
f1 = f1_score(y_val10, y_pred, zero_division=0)

print("\n===== FINAL METRICS (90/10) =====")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1-Score  = {f1:.4f}")
print("===============================\n")

#==================ПОДЫТОЖЕНИЕ ================

print("\n===== CLASSIFIER SUMMARY =====\n")
classifier.summary()
print("==============================\n")


metrics_callback1.plot_metrics("80-20")
metrics_callback2.plot_metrics("90-10")

