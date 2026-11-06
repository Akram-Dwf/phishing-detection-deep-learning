import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Seed agar hasil konsisten
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 1. Load Dataset
print("Memuat dataset UCI Phishing Websites...")
df = pd.read_csv('dataset/phishing_websites_uci.csv')

# Mengubah label target: -1 (Phishing) -> 1, dan 1 (Legitimate/Aman) -> 0
df['Label'] = df['Result'].apply(lambda x: 1 if x == -1 else 0)

print(f"Jumlah data: {len(df)} baris")
print("Distribusi Label (0 = Aman, 1 = Phishing):")
print(df['Label'].value_counts())

# Visualisasi 1: Distribusi Kelas
plt.figure(figsize=(6, 4), dpi=120)
kelas_counts = df['Label'].value_counts().sort_index()
plt.bar(['Legitimate / Aman (0)', 'Phishing (1)'], kelas_counts.values, color=['#3498db', '#e74c3c'], width=0.5)
plt.title('Distribusi Kelas Dataset Phishing Websites')
plt.ylabel('Jumlah Data')
for i, v in enumerate(kelas_counts.values):
    plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
plt.ylim(0, 7000)
plt.tight_layout()
plt.savefig('images/distribusi_kelas.png')
plt.close()

# 2. Preprocessing Data
feature_cols = [c for c in df.columns if c not in ['Result', 'Label']]
X = df[feature_cols].values.astype(np.float32)
y = df['Label'].values.astype(np.float32)

# Split data: 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape menjadi (samples, 30, 1) untuk input CNN dan RNN
X_train_seq = X_train.reshape(-1, 30, 1)
X_test_seq = X_test.reshape(-1, 30, 1)

print(f"Data Train: {X_train_seq.shape}, Data Test: {X_test_seq.shape}")

EPOCHS = 15
BATCH_SIZE = 64

# ==========================================
# MODEL 1: CNN (Convolutional Neural Network)
# ==========================================
print("\n--- 1. Melatih Model CNN ---")
cnn_model = models.Sequential([
    layers.Input(shape=(30, 1)),
    layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv_base_1'),
    layers.MaxPooling1D(pool_size=2, name='pool_base_1'),
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv_base_2'),
    layers.Flatten(name='flatten_base'),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
], name='CNN')

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_cnn = cnn_model.fit(
    X_train_seq, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)
cnn_model.save('models/cnn_model.keras')

# ==========================================
# MODEL 2: RNN (Recurrent Neural Network)
# ==========================================
print("\n--- 2. Melatih Model RNN ---")
rnn_model = models.Sequential([
    layers.Input(shape=(30, 1)),
    layers.SimpleRNN(32, activation='tanh'),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
], name='RNN')

rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_rnn = rnn_model.fit(
    X_train_seq, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)
rnn_model.save('models/rnn_model.keras')

# ==========================================
# MODEL 3: Transfer Learning
# (Menggunakan Pre-trained Feature Extractor dari CNN + Classifier Baru)
# ==========================================
print("\n--- 3. Melatih Model Transfer Learning ---")
# Mengambil layer ekstraksi fitur dari model CNN yang sudah dilatih
base_feature_extractor = models.Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.get_layer('flatten_base').output,
    name='pretrained_cnn_base'
)
# Membekukan bobot base model (freeze layers)
base_feature_extractor.trainable = False

# Menambahkan layer klasifikasi baru (Transfer Learning Head)
tl_model = models.Sequential([
    layers.Input(shape=(30, 1)),
    base_feature_extractor,
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
], name='Transfer_Learning')

tl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_tl = tl_model.fit(
    X_train_seq, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)
tl_model.save('models/transfer_learning_model.keras')

# ==========================================
# 4. EVALUASI KETIGA MODEL
# ==========================================
models_dict = {
    'CNN': cnn_model,
    'RNN': rnn_model,
    'Transfer Learning': tl_model
}

histories = {
    'CNN': history_cnn.history,
    'RNN': history_rnn.history,
    'Transfer Learning': history_tl.history
}

results = []
preds_dict = {}

print("\n=== HASIL EVALUASI PADA DATA TEST ===")
for name, model in models_dict.items():
    y_prob = model.predict(X_test_seq, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    preds_dict[name] = y_pred

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100

    results.append({
        'Model': name,
        'Accuracy (%)': round(acc, 2),
        'Precision (%)': round(prec, 2),
        'Recall (%)': round(rec, 2),
        'F1-Score (%)': round(f1, 2)
    })
    print(f"\nModel: {name}")
    print(classification_report(y_test, y_pred, target_names=['Aman (0)', 'Phishing (1)']))

results_df = pd.DataFrame(results)
results_df.to_csv('results/metrics_summary.csv', index=False)
print("\nRingkasan Perbandingan Model:")
print(results_df.to_string(index=False))

# Visualisasi 2: Grafik Akurasi & Loss
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
for name, hist in histories.items():
    axes[0].plot(range(1, EPOCHS + 1), hist['val_accuracy'], marker='o', markersize=4, label=f"{name}")
    axes[1].plot(range(1, EPOCHS + 1), hist['val_loss'], marker='o', markersize=4, label=f"{name}")

axes[0].set_title('Perbandingan Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Perbandingan Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/grafik_akurasi_loss.png')
plt.close()

# Visualisasi 3: Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=120)
for idx, (name, y_pred) in enumerate(preds_dict.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                xticklabels=['Aman', 'Phishing'], yticklabels=['Aman', 'Phishing'])
    axes[idx].set_title(f'Confusion Matrix - {name}')
    axes[idx].set_xlabel('Prediksi')
    axes[idx].set_ylabel('Aktual')

plt.tight_layout()
plt.savefig('images/confusion_matrix.png')
plt.close()

# Visualisasi 4: Bar Chart Perbandingan Akurasi & F1-Score
plt.figure(figsize=(8, 4.5), dpi=120)
x = np.arange(len(results_df['Model']))
width = 0.35
b1 = plt.bar(x - width/2, results_df['Accuracy (%)'], width, label='Accuracy (%)', color='#2980b9')
b2 = plt.bar(x + width/2, results_df['F1-Score (%)'], width, label='F1-Score (%)', color='#27ae60')

for bar in b1:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{bar.get_height():.2f}%", ha='center', fontsize=9)
for bar in b2:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{bar.get_height():.2f}%", ha='center', fontsize=9)

plt.xticks(x, results_df['Model'])
plt.ylabel('Persentase (%)')
plt.title('Perbandingan Performa Model (CNN vs RNN vs Transfer Learning)')
plt.ylim(80, 100)
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('images/perbandingan_akurasi.png')
plt.close()

print("\nSelesai! Semua model dan grafik telah diperbarui.")
