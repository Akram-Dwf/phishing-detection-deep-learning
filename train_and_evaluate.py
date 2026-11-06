import os
import time
import json
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
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Create directories
os.makedirs('dataset', exist_ok=True)
os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("1. Memuat Dataset UCI Phishing Websites...")
df = pd.read_csv('dataset/phishing_websites_uci.csv')

# Ensure integer types
for col in df.columns:
    df[col] = df[col].astype(int)

# Map target: -1 (Phishing) -> 1 (Phishing / Berbahaya), 1 (Legitimate) -> 0 (Legitimate / Aman)
# Wait: in UCI #327 documentation, -1 is Phishing and 1 is Legitimate.
# Let's make 1 = Phishing (Positive Class for Detection) and 0 = Legitimate (Aman)
df['Label'] = df['Result'].apply(lambda x: 1 if x == -1 else 0)
df['Status'] = df['Label'].map({0: 'Legitimate (Aman)', 1: 'Phishing (Berbahaya)'})

print(f"Total baris: {len(df)}, Total fitur: {df.shape[1] - 3}")
print(df['Status'].value_counts())

# ==========================================
# 2. VISUALISASI EDA (EXPLORATORY DATA ANALYSIS)
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Class Distribution
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
counts = df['Status'].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor='black', alpha=0.88)
for bar in bars:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    ax.text(bar.get_x() + bar.get_width()/2., h + 120, f"{h:,} ({pct:.1f}%)",
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Distribusi Kelas Dataset UCI Phishing Websites (N = 11.055)', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Jumlah Sampel Situs Web', fontsize=11)
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig('images/01_class_distribution.png')
plt.close()

# Plot 2: Feature Correlation with Phishing Label
feature_cols = [c for c in df.columns if c not in ['Result', 'Label', 'Status']]
correlations = df[feature_cols].apply(lambda x: x.corr(df['Label'])).abs().sort_values(ascending=True)
top15_features = correlations.tail(15)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
axes[0].barh(top15_features.index, top15_features.values, color='#3498db', edgecolor='black', alpha=0.85)
axes[0].set_title('Top 15 Fitur Keamanan Siber Paling Berpengaruh (|r|)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Korelasi Absolut terhadap Indikasi Phishing', fontsize=10)
for i, v in enumerate(top15_features.values):
    axes[0].text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=9)
axes[0].set_xlim(0, max(top15_features.values) * 1.18)

top10_cols = list(correlations.tail(10).index)[::-1] + ['Label']
corr_matrix = df[top10_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1], cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
axes[1].set_title('Heatmap Korelasi 10 Fitur Utama & Target Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images/02_feature_correlation.png')
plt.close()

# ==========================================
# 3. PREPROCESSING & DATA SPLITTING
# ==========================================
X = df[feature_cols].values.astype(np.float32)
y = df['Label'].values.astype(np.float32)

# Split Train (70%), Validation (15%), Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape for 1D-CNN / RNN / GRU: (samples, timesteps=30, features=1)
n_features = X_train_scaled.shape[1]
X_train_seq = X_train_scaled.reshape(-1, n_features, 1)
X_val_seq = X_val_scaled.reshape(-1, n_features, 1)
X_test_seq = X_test_scaled.reshape(-1, n_features, 1)

print(f"Shape X_train_seq: {X_train_seq.shape}, X_val_seq: {X_val_seq.shape}, X_test_seq: {X_test_seq.shape}")

# ==========================================
# 4. DEFINISI ARSITEKTUR MODEL DEEP LEARNING
# ==========================================

def build_cnn_model(input_shape):
    """Model 1: 1D Convolutional Neural Network (1D-CNN)"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ], name='1D_CNN')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_rnn_model(input_shape):
    """Model 2: Simple Recurrent Neural Network (Simple RNN)"""
    # Reshape 30 features into 6 sequential feature groups of 5 features each for faster & more stable RNN sequence learning
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((6, 5)),
        layers.SimpleRNN(64, return_sequences=True, activation='tanh'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.SimpleRNN(32, return_sequences=False, activation='tanh'),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ], name='Simple_RNN')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_gru_model(input_shape):
    """Model 3: Gated Recurrent Unit (GRU)"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((6, 5)),
        layers.Bidirectional(layers.GRU(64, return_sequences=True)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.GRU(32, return_sequences=False),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ], name='GRU_Model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_transfer_learning_model(input_shape, X_pretrain):
    """
    Model 4: Transfer Learning (Pre-trained Deep Feature Extractor + Fine-Tuning)
    Step A: Pre-train a 1D-Convolutional Autoencoder / Feature Backbone on unlabeled feature structure
    Step B: Transfer the pre-trained encoder weights and fine-tune with a classification head
    """
    inp = layers.Input(shape=input_shape, name='tl_input')
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='pretrained_conv1')(inp)
    x = layers.BatchNormalization(name='pretrained_bn1')(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu', name='pretrained_conv2')(x)
    x = layers.BatchNormalization(name='pretrained_bn2')(x)
    encoded = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='pretrained_encoder')(x)

    # Decoder for self-supervised pre-training
    decoded = layers.Conv1D(1, kernel_size=3, padding='same', activation='linear', name='decoder_out')(encoded)
    autoencoder = models.Model(inputs=inp, outputs=decoded, name='Pretrained_Backbone_AE')
    autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    # Quick self-supervised pretraining on feature patterns
    autoencoder.fit(X_pretrain, X_pretrain, epochs=8, batch_size=64, verbose=0)

    # Create base feature extractor from pre-trained encoder
    base_encoder = models.Model(inputs=inp, outputs=encoded, name='Pretrained_Base_Encoder')
    base_encoder.trainable = True  # Fine-tuning enabled

    # Attach new classification head on top of pre-trained backbone
    tl_input = layers.Input(shape=input_shape)
    features = base_encoder(tl_input)
    pooled = layers.GlobalMaxPooling1D()(features)
    dense1 = layers.Dense(128, activation='relu')(pooled)
    drop1 = layers.Dropout(0.25)(dense1)
    dense2 = layers.Dense(64, activation='relu')(drop1)
    out = layers.Dense(1, activation='sigmoid')(dense2)

    tl_model = models.Model(inputs=tl_input, outputs=out, name='Transfer_Learning_CNN')
    tl_model.compile(optimizer=keras.optimizers.Adam(learning_rate=8e-4),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
    return tl_model

# ==========================================
# 5. PROSES TRAINING SEMUA MODEL
# ==========================================
EPOCHS = 25
BATCH_SIZE = 64

model_builders = {
    '1D-CNN': lambda: build_cnn_model((n_features, 1)),
    'Simple RNN': lambda: build_rnn_model((n_features, 1)),
    'GRU': lambda: build_gru_model((n_features, 1)),
    'Transfer Learning': lambda: build_transfer_learning_model((n_features, 1), X_train_seq)
}

histories = {}
results_list = []
predictions_dict = {}
probabilities_dict = {}

for name, builder in model_builders.items():
    print(f"\n--- Melatih Model: {name} ---")
    t0 = time.time()
    model = builder()
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy', patience=6, restore_best_weights=True, mode='max'
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=0
    )
    
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    train_time = time.time() - t0
    histories[name] = history.history
    
    # Save model
    safe_name = name.lower().replace(' ', '_').replace('-', '_')
    model.save(f"models/{safe_name}_model.keras")
    
    # Evaluate on test set
    y_prob = model.predict(X_test_seq, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
    probabilities_dict[name] = y_prob
    predictions_dict[name] = y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"[{name}] Selesai dalam {train_time:.2f}s | Test Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | ROC-AUC: {auc*100:.2f}%")
    
    results_list.append({
        'Model': name,
        'Accuracy (%)': round(acc * 100, 2),
        'Precision (%)': round(prec * 100, 2),
        'Recall (%)': round(rec * 100, 2),
        'F1-Score (%)': round(f1 * 100, 2),
        'ROC-AUC (%)': round(auc * 100, 2),
        'Training Time (s)': round(train_time, 2),
        'Total Parameters': model.count_params()
    })

results_df = pd.DataFrame(results_list)
results_df.to_csv('results/metrics_summary.csv', index=False)
with open('results/metrics_summary.json', 'w') as f:
    json.dump(results_list, f, indent=2)

print("\n=== RINGKASAN HASIL EVALUASI MODEL ===")
print(results_df.to_string(index=False))

# ==========================================
# 6. VISUALISASI HASIL TRAINING & EVALUASI
# ==========================================

# Plot 3: Training & Validation Accuracy and Loss Curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
palette = {'1D-CNN': '#2980b9', 'Simple RNN': '#e67e22', 'GRU': '#27ae60', 'Transfer Learning': '#8e44ad'}

for idx, (name, hist) in enumerate(histories.items()):
    ax = axes[idx // 2, idx % 2]
    epochs_range = range(1, len(hist['accuracy']) + 1)
    ax.plot(epochs_range, [a * 100 for a in hist['accuracy']], label='Train Accuracy', color=palette[name], linewidth=2)
    ax.plot(epochs_range, [a * 100 for a in hist['val_accuracy']], label='Val Accuracy', color=palette[name], linestyle='--', linewidth=2)
    ax.set_title(f'Kurva Pembelajaran Akurasi: {name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(82, 100)
    ax.legend(loc='lower right')

plt.suptitle('Perbandingan Kurva Pembelajaran (Training vs Validation Accuracy)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/03_training_history.png', bbox_inches='tight')
plt.close()

# Plot 4: Confusion Matrices for All Models
fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
cmaps = ['Blues', 'Oranges', 'Greens', 'Purples']

for idx, (name, y_pred) in enumerate(predictions_dict.items()):
    ax = axes[idx // 2, idx % 2]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[idx], ax=ax, cbar=False,
                xticklabels=['Aman (0)', 'Phishing (1)'],
                yticklabels=['Aman (0)', 'Phishing (1)'],
                annot_kws={'size': 13, 'weight': 'bold'})
    acc_val = accuracy_score(y_test, y_pred) * 100
    ax.set_title(f'Confusion Matrix: {name} (Acc: {acc_val:.2f}%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediksi Model')
    ax.set_ylabel('Label Aktual')

plt.suptitle('Perbandingan Confusion Matrix pada Data Uji (N = 1.659 Sampel)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/04_confusion_matrices.png', bbox_inches='tight')
plt.close()

# Plot 5: Bar Chart Model Comparison
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
metrics_to_plot = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'ROC-AUC (%)']
x = np.arange(len(results_df['Model']))
width = 0.15
metric_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

for i, metric in enumerate(metrics_to_plot):
    offset = (i - 2) * width
    bars = ax.bar(x + offset, results_df[metric], width, label=metric.replace(' (%)', ''), color=metric_colors[i], edgecolor='black', alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{h:.1f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_title('Perbandingan Performa Arsitektur Deep Learning pada Deteksi Situs Phishing', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], fontsize=11, fontweight='bold')
ax.set_ylabel('Skor Evaluasi (%)', fontsize=11)
ax.set_ylim(85, 102)
ax.legend(loc='lower right', ncol=5, frameon=True)
plt.tight_layout()
plt.savefig('images/05_model_comparison.png')
plt.close()

# Plot 6: ROC-AUC Curve Comparison
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
for name, y_prob in probabilities_dict.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", color=palette[name], linewidth=2.2)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Guess (AUC = 0.5000)')
ax.set_title('Perbandingan Kurva ROC (Receiver Operating Characteristic)', fontsize=13, fontweight='bold')
ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=11)
ax.legend(loc='lower right', fontsize=10, frameon=True)
plt.tight_layout()
plt.savefig('images/06_roc_curves.png')
plt.close()

print("\nSemua grafik berhasil disimpan di folder 'images/' dan model disimpan di folder 'models/'!")
