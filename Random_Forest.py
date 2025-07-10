# ============================================================
# Intrusion Detection using Random Forest (RF)
# Dataset: NSL-KDD (KDDTrain.txt, KDDTest.txt)
# Purpose: Preprocess data, train RF, evaluate model performance
# ============================================================

# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, auc)
from scipy.stats import randint
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
start_time = time.time()

# ===============================
# 2. ATTACK LABEL GROUPING FUNCTION
# ===============================
def map_attack(label):
    """Group fine-grained attacks into 5 categories."""
    dos = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
    probe = ['ipsweep', 'nmap', 'portsweep', 'satan']
    r2l = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster']
    u2r = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'sqlattack', 'xterm']

    if label == 'normal':
        return 'normal'
    elif label in dos:
        return 'dos'
    elif label in probe:
        return 'probe'
    elif label in r2l:
        return 'r2l'
    elif label in u2r:
        return 'u2r'
    else:
        return 'other'  # Unused categories

# ===============================
# 3. LOAD KDD DATASET
# ===============================
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

base_dir = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(base_dir, 'KDDTrain.txt'), names=columns)
test = pd.read_csv(os.path.join(base_dir, 'KDDTest.txt'), names=columns)
train.drop(columns=['difficulty'], inplace=True)
test.drop(columns=['difficulty'], inplace=True)

train['label'] = train['label'].apply(map_attack)
test['label'] = test['label'].apply(map_attack)
train = train[train['label'] != 'other']
test = test[test['label'] != 'other']

# ===============================
# 4. ENCODING CATEGORICAL FEATURES
# ===============================
cat_features = ['protocol_type', 'service', 'flag']
encoders = {}

for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    test[col] = test[col].map(mapping)
    test[col] = test[col].fillna(-1).astype(int)
    encoders[col] = le

# ===============================
# 5. SPLIT FEATURES AND TARGET
# ===============================
X_train = train.drop('label', axis=1)
X_test = test.drop('label', axis=1)
y_train = train['label']
y_test = test['label']

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# ===============================
# 6. STANDARDIZE FEATURES
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 7. RANDOM FOREST TRAINING
# ===============================
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
rf_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20,
                               scoring='accuracy', cv=3, n_jobs=-1, verbose=1, random_state=42)
rf_search.fit(X_train_scaled, y_train_enc)
best_rf = rf_search.best_estimator_
print("Best RF Parameters:", rf_search.best_params_)

# ===============================
# 8. MODEL EVALUATION
# ===============================
y_pred = best_rf.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))

accuracy = accuracy_score(y_test_enc, y_pred)
precision = precision_score(y_test_enc, y_pred, average='weighted')
recall = recall_score(y_test_enc, y_pred, average='weighted')
f1 = f1_score(y_test_enc, y_pred, average='weighted')
y_test_bin = label_binarize(y_test_enc, classes=np.arange(len(label_encoder.classes_)))
y_pred_prob = best_rf.predict_proba(X_test_scaled)
roc_auc = roc_auc_score(y_test_bin, y_pred_prob, average='weighted', multi_class='ovr')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC (OvR): {roc_auc:.4f}")

# ===============================
# 9. CONFUSION MATRIX PLOT
# ===============================
cm = confusion_matrix(y_test_enc, y_pred)
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0)
cm_norm_reset = cm_norm.reset_index().melt(id_vars='index')
cm_norm_reset.columns = ['True Label', 'Predicted Label', 'Proportion']

plt.figure(figsize=(10, 6))
sns.barplot(data=cm_norm_reset, x='True Label', y='Proportion', hue='Predicted Label')
plt.title("Normalized Confusion Matrix (Class Prediction Breakdown)")
plt.xticks(rotation=45)
plt.ylabel("Proportion per True Class")
plt.tight_layout()
plt.show()

# ===============================
# 10. FEATURE IMPORTANCE
# ===============================
importances = best_rf.feature_importances_
top_indices = np.argsort(importances)[-15:][::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[top_indices], y=np.array(X_train.columns)[top_indices], palette="viridis")
plt.title("Top 15 Important Features (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# ===============================
# 11. ROC CURVES (One-vs-Rest)
# ===============================
fpr = dict()
tpr = dict()
roc_auc_dict = dict()
n_classes = len(label_encoder.classes_)

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc_dict[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"{label_encoder.classes_[i]} (AUC = {roc_auc_dict[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ===============================
# 12. T-SNE VISUALIZATION
# ===============================
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
X_tsne = tsne.fit_transform(X_test_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=label_encoder.inverse_transform(y_test_enc), palette='Set2', s=60)
plt.title("Random Forest - t-SNE Projection of Test Set")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# ===============================
# 13. RUNTIME SUMMARY
# ===============================
end_time = time.time()
elapsed = end_time - start_time
print(f"\nRuntime: {int(elapsed // 60)} min {int(elapsed % 60)} sec")