import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# ===================== 1. CARGA Y PREPROCESAMIENTO =====================
print("Cargando datos...")
df = pd.read_csv('df_analisis_final.csv')
df = df.rename(columns={'nivel_apropiacion_retroalimentacion': 'nivel'})

X = df.drop('nivel', axis=1)
y = df['nivel']

# Codificar clases
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"Clases detectadas: {dict(zip(range(len(class_names)), class_names))}")

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"Entrenamiento: {X_train.shape[0]} muestras | Prueba: {X_test.shape[0]} muestras\n")

# ===================== 2. MODELOS Y GRIDSEARCH =====================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Random Forest': (
        RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        {   'n_estimators': [300, 500],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    ),
    'XGBoost': (
        XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False),
        {   'n_estimators': [300, 500],
            'max_depth': [4, 6, 10],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0]
        }
    ),
    'Regresión Logística': (
        LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', multi_class='auto'),
        {'C': [0.1, 1, 10],
         'solver': ['lbfgs']
        }
    ),
    'SVM': (
        SVC(probability=True, random_state=42, class_weight='balanced'),
        {'C': [0.1, 1, 10],
         'kernel': ['rbf', 'poly'],
         'gamma': ['scale', 'auto']
        }
    ),
    'Red Neuronal (MLP)': (
        MLPClassifier(random_state=42, max_iter=1000),
        {   'hidden_layer_sizes': [(100,), (100, 50), (50, 50)],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['adaptive']
        }
    )
}

# ===================== 3. ENTRENAMIENTO Y EVALUACIÓN =====================
results = []
best_model = None
best_name = ""
best_f1_macro_test = -np.inf

print("Entrenando modelos con GridSearchCV...\n" + "-"*60)

for name, (model, params) in models.items():
    print(f"{name}...")
    grid = GridSearchCV(
        model,
        params,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        return_train_score=True
    )
    grid.fit(X_train, y_train)

    # Predicción en test
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)

    # Métricas en test
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

    # Métrica promedio en CV (F1-macro)
    mean_f1_cv = grid.cv_results_['mean_test_score'][grid.best_index_]
    std_f1_cv = grid.cv_results_['std_test_score'][grid.best_index_]

    results.append({
        'Modelo': name,
        'F1-macro CV (mean)': mean_f1_cv,
        'F1-macro CV (std)': std_f1_cv,
        'Accuracy test': acc,
        'Precision test (macro)': prec,
        'Recall test (macro)': rec,
        'F1-score test (macro)': f1,
        'AUC-ROC test (OvR)': auc,
        'Mejor Config': grid.best_params_
    })

    print(f"   CV F1-macro (mean±std): {mean_f1_cv:.4f} ± {std_f1_cv:.4f}")
    print(f"   Test F1-macro: {f1:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

    # Selección de mejor modelo por F1-macro en TEST
    if f1 > best_f1_macro_test:
        best_f1_macro_test = f1
        best_model = grid.best_estimator_
        best_name = name

# ===================== 4. RESULTADOS FINALES =====================
results_df = pd.DataFrame(results)
results_df = results_df.round(4)

# Orden principal: F1-macro TEST, secundario: F1-macro CV
results_df = results_df.sort_values(
    ['F1-score test (macro)', 'F1-macro CV (mean)'],
    ascending=False
)

print("\n" + "="*80)
print("RESULTADOS FINALES - MÉTRICAS CV y TEST")
print("="*80)
print(results_df.to_string(index=False))
print(f"\nMEJOR MODELO (según F1-macro en test): {best_name} (F1-macro test = {best_f1_macro_test:.4f})")

# ===================== 5. MATRIZ DE CONFUSIÓN Y REPORTE =====================
y_pred_best = best_model.predict(X_test)
print(f"\nReporte de clasificación ({best_name}):")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title(f'Matriz de confusión - {best_name}')
plt.tight_layout()
plt.savefig("matriz_confusion_mejor_modelo.png", dpi=300, bbox_inches='tight')
plt.show()

# ===================== 6. IMPORTANCIA DE CARACTERÍSTICAS (Permutation) =====================
print(f"\nCalculando importancia de características ({best_name})...")

perm = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=30,
    random_state=42,
    scoring='f1_macro'
)
sorted_idx = perm.importances_mean.argsort()[::-1][:10]

plt.figure(figsize=(10, 7))
sns.barplot(
    x=perm.importances_mean[sorted_idx],
    y=X.columns[sorted_idx],
    palette="viridis"
)
plt.title(f"Top 10 Características Más Importantes\n(Mejor modelo: {best_name})", fontsize=14, pad=20)
plt.xlabel("Importancia (Permutation Importance)")
plt.tight_layout()
plt.savefig("importancia_caracteristicas_mejor_modelo.png", dpi=300, bbox_inches='tight')
plt.show()

# ===================== 7. GUARDAR MEJOR MODELO =====================
selected_features = X.columns[sorted_idx[:9]].tolist()
feature_indices = sorted_idx[:9].tolist()

final_data = {
    'model': best_model,
    'scaler': scaler,
    'le': le,
    'selected_features': selected_features,
    'feature_indices': feature_indices,
    'all_feature_names': X.columns.tolist(),
    'best_model_name': best_name,
    'metrics_table': results_df.to_dict('records'),
    'classification_report': classification_report(
        y_test, y_pred_best, target_names=class_names, output_dict=True
    )
}

with open('modelo_retroalimentacion.pkl', 'wb') as f:
    pickle.dump(final_data, f)

print("\nModelo final guardado: modelo_retroalimentacion.pkl")
print("Gráficos guardados: matriz_confusion_mejor_modelo.png, importancia_caracteristicas_mejor_modelo.png")