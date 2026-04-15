# Telecom Customer Churn - End-to-End Data Analysis + Modeling
# Author: Data Scientist
# Goal: explain churn drivers + build a baseline model + actionable recommendations
# Visual Studio Code Ready Version - mit direktem Dateipfad

# ----------------------------
# 0) Imports & Config
# ----------------------------
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay,
    accuracy_score, f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# VSCode optimierte Einstellungen
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# Für bessere Ausgabe in VSCode
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("TELECOM CUSTOMER CHURN ANALYSIS")
print("="*80)
print(f"Python Version: {sys.version}")
print(f"Pandas Version: {pd.__version__}")
print("="*80)

# ----------------------------
# 1) Load Data - Direkter Pfad (Lösung 2)
# ----------------------------
# Definiere den direkten Pfad zur CSV-Datei
# Passe diesen Pfad an, falls deine Datei anders heißt
CSV_PATH = '/Users/samsam/Desktop/Telecom Custumor Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Falls die Datei nicht existiert, versuche alternative Namen
if not os.path.exists(CSV_PATH):
    # Alternative Dateinamen
    alternative_names = [
        'Telco-Customer-Churn.csv',
        'telco_customer_churn.csv',
        'WA_Fn-UseC_-Telco-Customer-Churn.csv',
        'churn.csv'
    ]
    
    found = False
    for alt_name in alternative_names:
        alt_path = f'/Users/samsam/Desktop/Telecom Custumor Churn/{alt_name}'
        if os.path.exists(alt_path):
            CSV_PATH = alt_path
            found = True
            break
    
    if not found:
        print(f"\n❌ ERROR: CSV-Datei nicht gefunden unter: {CSV_PATH}")
        print("\n📌 Bitte stelle sicher, dass die Datei existiert.")
        print("   Erwarteter Pfad: /Users/samsam/Desktop/Telecom Custumor Churn/")
        print("\n💡 Mögliche Dateinamen:")
        for name in alternative_names:
            print(f"   - {name}")
        print("\n📥 Falls du die Datei noch nicht hast, lade sie herunter von:")
        print("   https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        sys.exit(1)

print(f"\n📂 Lade Daten von: {CSV_PATH}")
df_raw = pd.read_csv(CSV_PATH)
print(f"✅ Daten erfolgreich geladen!")
print(f"📊 Shape: {df_raw.shape}")
print(f"📋 Spalten: {list(df_raw.columns)}")
print("\n🔍 Erste 5 Zeilen:")
print(df_raw.head())

# ----------------------------
# 2) Basic Cleaning
# ----------------------------
print("\n" + "="*80)
print("DATA CLEANING")
print("="*80)

# Bereinige TotalCharges - konvertiere zu numerisch
if 'TotalCharges' in df_raw.columns:
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')

col_lower_map = {c.lower(): c for c in df_raw.columns}

def pick_target_column(df_cols):
    candidates = ['churn', 'churned', 'is_churn', 'attrition', 'exited', 'churn?']
    for c in candidates:
        if c in df_cols:
            return df_cols[c]
    return None

target_col = pick_target_column(col_lower_map)

if target_col is None:
    for c in df_raw.columns:
        if 'churn' in c.lower() and df_raw[c].nunique(dropna=True) <= 2:
            target_col = c
            break

if target_col is None:
    print("\n❌ Konnte Target-Spalte nicht automatisch erkennen!")
    print("📌 Verfügbare Spalten:", list(df_raw.columns))
    print("💡 Bitte benenne deine Churn-Spalte in 'Churn' um oder aktualisiere das Skript")
    sys.exit(1)

print(f"🎯 Target-Spalte erkannt: '{target_col}'")

# Drop ID columns
id_like_cols = []
for c in df_raw.columns:
    if c.lower() in ['customerid', 'customer_id', 'id', 'userid', 'user_id', 'customerid?']:
        id_like_cols.append(c)

df = df_raw.copy()

# Normalize target to 0/1
print(f"\n🔄 Konvertiere Target-Spalte '{target_col}' zu binär...")
if df[target_col].dtype == 'O':
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    df[target_col] = df[target_col].map({
        'yes': 1, 'y': 1, 'true': 1, '1': 1, 
        'no': 0, 'n': 0, 'false': 0, '0': 0
    })

if df[target_col].isna().any():
    df[target_col] = df_raw[target_col]
    if df[target_col].dtype == 'O':
        df[target_col] = df[target_col].astype(str).str.strip().str.lower()
        df[target_col] = df[target_col].replace({'churn': 1, 'not churn': 0})
        df[target_col] = df[target_col].map({'yes': 1, 'no': 0})

if df[target_col].isna().any():
    print("\n❌ Target-Spalte enthält nicht zuordenbare Werte!")
    print(f"Unique values: {df[target_col].unique()}")
    sys.exit(1)

print(f"✅ Target-Spalte konvertiert. Unique values: {df[target_col].unique()}")

if len(id_like_cols) > 0:
    print(f"\n🗑️ Entferne ID-Spalten: {id_like_cols}")
    df = df.drop(columns=id_like_cols)

print(f"📊 Shape nach Bereinigung: {df.shape}")

# ----------------------------
# 3) Quick EDA
# ----------------------------
print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

churn_rate = df[target_col].mean()
print(f"📈 Gesamte Churn-Rate: {round(churn_rate * 100, 2)}%")

# Missing values
missing = (df.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing[missing > 0]
if len(missing_pct) > 0:
    print(f"\n⚠️ Fehlende Werte (%):")
    print(missing_pct)
    # Fülle fehlende Werte in TotalCharges mit Median
    if 'TotalCharges' in df.columns:
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
else:
    print("\n✅ Keine fehlenden Werte gefunden!")

# Basic statistics
print("\n📊 Numerische Spalten - Statistiken:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].describe())

print("\n📊 Kategorische Spalten:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts().head())

# Visualizations
print("\n🎨 Generiere Visualisierungen...")

# Target distribution
plt.figure(figsize=(8, 6))
ax = sns.countplot(x=target_col, data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Target Distribution (0=No Churn, 1=Churn)', fontsize=14, fontweight='bold')
plt.xlabel('Churn Status')
plt.ylabel('Count')
# Add percentage labels
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    ax.annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

# ----------------------------
# 4) Split Columns
# ----------------------------
print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

print(f"📊 Feature Aufteilung:")
print(f"   • Numerische Features: {len(num_cols)}")
print(f"   • Kategorische Features: {len(cat_cols)}")
print(f"   • Total Features: {len(num_cols) + len(cat_cols)}")
print(f"   • Target: {target_col}")

if len(num_cols) > 0:
    print(f"\n📈 Numerische Features: {num_cols[:10]}{'...' if len(num_cols) > 10 else ''}")
if len(cat_cols) > 0:
    print(f"📋 Kategorische Features: {cat_cols[:10]}{'...' if len(cat_cols) > 10 else ''}")

# ----------------------------
# 5) EDA: Numeric vs Churn
# ----------------------------
if len(num_cols) > 0:
    print("\n" + "="*80)
    print("NUMERISCHE FEATURES ANALYSE")
    print("="*80)
    
    mean_by_class = df.groupby(target_col)[num_cols].mean().T
    mean_by_class.columns = ['no_churn_mean', 'churn_mean']
    mean_by_class['abs_diff'] = (mean_by_class['churn_mean'] - mean_by_class['no_churn_mean']).abs()
    mean_by_class['diff_pct'] = ((mean_by_class['churn_mean'] - mean_by_class['no_churn_mean']) / mean_by_class['no_churn_mean'] * 100)
    mean_by_class = mean_by_class.sort_values('abs_diff', ascending=False)
    
    print("\n📊 Top numerische Features nach Churn-Impact:")
    print(mean_by_class.head(10))
    
    # Plot top numeric features
    top_num = list(mean_by_class.head(6).index)
    for i, c in enumerate(top_num):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x=target_col, y=c, palette=['#2ecc71', '#e74c3c'])
        plt.title(f'{c} Distribution by Churn', fontweight='bold')
        
        plt.subplot(1, 2, 2)
        sns.kdeplot(data=df, x=c, hue=target_col, common_norm=False, fill=True, alpha=0.5)
        plt.title(f'{c} Density by Churn', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# ----------------------------
# 6) EDA: Categorical vs Churn
# ----------------------------
if len(cat_cols) > 0:
    print("\n" + "="*80)
    print("KATEGORISCHE FEATURES ANALYSE")
    print("="*80)
    
    def plot_cat_churn(colname, max_cats=10):
        tmp = df[[colname, target_col]].copy()
        tmp[colname] = tmp[colname].astype(str).fillna('Missing')
        grp = tmp.groupby(colname)[target_col].agg(['mean', 'count', 'size']).reset_index()
        grp = grp.sort_values('count', ascending=False)
        grp = grp.head(max_cats)
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(grp)), grp['mean'], color=['#e74c3c' if x > churn_rate else '#2ecc71' for x in grp['mean']])
        plt.yticks(range(len(grp)), grp[colname])
        plt.xlabel('Churn Rate')
        plt.title(f'Churn Rate by {colname}', fontsize=14, fontweight='bold')
        plt.xlim(0, 1)
        
        # Add value labels
        for i, (rate, count) in enumerate(zip(grp['mean'], grp['count'])):
            plt.text(rate + 0.02, i, f'{rate:.1%} (n={count})', va='center')
        
        plt.axvline(x=churn_rate, color='gray', linestyle='--', label=f'Overall Churn Rate: {churn_rate:.1%}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return grp
    
    cat_unique_counts = pd.Series({c: df[c].nunique(dropna=True) for c in cat_cols}).sort_values()
    cols_to_plot = list(cat_unique_counts[cat_unique_counts <= 15].head(6).index)
    
    if len(cols_to_plot) == 0:
        cols_to_plot = cat_cols[:3]
    
    for c in cols_to_plot:
        print(f"\n📊 Analysiere '{c}':")
        plot_cat_churn(c, max_cats=12)

# ----------------------------
# 7) Train/Test split
# ----------------------------
print("\n" + "="*80)
print("MODELL TRAINING")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Trainingsset Größe: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"📊 Testset Größe: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"📊 Churn-Rate im Training: {y_train.mean():.1%}")
print(f"📊 Churn-Rate im Test: {y_test.mean():.1%}")

# ----------------------------
# 8) Preprocessing pipeline
# ----------------------------
print("\n🔧 Baue Preprocessing Pipeline...")

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)

print("✅ Preprocessing Pipeline bereit")

# ----------------------------
# 9) Model 1: Logistic Regression
# ----------------------------
print("\n" + "-"*40)
print("MODELL 1: LOGISTIC REGRESSION")
print("-"*40)

log_reg = LogisticRegression(
    max_iter=2000, 
    class_weight='balanced',
    random_state=42,
    C=1.0
)
log_clf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', log_reg)
])

print("🔄 Trainiere Logistic Regression...")
log_clf.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(log_clf, X_train, y_train, cv=5, scoring='roc_auc')
print(f"📊 Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Predictions
proba_test = log_clf.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)

print(f"\n📈 Performance Metriken:")
print(f"   • ROC-AUC: {roc_auc_score(y_test, proba_test):.4f}")
print(f"   • PR-AUC: {average_precision_score(y_test, proba_test):.4f}")
print(f"   • Accuracy: {accuracy_score(y_test, pred_test):.4f}")
print(f"   • F1-Score: {f1_score(y_test, pred_test):.4f}")

print(f"\n📋 Klassifikationsreport:")
print(classification_report(y_test, pred_test, digits=3))

# Confusion Matrix
cm = confusion_matrix(y_test, pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)

# Add percentages
for i in range(2):
    for j in range(2):
        plt.text(j+0.5, i+0.5, f'{cm[i,j]}\n({cm[i,j]/cm.sum():.1%})', 
                ha='center', va='center', fontsize=10)
plt.tight_layout()
plt.show()

# ROC & PR Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

RocCurveDisplay.from_predictions(y_test, proba_test, ax=axes[0])
axes[0].set_title('ROC Curve - Logistic Regression', fontweight='bold')
axes[0].grid(True, alpha=0.3)

PrecisionRecallDisplay.from_predictions(y_test, proba_test, ax=axes[1])
axes[1].set_title('Precision-Recall Curve - Logistic Regression', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------
# 10) Model 2: Random Forest
# ----------------------------
print("\n" + "-"*40)
print("MODELL 2: RANDOM FOREST")
print("-"*40)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced_subsample',
    n_jobs=-1
)

rf_clf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', rf)
])

print("🔄 Trainiere Random Forest...")
rf_clf.fit(X_train, y_train)

# Cross-validation
rf_cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring='roc_auc')
print(f"📊 Cross-Validation ROC-AUC: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std()*2:.4f})")

# Predictions
rf_proba_test = rf_clf.predict_proba(X_test)[:, 1]
rf_pred_test = (rf_proba_test >= 0.5).astype(int)

print(f"\n📈 Performance Metriken:")
print(f"   • ROC-AUC: {roc_auc_score(y_test, rf_proba_test):.4f}")
print(f"   • PR-AUC: {average_precision_score(y_test, rf_proba_test):.4f}")
print(f"   • Accuracy: {accuracy_score(y_test, rf_pred_test):.4f}")
print(f"   • F1-Score: {f1_score(y_test, rf_pred_test):.4f}")

print(f"\n📋 Klassifikationsreport:")
print(classification_report(y_test, rf_pred_test, digits=3))

# Confusion Matrix
cm2 = confusion_matrix(y_test, rf_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)

for i in range(2):
    for j in range(2):
        plt.text(j+0.5, i+0.5, f'{cm2[i,j]}\n({cm2[i,j]/cm2.sum():.1%})', 
                ha='center', va='center', fontsize=10)
plt.tight_layout()
plt.show()

# ROC & PR Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

RocCurveDisplay.from_predictions(y_test, rf_proba_test, ax=axes[0])
axes[0].set_title('ROC Curve - Random Forest', fontweight='bold')
axes[0].grid(True, alpha=0.3)

PrecisionRecallDisplay.from_predictions(y_test, rf_proba_test, ax=axes[1])
axes[1].set_title('Precision-Recall Curve - Random Forest', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------
# 11) Feature Importance & Interpretation
# ----------------------------
print("\n" + "="*80)
print("MODELL INTERPRETATION")
print("="*80)

# Logistic Regression Coefficients
preprocess_fit = log_clf.named_steps['preprocess']

feature_names = []
feature_names += num_cols
if len(cat_cols) > 0:
    ohe = preprocess_fit.named_transformers_['cat'].named_steps['onehot']
    ohe_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names += ohe_names

coef_vals = log_clf.named_steps['model'].coef_.ravel()
coef_df = pd.DataFrame({'feature': feature_names, 'coef': coef_vals})
coef_df['abs_coef'] = coef_df['coef'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False)

print("\n🏆 Top 20 Logistic Regression Koeffizienten (Churn-Treiber):")
print(coef_df.head(20).to_string())

plt.figure(figsize=(12, 8))
colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in coef_df.head(15)['coef']]
sns.barplot(data=coef_df.head(15), x='coef', y='feature', palette=colors, orient='h')
plt.title('Top 15 Logistic Regression Koeffizienten', fontsize=14, fontweight='bold')
plt.xlabel('Koeffizientenwert (Impact auf Churn-Wahrscheinlichkeit)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.axvline(0, color='black', linewidth=1, linestyle='-')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Random Forest Feature Importance
rf_preprocess_fit = rf_clf.named_steps['preprocess']

rf_feature_names = []
rf_feature_names += num_cols
if len(cat_cols) > 0:
    rf_ohe = rf_preprocess_fit.named_transformers_['cat'].named_steps['onehot']
    rf_feature_names += list(rf_ohe.get_feature_names_out(cat_cols))

rf_model = rf_clf.named_steps['model']
imp = rf_model.feature_importances_
imp_df = pd.DataFrame({'feature': rf_feature_names, 'importance': imp}).sort_values('importance', ascending=False)

print("\n🏆 Top 20 Random Forest Feature Importances:")
print(imp_df.head(20).to_string())

plt.figure(figsize=(12, 8))
sns.barplot(data=imp_df.head(15), x='importance', y='feature', palette='viridis', orient='h')
plt.title('Top 15 Random Forest Feature Importances', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------
# 12) Actionable Segments & Recommendations
# ----------------------------
print("\n" + "="*80)
print("AKTIONSBARE EINBLICKE & EMPFEHLUNGEN")
print("="*80)

# Risk segmentation
risk_df = X_test.copy()
risk_df['y_true'] = y_test.values
risk_df['p_churn_rf'] = rf_proba_test
risk_df['p_churn_lr'] = proba_test
risk_df['risk_band'] = pd.cut(
    risk_df['p_churn_rf'],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=['Very Low (0-20%)', 'Low (20-40%)', 'Medium (40-60%)', 'High (60-80%)', 'Very High (80-100%)'],
    include_lowest=True
)

band = risk_df.groupby('risk_band', observed=True).agg(
    customers=('p_churn_rf', 'size'),
    avg_risk=('p_churn_rf', 'mean'),
    actual_churn_rate=('y_true', 'mean')
).reset_index()

band['customers_share'] = band['customers'] / band['customers'].sum()
band['potential_savings'] = band['customers'] * band['avg_risk'] * 100

print("\n📊 Risikoband Analyse:")
print(band.to_string())

# Visualize risk bands
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs Predicted churn rate
axes[0].bar(range(len(band)), band['actual_churn_rate'], color=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b'])
axes[0].set_xticks(range(len(band)))
axes[0].set_xticklabels(band['risk_band'], rotation=45, ha='right')
axes[0].set_ylabel('Churn Rate')
axes[0].set_title('Tatsächliche Churn-Rate nach Vorhergesagtem Risikoband', fontweight='bold')
axes[0].set_ylim(0, 1)
axes[0].axhline(y=churn_rate, color='blue', linestyle='--', label=f'Gesamt: {churn_rate:.1%}')
axes[0].legend()

# Customer distribution
axes[1].pie(band['customers'], labels=band['risk_band'], autopct='%1.1f%%', startangle=90)
axes[1].set_title('Kundenverteilung nach Risikoband', fontweight='bold')

plt.tight_layout()
plt.show()

# Specific recommendations
print("\n💡 SPEZIFISCHE EMPFEHLUNGEN:")
print("\n1. HOCH-RISIKO KUNDEN (60-100% Churn-Wahrscheinlichkeit):")
print("   • Proaktive Retention-Anrufe innerhalb von 7 Tagen")
print("   • Biete Loyalitätsrabatte oder exklusive Vorteile an")
print("   • Priorisierten Kundensupport-Zugang")
print("   • Personalisierte Retention-Pakete")

print("\n2. MITTEL-RISIKO KUNDEN (40-60% Churn-Wahrscheinlichkeit):")
print("   • Sende Zufriedenheitsumfragen")
print("   • Gezielte Email-Kampagnen mit speziellen Angeboten")
print("   • Produktnutzungs-Tipps und Tutorials")
print("   • Regelmäßige Check-in-Anrufe")

print("\n3. NIEDRIG-RISIKO KUNDEN (0-40% Churn-Wahrscheinlichkeit):")
print("   • Pflege regelmäßiger Kommunikation")
print("   • Cross-Selling und Upselling Möglichkeiten")
print("   • Empfehlungsprogramme")
print("   • Treueprogramm-Einschreibung")

print("\n📈 WICHTIGSTE CHURN-TREIBER DIE ANGEGANGEN WERDEN MÜSSEN:")
top_drivers = coef_df.head(10)[coef_df['coef'] > 0]['feature'].tolist()
if len(top_drivers) > 0:
    for i, driver in enumerate(top_drivers, 1):
        print(f"   {i}. {driver}")

# ----------------------------
# 13) Save outputs
# ----------------------------
print("\n" + "="*80)
print("SPEICHERE ERGEBNISSE")
print("="*80)

OUT_DIR = './outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# Save all outputs
coef_df.to_csv(os.path.join(OUT_DIR, 'logreg_top_coefficients.csv'), index=False)
imp_df.to_csv(os.path.join(OUT_DIR, 'rf_feature_importances.csv'), index=False)
band.to_csv(os.path.join(OUT_DIR, 'risk_bands_summary.csv'), index=False)

# Save predictions
predictions_df = pd.DataFrame({
    'actual_churn': y_test,
    'rf_churn_probability': rf_proba_test,
    'lr_churn_probability': proba_test,
    'rf_prediction': rf_pred_test,
    'lr_prediction': pred_test
})
predictions_df.to_csv(os.path.join(OUT_DIR, 'predictions.csv'), index=False)

# Save model summary
with open(os.path.join(OUT_DIR, 'model_summary.txt'), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("TELECOM CHURN MODEL SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset size: {len(df)} rows, {len(X.columns)} features\n")
    f.write(f"Churn rate: {churn_rate:.1%}\n\n")
    f.write("MODEL PERFORMANCE:\n")
    f.write("-"*40 + "\n")
    f.write(f"Logistic Regression:\n")
    f.write(f"  • ROC-AUC: {roc_auc_score(y_test, proba_test):.4f}\n")
    f.write(f"  • PR-AUC: {average_precision_score(y_test, proba_test):.4f}\n")
    f.write(f"  • Accuracy: {accuracy_score(y_test, pred_test):.4f}\n")
    f.write(f"  • F1-Score: {f1_score(y_test, pred_test):.4f}\n\n")
    f.write(f"Random Forest:\n")
    f.write(f"  • ROC-AUC: {roc_auc_score(y_test, rf_proba_test):.4f}\n")
    f.write(f"  • PR-AUC: {average_precision_score(y_test, rf_proba_test):.4f}\n")
    f.write(f"  • Accuracy: {accuracy_score(y_test, rf_pred_test):.4f}\n")
    f.write(f"  • F1-Score: {f1_score(y_test, rf_pred_test):.4f}\n\n")
    f.write("TOP FEATURES:\n")
    f.write("-"*40 + "\n")
    f.write("Logistic Regression (Top 5):\n")
    for i, row in coef_df.head(5).iterrows():
        f.write(f"  • {row['feature']}: {row['coef']:.4f}\n")
    f.write("\nRandom Forest (Top 5):\n")
    for i, row in imp_df.head(5).iterrows():
        f.write(f"  • {row['feature']}: {row['importance']:.4f}\n")

print(f"✅ Alle Ergebnisse gespeichert in: {OUT_DIR}")
print(f"   • logreg_top_coefficients.csv")
print(f"   • rf_feature_importances.csv")
print(f"   • risk_bands_summary.csv")
print(f"   • predictions.csv")
print(f"   • model_summary.txt")

print("\n" + "="*80)
print("✅ ANALYSE ABGESCHLOSSEN!")
print("="*80)
print("\n🎯 Nächste Schritte:")
print("   1. Überprüfe die model_summary.txt für wichtige Einblicke")
print("   2. Implementiere das Hochrisiko-Kundenretentionsprogramm")
print("   3. Richte wöchentliche Überwachung der Churn-Vorhersagen ein")
print("   4. Teste die Empfehlungen mit A/B-Tests auf einem kleinen Segment")
print("   5. Plane regelmäßiges Modell-Retraining (monatlich)")

plt.show()













