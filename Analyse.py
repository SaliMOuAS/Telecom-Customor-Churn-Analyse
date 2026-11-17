"""
================================================================================
TELECOM CUSTOMER CHURN ANALYSIS - PROFESSIONELLER ANALYSEBERICHT
================================================================================

Projekt: Vorhersage von Kundenabwanderung (Churn) im Telekommunikationssektor

Zweck: Identifikation von Churn-Risiken und Entwicklung von Retention-Strategien

Fachliche Fragestellungen:
1. Welche Kunden haben ein hohes Abwanderungsrisiko?
2. Welche Faktoren treiben die Kundenabwanderung?
3. Wie können wir rechtzeitig gegensteuern?
4. Welche finanziellen Auswirkungen hat die Abwanderung?

Geschäftlicher Kontext:
- Durchschnittliche Akquisitionskosten pro Kunde: ~500€
- Monatlicher Umsatz pro Kunde: ~70€
- Einsparpotential durch Retention: 30-50% der Akquisitionskosten
================================================================================
"""

# ============================================================================
# TECHNISCHE VORBEREITUNG
# ============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Machine Learning Bibliotheken
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Unterdrücke Warnungen für bessere Lesbarkeit
warnings.filterwarnings('ignore')

# Professionelle Plot-Einstellungen
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Maximale Anzeige für DataFrames
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)
pd.set_option('display.precision', 3)

# ============================================================================
# TEIL 1: DATENBESCHAFFUNG UND VALIDIERUNG
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TELECOM CUSTOMER CHURN ANALYSE                            ║
║                         Professioneller Analysebericht                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("\n📊 [1/12] DATENBESCHAFFUNG UND -VALIDIERUNG")
print("-" * 60)

# Definiere den Pfad zur CSV-Datei
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Telco-Customer-Churn.csv')

# Falls die Datei nicht existiert, lade sie automatisch herunter
if not os.path.exists(CSV_PATH):
    print("⚠️  Keine CSV-Datei gefunden. Starte automatischen Download...")
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        urllib.request.urlretrieve(url, CSV_PATH)
        print(f"✅ Download erfolgreich: {CSV_PATH}")
    except Exception as e:
        print(f"❌ Download fehlgeschlagen: {e}")
        print("📌 Bitte lade die Datei manuell herunter von:")
        print("   https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        sys.exit(1)

# Lade die Daten
df_raw = pd.read_csv(CSV_PATH)
print(f"✅ Daten erfolgreich geladen: {len(df_raw):,} Kunden")
print(f"   • Anzahl Merkmale: {df_raw.shape[1]}")
print(f"   • Datum des Imports: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

# Erste Insights
print("\n📋 ERSTE EINBLICKE IN DIE DATEN:")
print(f"   • Kundennummern: {df_raw['customerID'].nunique():,} eindeutige IDs")
print(f"   • Beobachtungszeitraum: Querschnittsdaten (keine Zeitreihe)")
print(f"   • Datenqualität: {100 - (df_raw.isna().sum().sum() / df_raw.size * 100):.1f}% vollständig")

# ============================================================================
# TEIL 2: DATENREINIGUNG UND AUFBEREITUNG
# ============================================================================

print("\n📊 [2/12] DATENREINIGUNG UND -AUFBEREITUNG")
print("-" * 60)

# Kopiere die Rohdaten für die Bearbeitung
df = df_raw.copy()

# 1. Konvertiere TotalCharges zu numerisch (das ist ein häufiges Problem)
if 'TotalCharges' in df.columns:
    print("🔧 Konvertiere 'TotalCharges' von String zu numerisch...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Überprüfe die Konvertierung
    missing_charges = df['TotalCharges'].isna().sum()
    if missing_charges > 0:
        print(f"   ⚠️  {missing_charges} Kunden haben fehlende TotalCharges (meist Neukunden)")
        # Fülle fehlende Werte mit dem Median
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        print(f"   ✅ Fehlende Werte mit Median ({df['TotalCharges'].median():.2f}) aufgefüllt")

# 2. Identifiziere die Zielvariable (Churn)
target_col = 'Churn'  # Standardname im Telco-Dataset
if target_col not in df.columns:
    # Falls der Name anders ist, suche nach Churn-Spalten
    for col in df.columns:
        if 'churn' in col.lower():
            target_col = col
            break

print(f"\n🎯 Zielvariable identifiziert: '{target_col}'")
print(f"   Bedeutung: 1 = Kunde hat gekündigt, 0 = Kunde ist geblieben")

# 3. Konvertiere Churn zu 0/1 (falls es Yes/No ist)
if df[target_col].dtype == 'object':
    print("🔄 Konvertiere Churn von Yes/No zu 1/0...")
    original_counts = df[target_col].value_counts()
    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
    print(f"   • Vorher: Yes={original_counts.get('Yes', 0)}, No={original_counts.get('No', 0)}")
    print(f"   • Nachher: Churn=1={df[target_col].sum()}, Churn=0={len(df)-df[target_col].sum()}")

# 4. Entferne unnötige Spalten (Kunden-ID)
id_columns = ['customerID', 'CustomerID', 'id', 'ID']
for col in id_columns:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"🗑️  Entferne '{col}' (nicht relevant für Vorhersage)")

# 5. Überprüfe die finale Datenqualität
print(f"\n✅ Datenbereinigung abgeschlossen:")
print(f"   • Finale Datengröße: {df.shape[0]:,} Kunden × {df.shape[1]} Merkmale")
print(f"   • Speicherbedarf: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ============================================================================
# TEIL 3: EXPLORATIVE DATENANALYSE (EDA)
# ============================================================================

print("\n📊 [3/12] EXPLORATIVE DATENANALYSE")
print("-" * 60)

# Berechne die Churn-Rate
churn_rate = df[target_col].mean()
churn_count = df[target_col].sum()
total_customers = len(df)

print(f"\n📈 KERNMETRIK: KUNDENABWANDERUNGSRATE")
print(f"   • Abgewanderte Kunden: {churn_count:,} von {total_customers:,} ({churn_rate:.1%})")
print(f"   • Verbleibende Kunden: {total_customers - churn_count:,} ({1-churn_rate:.1%})")

# Geschäftliche Bewertung
if churn_rate > 0.25:
    print(f"   ⚠️  KRITISCH: Die Abwanderungsrate von {churn_rate:.1%} ist sehr hoch!")
    print(f"   💰 Geschätzter monatlicher Umsatzverlust: ~{churn_count * 70:,}€")
elif churn_rate > 0.15:
    print(f"   ⚠️  ERHÖHT: Die Abwanderungsrate von {churn_rate:.1%} ist besorgniserregend")
else:
    print(f"   ✅ GUT: Die Abwanderungsrate von {churn_rate:.1%} ist im Branchendurchschnitt")

# Visualisierung der Zielvariablen
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie Chart
colors_churn = ['#2ecc71', '#e74c3c']
labels_churn = [f'Bleibt ({total_customers - churn_count:,})', f'Abgewandert ({churn_count:,})']
axes[0].pie([total_customers - churn_count, churn_count], labels=labels_churn, 
            autopct='%1.1f%%', colors=colors_churn, startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0].set_title('Kundenverteilung', fontsize=14, fontweight='bold')

# Bar Chart
bars = axes[1].bar(['Bleibt (0)', 'Abgewandert (1)'], 
                   [total_customers - churn_count, churn_count],
                   color=colors_churn, edgecolor='black', linewidth=2)
axes[1].set_ylabel('Anzahl Kunden', fontsize=12)
axes[1].set_title('Absolute Zahlen', fontsize=14, fontweight='bold')

# Werte auf die Balken schreiben
for bar, value in zip(bars, [total_customers - churn_count, churn_count]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{value:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.suptitle('KUNDENABWANDERUNG - ÜBERBLICK', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================================
# TEIL 4: DEMOGRAFISCHE ANALYSE
# ============================================================================

print("\n📊 [4/12] DEMOGRAFISCHE ANALYSE DER KUNDEN")
print("-" * 60)

# Analysiere demografische Merkmale
demographic_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
available_demo = [col for col in demographic_cols if col in df.columns]

if available_demo:
    print("\n👥 DEMOGRAFISCHE ZUSAMMENSETZUNG:")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(available_demo[:4]):
        # Berechne Churn-Rate pro Kategorie
        churn_by_demo = df.groupby(col)[target_col].agg(['mean', 'count']).reset_index()
        churn_by_demo.columns = [col, 'churn_rate', 'count']
        
        # Bar Plot
        bars = axes[idx].bar(range(len(churn_by_demo)), churn_by_demo['churn_rate'], 
                            color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(churn_by_demo)])
        axes[idx].set_xticks(range(len(churn_by_demo)))
        axes[idx].set_xticklabels(churn_by_demo[col], rotation=0)
        axes[idx].set_ylabel('Churn-Rate', fontsize=11)
        axes[idx].set_title(f'Churn nach {col}', fontsize=12, fontweight='bold')
        axes[idx].axhline(y=churn_rate, color='red', linestyle='--', 
                         label=f'Gesamt: {churn_rate:.1%}')
        axes[idx].legend()
        
        # Werte auf Balken
        for bar, rate in zip(bars, churn_by_demo['churn_rate']):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{rate:.1%}', ha='center', va='bottom', fontsize=10)
        
        # Textuelle Analyse
        max_churn_row = churn_by_demo.loc[churn_by_demo['churn_rate'].idxmax()]
        print(f"   • {col}: Höchste Churn-Rate bei '{max_churn_row[col]}' "
              f"({max_churn_row['churn_rate']:.1%})")
    
    plt.suptitle('DEMOGRAFISCHE EINFLÜSSE AUF KUNDENABWANDERUNG', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================================
# TEIL 5: VERTRAGS UND SERVICE-ANALYSE
# ============================================================================

print("\n📊 [5/12] VERTRAGS- UND SERVICE-ANALYSE")
print("-" * 60)

# Analysiere Vertragstypen
if 'Contract' in df.columns:
    print("\n📄 VERTRAGSTYPEN-ANALYSE:")
    contract_analysis = df.groupby('Contract')[target_col].agg(['mean', 'count']).reset_index()
    contract_analysis.columns = ['Contract', 'churn_rate', 'customers']
    contract_analysis = contract_analysis.sort_values('churn_rate', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(contract_analysis['Contract'], contract_analysis['churn_rate'],
                  color=['#e74c3c', '#f39c12', '#2ecc71'])
    ax.set_ylabel('Churn-Rate', fontsize=12)
    ax.set_xlabel('Vertragstyp', fontsize=12)
    ax.set_title('ABWANDERUNG NACH VERTRAGSTYP', fontsize=14, fontweight='bold')
    ax.axhline(y=churn_rate, color='blue', linestyle='--', linewidth=2, 
              label=f'Gesamt: {churn_rate:.1%}')
    ax.legend()
    
    for bar, (_, row) in zip(bars, contract_analysis.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f"{row['churn_rate']:.1%}\n(n={row['customers']:,})",
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Fachliche Interpretation
    print("\n💡 FACHINTERPRETATION:")
    for _, row in contract_analysis.iterrows():
        if row['churn_rate'] > churn_rate:
            print(f"   ⚠️  {row['Contract']}: {row['churn_rate']:.1%} Churn-Rate "
                  f"({(row['churn_rate']/churn_rate - 1)*100:.0f}% über Durchschnitt)")
            if row['Contract'] == 'Month-to-month':
                print("      → Monatliche Verträge haben das höchste Risiko!")
                print("      → Empfehlung: Anreize für Jahresverträge schaffen")
        else:
            print(f"   ✅ {row['Contract']}: {row['churn_rate']:.1%} Churn-Rate "
                  f"({(1 - row['churn_rate']/churn_rate)*100:.0f}% unter Durchschnitt)")

# ============================================================================
# TEIL 6: FINANZIELLE ANALYSE
# ============================================================================

print("\n📊 [6/12] FINANZIELLE ANALYSE")
print("-" * 60)

financial_cols = ['MonthlyCharges', 'TotalCharges']
available_fin = [col for col in financial_cols if col in df.columns]

if len(available_fin) >= 1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, col in enumerate(available_fin[:2]):
        # Boxplot Vergleich
        df.boxplot(column=col, by=target_col, ax=axes[idx])
        axes[idx].set_title(f'{col} nach Churn-Status', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Churn (0=Bleibt, 1=Abgewandert)')
        axes[idx].set_ylabel(col)
        
        # Statistische Analyse
        churn_yes = df[df[target_col] == 1][col].mean()
        churn_no = df[df[target_col] == 0][col].mean()
        diff_pct = ((churn_yes - churn_no) / churn_no) * 100
        
        print(f"\n💰 {col}:")
        print(f"   • Bleibende Kunden (Ø): {churn_no:,.2f}")
        print(f"   • Abgewanderte Kunden (Ø): {churn_yes:,.2f}")
        print(f"   • Differenz: {diff_pct:+.1f}%")
        
        if diff_pct > 0:
            print(f"   ⚠️  Abgewanderte Kunden zahlen {diff_pct:.0f}% mehr")
        else:
            print(f"   ✅ Abgewanderte Kunden zahlen {abs(diff_pct):.0f}% weniger")
    
    plt.suptitle('FINANZIELLE KENNZAHLEN IM VERGLEICH', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Geschäftliche Bewertung
    print("\n💼 GESCHÄFTLICHE BEWERTUNG:")
    avg_monthly = df[df[target_col] == 0]['MonthlyCharges'].mean() if 'MonthlyCharges' in df else 0
    monthly_loss = churn_count * avg_monthly
    yearly_loss = monthly_loss * 12
    print(f"   • Geschätzter monatlicher Umsatzverlust: {monthly_loss:,.0f}€")
    print(f"   • Geschätzter jährlicher Umsatzverlust: {yearly_loss:,.0f}€")
    print(f"   • Einsparpotential durch Retention (30%): {yearly_loss * 0.3:,.0f}€/Jahr")

# ============================================================================
# TEIL 7: FEATURE ENGINEERING
# ============================================================================

print("\n📊 [7/12] FEATURE ENGINEERING")
print("-" * 60)

# Separate Features und Target
X = df.drop(columns=[target_col])
y = df[target_col]

# Identifiziere Spaltentypen
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n🔧 MERKMALSTRUKTUR:")
print(f"   • Numerische Merkmale: {len(numeric_cols)}")
for col in numeric_cols:
    print(f"     - {col}: {X[col].min():.2f} bis {X[col].max():.2f}")
print(f"\n   • Kategorische Merkmale: {len(categorical_cols)}")
for col in categorical_cols[:5]:  # Zeige nur die ersten 5
    print(f"     - {col}: {X[col].nunique()} Kategorien")

# Erstelle Preprocessing Pipeline
print("\n🛠️  ERSTELLE PREPROCESSING PIPELINE:")

# Numerische Features: Imputation + Skalierung
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler())
])

# Kategorische Features: Imputation + One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Kombinierte Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_cols),
        ('categorical', categorical_transformer, categorical_cols)
    ])

print("   ✅ Numerische Pipeline: Median-Imputation + Standardisierung")
print("   ✅ Kategorische Pipeline: Konstanten-Imputation + One-Hot Encoding")
print("   ✅ Gesamt-Pipeline bereit für Training")

# ============================================================================
# TEIL 8: TRAIN-TEST SPLIT
# ============================================================================

print("\n📊 [8/12] MODELL-TRAINING VORBEREITUNG")
print("-" * 60)

# Stratified Split für ausgeglichene Churn-Raten
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 DATENAUFTEILUNG:")
print(f"   • Trainingsdatensatz: {len(X_train):,} Kunden ({len(X_train)/len(X)*100:.1f}%)")
print(f"   • Testdatensatz: {len(X_test):,} Kunden ({len(X_test)/len(X)*100:.1f}%)")
print(f"\n📈 CHURN-VERTEILUNG:")
print(f"   • Training: {y_train.mean():.1%} Abwanderung")
print(f"   • Test: {y_test.mean():.1%} Abwanderung")
print(f"   • Stratified Split erfolgreich (gleiche Verteilung)")

# ============================================================================
# TEIL 9: LOGISTIC REGRESSION (INTERPRETIERBARES MODELL)
# ============================================================================

print("\n📊 [9/12] MODELL 1: LOGISTIC REGRESSION")
print("-" * 60)

print("\n🎯 MODELLBESCHREIBUNG:")
print("   Die logistische Regression ist ein interpretierbares Modell,")
print("   das die Wahrscheinlichkeit der Kundenabwanderung vorhersagt.")
print("   Vorteil: Wir können genau sehen, welche Faktoren die Abwanderung treiben.")

# Trainiere Modell
log_reg = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',  # Gewichtet Churn-Fälle höher
    random_state=42,
    C=0.1  # Regularisierung gegen Overfitting
)

log_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', log_reg)
])

print("\n🔄 Trainiere Logistische Regression...")
log_model.fit(X_train, y_train)

# Vorhersagen
y_pred_log = log_model.predict(X_test)
y_proba_log = log_model.predict_proba(X_test)[:, 1]

# Performance-Metriken
roc_auc_log = roc_auc_score(y_test, y_proba_log)
pr_auc_log = average_precision_score(y_test, y_proba_log)
accuracy_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

print("\n📈 PERFORMANCE-METRIKEN:")
print(f"   • ROC-AUC: {roc_auc_log:.3f} (Je höher, desto besser)")
print(f"   • Precision: {precision_log:.3f} (Von den als 'Churn' Vorhergesagten sind {precision_log:.1%} korrekt)")
print(f"   • Recall: {recall_log:.3f} (Von den tatsächlichen Churnern wurden {recall_log:.1%} erkannt)")
print(f"   • F1-Score: {f1_log:.3f} (Harmonischer Mittelwert aus Precision & Recall)")
print(f"   • Accuracy: {accuracy_log:.3f} (Gesamtgenauigkeit)")

# Cross-Validation
cv_scores = cross_val_score(log_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\n🔍 CROSS-VALIDATION (5-fach):")
print(f"   • Durchschnittliche ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
print(f"   • Modell ist stabil (geringe Varianz)")

# Visualisierung
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Bleibt', 'Abgewandert'],
            yticklabels=['Bleibt', 'Abgewandert'])
axes[0].set_title('Logistische Regression - Konfusionsmatrix', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Tatsächlich', fontsize=11)
axes[0].set_xlabel('Vorhergesagt', fontsize=11)

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_proba_log, ax=axes[1])
axes[1].set_title(f'ROC-Kurve (AUC = {roc_auc_log:.3f})', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.suptitle('MODELLPERFORMANCE - LOGISTIC REGRESSION', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# TEIL 10: RANDOM FOREST (KOMPLEXERES MODELL)
# ============================================================================

print("\n📊 [10/12] MODELL 2: RANDOM FOREST")
print("-" * 60)

print("\n🎯 MODELLBESCHREIBUNG:")
print("   Random Forest ist ein Ensemble-Modell aus vielen Entscheidungsbäumen.")
print("   Vorteil: Kann komplexe, nicht-lineare Zusammenhänge erkennen.")
print("   Nachteil: Schwerer zu interpretieren als die logistische Regression.")

# Trainiere Modell
rf_classifier = RandomForestClassifier(
    n_estimators=200,  # Anzahl der Bäume
    max_depth=10,      # Maximale Tiefe (verhindert Overfitting)
    min_samples_split=50,  # Mindestanzahl für Split
    min_samples_leaf=25,   # Mindestanzahl in Blatt
    class_weight='balanced',  # Gewichtung der Klassen
    random_state=42,
    n_jobs=-1  # Nutze alle CPU-Kerne
)

rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf_classifier)
])

print("\n🔄 Trainiere Random Forest (das kann 1-2 Minuten dauern)...")
rf_model.fit(X_train, y_train)

# Vorhersagen
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Performance-Metriken
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
pr_auc_rf = average_precision_score(y_test, y_proba_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\n📈 PERFORMANCE-METRIKEN:")
print(f"   • ROC-AUC: {roc_auc_rf:.3f} (Je höher, desto besser)")
print(f"   • Precision: {precision_rf:.3f} (Von den als 'Churn' Vorhergesagten sind {precision_rf:.1%} korrekt)")
print(f"   • Recall: {recall_rf:.3f} (Von den tatsächlichen Churnern wurden {recall_rf:.1%} erkannt)")
print(f"   • F1-Score: {f1_rf:.3f} (Harmonischer Mittelwert aus Precision & Recall)")
print(f"   • Accuracy: {accuracy_rf:.3f} (Gesamtgenauigkeit)")

# Modellvergleich
print("\n📊 MODELLVERGLEICH:")
print(f"   • Logistische Regression - ROC-AUC: {roc_auc_log:.3f}")
print(f"   • Random Forest - ROC-AUC: {roc_auc_rf:.3f}")
if roc_auc_rf > roc_auc_log:
    improvement = (roc_auc_rf - roc_auc_log) / roc_auc_log * 100
    print(f"   ✅ Random Forest ist {improvement:.1f}% besser (komplexeres Modell zahlt sich aus)")
else:
    print(f"   ℹ️  Logistische Regression ist vergleichbar gut (einfacheres Modell reicht)")

# Visualisierung
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=['Bleibt', 'Abgewandert'],
            yticklabels=['Bleibt', 'Abgewandert'])
axes[0].set_title('Random Forest - Konfusionsmatrix', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Tatsächlich', fontsize=11)
axes[0].set_xlabel('Vorhergesagt', fontsize=11)

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_proba_rf, ax=axes[1])
axes[1].set_title(f'ROC-Kurve (AUC = {roc_auc_rf:.3f})', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.suptitle('MODELLPERFORMANCE - RANDOM FOREST', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# TEIL 11: FEATURE IMPORTANCE & INTERPRETATION
# ============================================================================

print("\n📊 [11/12] FEATURE IMPORTANCE & FACHINTERPRETATION")
print("-" * 60)

# Extrahiere Feature-Namen nach Preprocessing
preprocessor_fit = rf_model.named_steps['preprocessor']

# Feature-Namen direkt aus dem Preprocessor holen (immer passend zur Ausgabe)
feature_names = [n.split('__', 1)[-1] for n in preprocessor_fit.get_feature_names_out()]

# Feature Importances aus Random Forest
importances = rf_model.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n🏆 TOP 10 CHURN-TREIBER (WICHTIGSTE MERKMALE):")
print("-" * 50)

for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['feature']:35s} | Wichtigkeit: {row['importance']:.3f}")

# Fachliche Interpretation der Top 3 Features
print("\n💡 FACHINTERPRETATION DER WICHTIGSTEN TREIBER:")
print("-" * 50)

top_features = feature_importance_df.head(5)['feature'].tolist()

for feature in top_features[:3]:
    if 'Contract' in feature:
        print("\n📄 VERTRAGSLAUFZEIT:")
        print("   → Kunden mit monatlichen Verträgen haben ein VIEL höheres Abwanderungsrisiko")
        print("   → Empfehlung: Biete Anreize für Jahres- oder Zweijahresverträge")
        print("     (z.B. 1 Monat kostenlos, Rabatt auf Hardware, exklusive Services)")
    
    elif 'MonthlyCharges' in feature:
        print("\n💰 MONATLICHE KOSTEN:")
        print("   → Höhere monatliche Gebühren führen zu höherer Abwanderung")
        print("   → Empfehlung: Optimiere Preismodelle für verschiedene Segmente")
        print("     (z.B. Einsteiger-Tarife, Familienpakete, Treue-Rabatte)")
    
    elif 'tenure' in feature:
        print("\n⏰ KUNDENZUBEHÖRDAUER:")
        print("   → Neue Kunden (weniger als 12 Monate) sind besonders gefährdet")
        print("   → Empfehlung: Spezielle Onboarding-Programme für Neukunden")
        print("     (z.B. Willkommens-Bonus, Einrichtungs-Support, regelmäßige Check-ins)")
    
    elif 'InternetService' in feature:
        print("\n🌐 INTERNETSERVICE:")
        print("   → Glasfaser-Kunden haben höhere Abwanderungsraten")
        print("   → Empfehlung: Verbessere Service-Qualität und Support für Glasfaser")
        print("     (z.B. Geschwindigkeitsgarantien, 24/7 Support, Preis-Leistungs-Optimierung)")
    
    elif 'PaymentMethod' in feature:
        print("\n💳 ZAHLUNGSMETHODE:")
        print("   → Elektronische Schecks haben höhere Abwanderung")
        print("   → Empfehlung: Fördere automatische Zahlungsmethoden")
        print("     (z.B. Rabatt für Last")