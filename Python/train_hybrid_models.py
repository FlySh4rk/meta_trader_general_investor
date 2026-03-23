import pandas as pd
import numpy as np
import warnings
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

warnings.filterwarnings('ignore')

# ==========================================
# 🛑 MODIFICA SOLO QUESTI NOMI OGNI VOLTA 🛑
# ==========================================
CSV_FILE = "Dataset.csv"
REPORT_FILE = "ReportTester.csv"
ONNX_FILENAME = "hybrid_gatekeeper_rf.onnx"
# ==========================================

print("--- AVVIO ADDESTRAMENTO IA DEFINITIVO (CRYPTO/EQUITIES) ---")

# 1. LETTURA DATASET EA (ORA CON COLONNA TIME!)
try:
    # engine='python' e sep=None permettono di riconoscere sia virgole che punti e virgola
    df_csv = pd.read_csv(CSV_FILE, sep=None, engine='python')
    if 'Time' not in df_csv.columns:
        print("❌ ERRORE CRITICO: La colonna 'Time' manca nel Dataset.csv!")
        print("Assicurati di aver aggiornato le funzioni MQL5 come indicato.")
        exit(1)
    df_csv['Time'] = pd.to_datetime(df_csv['Time'])
except Exception as e:
    print(f"Errore lettura Dataset: {e}")
    exit(1)

# 2. LETTURA REPORT MT5 (PARSER INTELLIGENTE ANTI-SPAZZATURA)
with open(REPORT_FILE, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

header_idx = -1
delimiter = ','
for i, line in enumerate(lines):
    # Cerca la riga di intestazione della tabella
    if 'Time' in line or 'Ora' in line:
        header_idx = i
        if ';' in line: delimiter = ';'
        elif '\t' in line: delimiter = '\t'
        break

if header_idx == -1:
    print("❌ ERRORE: Impossibile trovare l'intestazione dei trade nel Report MT5.")
    exit(1)

# Carica il CSV usando la libreria standard solo dalla riga corretta in poi
csv_data = "".join(lines[header_idx:])
df_report = pd.read_csv(io.StringIO(csv_data), sep=delimiter)

# Auto-riconoscimento della colonna Orario e Profitto (Italiano/Inglese)
time_col = next((c for c in df_report.columns if c.lower() in ['time', 'ora']), None)
profit_col = next((c for c in df_report.columns if c.lower() in ['profitto', 'profit', 'result', 'p/l']), None)

if not time_col or not profit_col:
    print("❌ ERRORE: Impossibile identificare le colonne Orario o Profitto nel Report MT5.")
    exit(1)

# Pulizia dei dati del report
df_report[time_col] = pd.to_datetime(df_report[time_col], errors='coerce')
df_report[profit_col] = pd.to_numeric(df_report[profit_col], errors='coerce')
df_report = df_report.dropna(subset=[time_col, profit_col])
df_report = df_report[df_report[profit_col] != 0.0] # Scarta depositi e trade nulli

# 3. MERGE ESATTO: LO SCUDO CONTRO I GHOST TRADES
df_merged = pd.merge(df_csv, df_report, left_on='Time', right_on=time_col, how='inner')

if len(df_merged) == 0:
    print("❌ ERRORE CRITICO: Nessun trade combacia tra il Dataset e il Report!")
    print("Questo significa che tutti i trade sono Ghost Trades o le date non coincidono.")
    exit(1)

# Il Target: 1 se in profitto, 0 se in perdita
df_merged['Target'] = (df_merged[profit_col] > 0).astype(np.int64)

# 4. PREPARAZIONE DATI E FEATURES (Le 10 di Cursor)
FEATURES = ['SlopeNorm', 'RSI', 'Hour', 'Day', 'ATR_Norm', 'DistEMA', 'BB_Pos', 'Donchian_Pos', 'isTrend', 'isLong']

for feat in FEATURES:
    if feat not in df_merged.columns:
        print(f"❌ ERRORE: Feature mancante nel dataset: {feat}")
        exit(1)

X = df_merged[FEATURES].values.astype(np.float32)
y = df_merged['Target'].values.astype(np.int64)

# 5. BIAS "BUY THE DIP" PER ASSET FORTI (Azioni/Crypto)
weights = np.ones(len(df_merged), dtype=np.float32)
# Premiamo con un peso x2.5 le entrate Long vicino alla base delle Bollinger in Uptrend
mask_dip = (df_merged['isTrend'] > 0.5) & (df_merged['isLong'] > 0.5) & (df_merged['BB_Pos'] <= 0.20)
weights[mask_dip] = 2.5

# 6. ADDESTRAMENTO RANDOM FOREST
rf = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_split=10, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
rf.fit(X, y, sample_weight=weights)

# 7. VALUTAZIONE E REPORTISTICA A SCHERMO
y_pred = rf.predict(X)

print("\n==================================================")
print("             REPORT DI AFFIDABILITÀ               ")
print("==================================================")
print(f"Trade registrati dal Bot : {len(df_csv)}")
print(f"Trade eseguiti in MT5    : {len(df_report)}")
print(f"Trade Sincronizzati      : {len(df_merged)} (Scartati: {len(df_csv) - len(df_merged)} ghost trades)")
print("--------------------------------------------------")
print(f"Win Rate senza IA        : {(y.sum() / len(y)) * 100:.2f}%")
print(f"Accuratezza Globale IA   : {accuracy_score(y, y_pred) * 100:.2f}%\n")

print("Dettaglio Metriche (0 = Trade da Bloccare, 1 = Trade da Eseguire):")
print(classification_report(y, y_pred))
print("==================================================\n")

# 8. ESPORTAZIONE ONNX: L'OPZIONE NUCLEARE [1, 10]
initial_type = [('float_input', FloatTensorType([1, 10]))]
onnx_model = convert_sklearn(rf, initial_types=initial_type, options={id(rf): {'zipmap': False}})

# Rimozione forzata delle probabilità per evitare l'errore 5808 su MT5
if len(onnx_model.graph.output) < 2:
    print("⚠️ ERRORE ONNX: Struttura inattesa, servono almeno 2 output per il pop().")
    exit(1)

onnx_model.graph.output.pop()

with open(ONNX_FILENAME, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ Modello ONNX esportato con successo in: {ONNX_FILENAME}")