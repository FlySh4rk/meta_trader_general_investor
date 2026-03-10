# NOT_FOREX_TRADER

Infrastruttura ibrida MT5 + ONNX per Crypto e Equities/Indici:
- `MQL5/Bot_Crypto.mq5` (trend-following con spread ampio, 24/7)
- `MQL5/Bot_Equities.mq5` (bias Buy-the-Dip in uptrend)
- `Python/train_hybrid_models.py` (RandomForest -> ONNX label-only)

## 1) Data collection (EA in modalità detective)

1. Compila l'EA desiderato in MetaEditor.
2. Imposta:
   - `InpDataCollectionMode = true`
   - `InpDatasetFileName = Dataset.csv`
3. Avvia backtest/live: l'EA **non apre trade**, ma registra solo le 10 feature:
   - `SlopeNorm, RSI, Hour, Day, ATR_Norm, DistEMA, BB_Pos, Donchian_Pos, isTrend, isLong`
4. Esporta anche il report MT5 dei deal in CSV e rinominalo `ReportTester.csv`.
5. Copia `Dataset.csv` e `ReportTester.csv` nella root:
   - `NOT_FOREX_TRADER/Dataset.csv`
   - `NOT_FOREX_TRADER/ReportTester.csv`

## 2) Training ONNX (gatekeeper Python)

Da terminale, dentro `NOT_FOREX_TRADER`:

```powershell
python -m pip install pandas scikit-learn skl2onnx
python .\Python\train_hybrid_models.py
```

Output:
- `NOT_FOREX_TRADER/hybrid_gatekeeper_rf.onnx`

Dettagli tecnici:
- `RandomForestClassifier(n_estimators=150, max_depth=5)`
- shape input ONNX fissata a `[1,10]` con:
  - `initial_type = [('float_input', FloatTensorType([1, 10]))]`
- **Nuclear Option obbligatoria**:
  - `onnx_model.graph.output.pop()`
  - output finale label-only `[1,1]` (evita il crash MT5 5808)

## 3) Modalità inference (EA con gatekeeper)

1. In EA imposta:
   - `InpDataCollectionMode = false`
   - `InpOnnxModelPath = C:\Users\metat\Desktop\CursorTrader\NOT_FOREX_TRADER\hybrid_gatekeeper_rf.onnx`
2. L'EA invia le 10 feature al modello ONNX.
3. Se l'output label `[1,1]` è `1` -> trade eseguito; se è `0` -> trade scartato.

Architettura runtime:
- `OnTick()` gestisce il trailing stop a **ogni tick** tramite `ManageTrailingStop()`.
- Il controllo "nuova barra" è usato solo per generazione segnale + decisione di trade.
- Donchian reintegrato nel feature engineering (`Donchian_Pos`).

## 4) Tuning spread e magic number

- **Crypto (`Bot_Crypto.mq5`)**
  - `InpMagicNumber` default: `20001`
  - `InpMaxSpreadPoints` default: `6000` (adatto a BTCUSD con spread molto ampio)
  - Se il broker ha spread maggiore, aumenta gradualmente (es. 7000-9000).

- **Equities/Indici (`Bot_Equities.mq5`)**
  - `InpMagicNumber` default: `30001`
  - `InpMaxSpreadPoints` default: `300`
  - Indicazioni:
    - US_Tech100: tipicamente 200-350
    - NVDA CFD: tipicamente 15-40
  - `InpAllowShorts=false` di default per mantenere bias long.
  - Stepped trailing stop: a 50% del target TP, SL spostato a Break-Even + buffer.

## 5) Note operative

- Mantieni file separati per asset class (no mix cross-market nel medesimo dataset).
- Se il naming broker del simbolo cambia (es. `BTCUSD.a`, `USTEC`), l'EA continua a funzionare perché usa `_Symbol`.
- **Hedging bug fix**: per la selezione posizioni usa sempre loop `PositionsTotal()` + `PositionGetTicket()` + `PositionSelectByTicket()` con filtro `_Symbol` + `InpMagicNumber` (mai `PositionSelect(_Symbol)` in `OnTick()`).
- Se MT5 mostra errori ONNX di shape, verifica:
  - input `[1,10]`
  - output `[1,1]`
  - export con probabilità rimossa (`onnx_model.graph.output.pop()`)