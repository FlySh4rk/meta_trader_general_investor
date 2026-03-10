#property strict
#property version   "1.00"
#property description "Hybrid Detective/Gatekeeper EA for Crypto symbols"

#include <Trade/Trade.mqh>

input bool   InpDataCollectionMode = true;
input string InpDatasetFileName    = "Dataset.csv";
input string InpOnnxModelPath      = "C:\\Users\\metat\\Desktop\\CursorTrader\\NOT_FOREX_TRADER\\hybrid_gatekeeper_rf.onnx";
input long   InpMagicNumber        = 20001;
input int    InpMaxSpreadPoints    = 6000;
input double InpLots               = 0.01;

input int    InpEMAPeriod          = 100;
input int    InpBBPeriod           = 20;
input double InpBBDeviation        = 2.0;
input int    InpRSIPeriod          = 14;
input int    InpATRPeriod          = 14;
input int    InpDonchianPeriod     = 20;
input double InpTrendSlopeThreshold= 0.12;

input double InpBBPosLongMax       = 0.35;
input double InpBBPosShortMin      = 0.65;
input double InpRSILongMax         = 48.0;
input double InpRSIShortMin        = 52.0;

input double InpSL_ATR_Mult        = 2.2;
input double InpTP_ATR_Mult        = 3.0;
input double InpTrailing_ATR_Mult  = 1.4;

CTrade trade;
int g_ema = INVALID_HANDLE;
int g_bb = INVALID_HANDLE;
int g_rsi = INVALID_HANDLE;
int g_atr = INVALID_HANDLE;
long g_onnx = INVALID_HANDLE;
datetime g_lastBarTime = 0;

void ManageTrailingStop();
bool BuildBaseSignalAndFeatures(float &features[][10], bool &isLong);
void AppendDatasetRow(const float &features[][10]);

int OnInit()
{
   g_ema = iMA(_Symbol, _Period, InpEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   g_bb  = iBands(_Symbol, _Period, InpBBPeriod, 0, InpBBDeviation, PRICE_CLOSE);
   g_rsi = iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE);
   g_atr = iATR(_Symbol, _Period, InpATRPeriod);

   if(g_ema == INVALID_HANDLE || g_bb == INVALID_HANDLE || g_rsi == INVALID_HANDLE || g_atr == INVALID_HANDLE)
   {
      Print("Errore creazione handle indicatori: ", GetLastError());
      return INIT_FAILED;
   }

   int dcHighestShift = iHighest(_Symbol, _Period, MODE_HIGH, InpDonchianPeriod, 1);
   int dcLowestShift  = iLowest(_Symbol, _Period, MODE_LOW, InpDonchianPeriod, 1);
   if(dcHighestShift < 0 || dcLowestShift < 0)
   {
      Print("Storico insufficiente per Donchian(", InpDonchianPeriod, "). Errore: ", GetLastError());
      return INIT_FAILED;
   }

   if(!InpDataCollectionMode)
   {
      g_onnx = OnnxCreate(InpOnnxModelPath, ONNX_DEFAULT);
      if(g_onnx == INVALID_HANDLE)
      {
         Print("Errore OnnxCreate: ", GetLastError(), " path=", InpOnnxModelPath);
         return INIT_FAILED;
      }

      const long input_shape[] = {1, 10};
      if(!OnnxSetInputShape(g_onnx, 0, input_shape))
      {
         Print("Errore OnnxSetInputShape [1,10]: ", GetLastError());
         return INIT_FAILED;
      }

      const long output_shape[] = {1, 1};
      if(!OnnxSetOutputShape(g_onnx, 0, output_shape))
      {
         Print("Errore OnnxSetOutputShape [1,1]: ", GetLastError());
         return INIT_FAILED;
      }
   }

   EnsureDatasetHeader();
   trade.SetExpertMagicNumber(InpMagicNumber);
   return INIT_SUCCEEDED;
}


void OnDeinit(const int reason)
{
   if(g_ema != INVALID_HANDLE) IndicatorRelease(g_ema);
   if(g_bb  != INVALID_HANDLE) IndicatorRelease(g_bb);
   if(g_rsi != INVALID_HANDLE) IndicatorRelease(g_rsi);
   if(g_atr != INVALID_HANDLE) IndicatorRelease(g_atr);
   if(g_onnx != INVALID_HANDLE) OnnxRelease(g_onnx);
}


void OnTick()
{
   ManageTrailingStop();

   datetime currBar = iTime(_Symbol, _Period, 0);
   if(currBar == g_lastBarTime)
      return;
   g_lastBarTime = currBar;

   if(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) > InpMaxSpreadPoints)
      return;

   if(HasOpenPositionByMagic())
      return;

   float features[1][10];
   bool isLong = false;
   bool hasSignal = BuildBaseSignalAndFeatures(features, isLong);
   if(!hasSignal)
      return;

   if(InpDataCollectionMode)
   {
      AppendDatasetRow(features);
      return;
   }

   long output_label[1][1];
   output_label[0][0] = 0;
   if(!OnnxRun(g_onnx, ONNX_NO_CONVERSION, features, output_label))
   {
      Print("OnnxRun fallita: ", GetLastError());
      return;
   }

   int pred = (int)output_label[0][0];
   if(pred == 1)
      ExecuteTrade(isLong, (double)features[0][4], (double)features[0][5]);
}


bool BuildBaseSignalAndFeatures(float &features[][10], bool &isLong)
{
   const int shift = 1;
   double emaBuff[2], rsiBuff[1], atrBuff[1], bbUp[1], bbMid[1], bbLow[1];
   if(CopyBuffer(g_ema, 0, shift, 2, emaBuff) <= 0) return false;
   if(CopyBuffer(g_rsi, 0, shift, 1, rsiBuff) <= 0) return false;
   if(CopyBuffer(g_atr, 0, shift, 1, atrBuff) <= 0) return false;
   if(CopyBuffer(g_bb, 0, shift, 1, bbUp) <= 0) return false;
   if(CopyBuffer(g_bb, 1, shift, 1, bbMid) <= 0) return false;
   if(CopyBuffer(g_bb, 2, shift, 1, bbLow) <= 0) return false;

   double close1 = iClose(_Symbol, _Period, shift);
   if(close1 <= 0.0) return false;

   double atr = atrBuff[0];
   if(atr <= _Point) return false;

   double slopeNorm = (emaBuff[0] - emaBuff[1]) / atr;
   bool isTrend = (MathAbs(slopeNorm) >= InpTrendSlopeThreshold);
   bool regimeUp = (emaBuff[0] > emaBuff[1]) && isTrend;
   bool regimeDown = (emaBuff[0] < emaBuff[1]) && isTrend;

   double bbRange = bbUp[0] - bbLow[0];
   if(bbRange <= 0.0) return false;
   double bbPos = (close1 - bbLow[0]) / bbRange;
   if(bbPos < 0.0) bbPos = 0.0;
   if(bbPos > 1.0) bbPos = 1.0;

   int dcHighestShift = iHighest(_Symbol, _Period, MODE_HIGH, InpDonchianPeriod, shift);
   int dcLowestShift  = iLowest(_Symbol, _Period, MODE_LOW, InpDonchianPeriod, shift);
   if(dcHighestShift < 0 || dcLowestShift < 0) return false;

   double dcUpper = iHigh(_Symbol, _Period, dcHighestShift);
   double dcLower = iLow(_Symbol, _Period, dcLowestShift);
   double dcRange = dcUpper - dcLower;
   if(dcRange <= 0.0) return false;

   double donchianPos = (close1 - dcLower) / dcRange;
   if(donchianPos < 0.0) donchianPos = 0.0;
   if(donchianPos > 1.0) donchianPos = 1.0;

   double rsi = rsiBuff[0];
   bool longSignal = regimeUp && bbPos <= InpBBPosLongMax && rsi <= InpRSILongMax;
   bool shortSignal = regimeDown && bbPos >= InpBBPosShortMin && rsi >= InpRSIShortMin;

   if(!longSignal && !shortSignal)
      return false;

   isLong = longSignal;
   datetime t = iTime(_Symbol, _Period, shift);
   MqlDateTime dt;
   TimeToStruct(t, dt);

   features[0][0] = (float)slopeNorm;                    // SlopeNorm
   features[0][1] = (float)rsi;                          // RSI
   features[0][2] = (float)dt.hour;                      // Hour
   features[0][3] = (float)dt.day_of_week;               // Day
   features[0][4] = (float)(atr / close1);               // ATR_Norm
   features[0][5] = (float)((close1 - emaBuff[0]) / atr);// DistEMA
   features[0][6] = (float)bbPos;                        // BB_Pos
   features[0][7] = (float)donchianPos;                  // Donchian_Pos
   features[0][8] = isTrend ? 1.0f : 0.0f;               // isTrend
   features[0][9] = isLong ? 1.0f : 0.0f;                // isLong
   return true;
}


void ManageTrailingStop()
{
   double atrBuff[1];
   if(CopyBuffer(g_atr, 0, 0, 1, atrBuff) <= 0)
      return;
   double atr = atrBuff[0];
   if(atr <= _Point)
      return;

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(bid <= 0.0 || ask <= 0.0)
      return;

   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      long mg = PositionGetInteger(POSITION_MAGIC);
      if(sym != _Symbol || mg != InpMagicNumber)
         continue;

      long type = PositionGetInteger(POSITION_TYPE);
      double currentSL = PositionGetDouble(POSITION_SL);
      double currentTP = PositionGetDouble(POSITION_TP);
      double newSL = currentSL;

      if(type == POSITION_TYPE_BUY)
      {
         double candidate = NormalizeDouble(bid - InpTrailing_ATR_Mult * atr, _Digits);
         if(currentSL <= 0.0 || candidate > currentSL)
            newSL = candidate;
      }
      else if(type == POSITION_TYPE_SELL)
      {
         double candidate = NormalizeDouble(ask + InpTrailing_ATR_Mult * atr, _Digits);
         if(currentSL <= 0.0 || candidate < currentSL)
            newSL = candidate;
      }
      else
      {
         continue;
      }

      if(newSL != currentSL)
         trade.PositionModify(ticket, newSL, currentTP);
   }
}


void ExecuteTrade(bool isLong, double atrNorm, double distEMA)
{
   double atrBuff[1];
   if(CopyBuffer(g_atr, 0, 1, 1, atrBuff) <= 0)
      return;
   double atr = atrBuff[0];
   if(atr <= _Point)
      return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0)
      return;

   double sl = 0.0, tp = 0.0;
   if(isLong)
   {
      sl = NormalizeDouble(ask - InpSL_ATR_Mult * atr, _Digits);
      tp = NormalizeDouble(ask + InpTP_ATR_Mult * atr, _Digits);
      trade.Buy(InpLots, _Symbol, 0.0, sl, tp, "CryptoHybrid");
   }
   else
   {
      sl = NormalizeDouble(bid + InpSL_ATR_Mult * atr, _Digits);
      tp = NormalizeDouble(bid - InpTP_ATR_Mult * atr, _Digits);
      trade.Sell(InpLots, _Symbol, 0.0, sl, tp, "CryptoHybrid");
   }
}


bool HasOpenPositionByMagic()
{
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      long mg = PositionGetInteger(POSITION_MAGIC);
      if(sym == _Symbol && mg == InpMagicNumber)
         return true;
   }
   return false;
}


void EnsureDatasetHeader()
{
   int handle = FileOpen(InpDatasetFileName, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return;

   if(FileSize(handle) == 0)
   {
      FileWrite(handle, "SlopeNorm", "RSI", "Hour", "Day", "ATR_Norm", "DistEMA", "BB_Pos", "Donchian_Pos", "isTrend", "isLong");
   }
   FileClose(handle);
}


void AppendDatasetRow(const float &features[][10])
{
   int handle = FileOpen(InpDatasetFileName, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
   {
      Print("FileOpen dataset fallita: ", GetLastError());
      return;
   }

   FileSeek(handle, 0, SEEK_END);
   FileWrite(
      handle,
      DoubleToString(features[0][0], 6),
      DoubleToString(features[0][1], 6),
      DoubleToString(features[0][2], 0),
      DoubleToString(features[0][3], 0),
      DoubleToString(features[0][4], 8),
      DoubleToString(features[0][5], 6),
      DoubleToString(features[0][6], 6),
      DoubleToString(features[0][7], 6),
      DoubleToString(features[0][8], 0),
      DoubleToString(features[0][9], 0)
   );
   FileClose(handle);
}
