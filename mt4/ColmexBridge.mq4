//+------------------------------------------------------------------+
//|  ColmexBridge.mq4                                                 |
//|  Bridge between the trading bot (Python) and Colmex Pro's MT4.     |
//+------------------------------------------------------------------+
//
//  Colmex Pro publishes no programmatic API. MT4 is the only automation-capable
//  platform it offers, and MT4 runs MQL4, not Python. This EA is the MT4 half
//  of the bridge; broker_colmex.py is the Python half.
//
//  INSTALL
//    1. MT4 -> File -> Open Data Folder
//    2. copy this file into MQL4\Experts\
//    3. MT4 -> Navigator -> Expert Advisors -> right-click -> Refresh
//    4. drag ColmexBridge onto ANY chart (the chart symbol is irrelevant —
//       the EA works on whatever symbols the bot names)
//    5. in the dialog tick "Allow live trading", and enable AutoTrading on the
//       toolbar. DLL imports are NOT required.
//    6. set COLMEX_MT4_FILES_DIR in .env to that data folder's MQL4\Files
//
//  PROTOCOL  (all files live in MQL4\Files, MT4's sandbox)
//    colmex_state.json      written by this EA every timer tick
//    colmex_req_<id>.json   written by Python, one order request
//    colmex_res_<id>.json   written by this EA, that order's outcome
//
//  SHARES, NOT LOTS
//    MT4 sizes orders in lots; the bot thinks in shares, like it does on IBKR.
//    One lot is MarketInfo(sym, MODE_LOTSIZE) units — often 1 for equity
//    instruments but sometimes 100. This EA converts in both directions so the
//    Python side never has to know, and refuses any order whose share count
//    cannot be expressed in the symbol's lot step rather than silently
//    rounding to a different size than the bot intended.
//
//  The EA never decides anything. It reports state and executes explicit
//  instructions. Every entry/exit decision, and every safety guard, stays in
//  the bot where it can be tested.
//
#property copyright "Trading bot Colmex bridge"
#property version   "1.00"
#property strict

input int    PollMillis      = 1000;   // how often to refresh state / check requests
input int    MaxSlippage     = 30;     // points
input int    MagicNumber     = 20260831;
input string SymbolsToReport = "";     // comma-separated MT4 symbols for get_asset()

string STATE_FILE = "colmex_state.json";
string REQ_PREFIX = "colmex_req_";
string RES_PREFIX = "colmex_res_";

//+------------------------------------------------------------------+
int OnInit()
  {
   if(!IsTradeAllowed())
      Print("ColmexBridge WARNING: AutoTrading is off — state will be reported ",
            "but orders will fail until it is enabled.");
   EventSetMillisecondTimer(PollMillis);
   Print("ColmexBridge started. Files dir: ", TerminalInfoString(TERMINAL_DATA_PATH), "\\MQL4\\Files");
   WriteState();
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   // Leave the last state file in place but stop refreshing it. Python checks
   // the `ts` field and treats anything stale as a dead bridge, so an EA that
   // stops is detected rather than believed.
   Print("ColmexBridge stopped (reason ", reason, ")");
  }

void OnTimer()
  {
   WriteState();
   ProcessRequests();
  }

//+------------------------------------------------------------------+
//| JSON helpers — MQL4 has no JSON library, and the payloads here are |
//| small and fixed, so they are built by hand.                        |
//+------------------------------------------------------------------+
string JsonEscape(string s)
  {
   string out = "";
   for(int i = 0; i < StringLen(s); i++)
     {
      ushort c = StringGetCharacter(s, i);
      if(c == '"' || c == '\\')
        { out += "\\"; out += ShortToString(c); }
      else if(c == '\n') out += "\\n";
      else if(c == '\r') out += "\\r";
      else if(c == '\t') out += "\\t";
      else if(c < 32)    out += " ";
      else               out += ShortToString(c);
     }
   return(out);
  }

string JsonNum(double v, int digits)
  {
   return(DoubleToString(v, digits));
  }

// Minimal string-value reader. The request files are written by our own Python
// side with a known flat shape, so a full parser would be dead weight.
string JsonGetString(string json, string key)
  {
   string needle = "\"" + key + "\"";
   int k = StringFind(json, needle);
   if(k < 0) return("");
   int colon = StringFind(json, ":", k + StringLen(needle));
   if(colon < 0) return("");
   int q1 = StringFind(json, "\"", colon);
   if(q1 < 0) return("");
   int q2 = StringFind(json, "\"", q1 + 1);
   if(q2 < 0) return("");
   return(StringSubstr(json, q1 + 1, q2 - q1 - 1));
  }

double JsonGetNumber(string json, string key)
  {
   string needle = "\"" + key + "\"";
   int k = StringFind(json, needle);
   if(k < 0) return(0);
   int colon = StringFind(json, ":", k + StringLen(needle));
   if(colon < 0) return(0);
   int i = colon + 1;
   string num = "";
   while(i < StringLen(json))
     {
      ushort c = StringGetCharacter(json, i);
      if(c == ' ') { i++; continue; }
      if((c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+')
        { num += ShortToString(c); i++; }
      else break;
     }
   return(StringToDouble(num));
  }

//+------------------------------------------------------------------+
//| Lot <-> share conversion                                          |
//+------------------------------------------------------------------+
double LotSizeOf(string sym)
  {
   double ls = MarketInfo(sym, MODE_LOTSIZE);
   if(ls <= 0) ls = 1.0;      // defensive: never divide by zero below
   return(ls);
  }

double SharesToLots(string sym, double shares)
  {
   return(shares / LotSizeOf(sym));
  }

double LotsToShares(string sym, double lots)
  {
   return(lots * LotSizeOf(sym));
  }

// Returns "" when the lot count is valid, or an explanation when it is not.
// Rejecting is deliberate: quietly clamping to MODE_MINLOT or rounding to
// MODE_LOTSTEP would fill a different size than the bot sized and risked,
// and the bot would record the size it asked for, not the one it got.
string ValidateLots(string sym, double lots)
  {
   double minLot  = MarketInfo(sym, MODE_MINLOT);
   double maxLot  = MarketInfo(sym, MODE_MAXLOT);
   double lotStep = MarketInfo(sym, MODE_LOTSTEP);

   if(minLot > 0 && lots < minLot - 0.0000001)
      return(StringFormat("%.8g lots is below the %s minimum of %.8g", lots, sym, minLot));
   if(maxLot > 0 && lots > maxLot + 0.0000001)
      return(StringFormat("%.8g lots is above the %s maximum of %.8g", lots, sym, maxLot));
   if(lotStep > 0)
     {
      double steps = lots / lotStep;
      if(MathAbs(steps - MathRound(steps)) > 0.0001)
         return(StringFormat("%.8g lots is not a multiple of the %s lot step %.8g",
                             lots, sym, lotStep));
     }
   return("");
  }

//+------------------------------------------------------------------+
//| State reporting                                                   |
//+------------------------------------------------------------------+

// Aggregate every open ticket for a symbol into one net position, so Python
// sees the same "one row per ticker" shape IBKR reports. MT4 keeps each entry
// as a separate ticket, so a symbol entered twice would otherwise appear twice.
// Gross value of all open positions, filled in by BuildPositions() and read by
// WriteState(). Mirrors IBKR's GrossPositionValue, which is what the bot's
// portfolio_value field means everywhere else.
double g_grossPositionValue = 0;

void BuildPositions(string &json)
  {
   g_grossPositionValue = 0;
   string syms[];
   double netShares[];
   double costSum[];      // shares * entry price, for a weighted average
   int    n = 0;

   ArrayResize(syms, 0);
   ArrayResize(netShares, 0);
   ArrayResize(costSum, 0);

   for(int i = 0; i < OrdersTotal(); i++)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderType() != OP_BUY && OrderType() != OP_SELL) continue;   // skip pending

      string sym    = OrderSymbol();
      double shares = LotsToShares(sym, OrderLots());
      if(OrderType() == OP_SELL) shares = -shares;   // sign carries direction

      int idx = -1;
      for(int j = 0; j < n; j++) if(syms[j] == sym) { idx = j; break; }
      if(idx < 0)
        {
         idx = n; n++;
         ArrayResize(syms, n); ArrayResize(netShares, n); ArrayResize(costSum, n);
         syms[idx] = sym; netShares[idx] = 0; costSum[idx] = 0;
        }
      netShares[idx] += shares;
      costSum[idx]   += shares * OrderOpenPrice();
     }

   json += "\"positions\":[";
   bool first = true;
   for(int p = 0; p < n; p++)
     {
      if(MathAbs(netShares[p]) < 0.0000001) continue;   // fully hedged out
      string sym = syms[p];
      double avg = costSum[p] / netShares[p];
      double cur = (netShares[p] > 0) ? MarketInfo(sym, MODE_BID) : MarketInfo(sym, MODE_ASK);
      if(cur <= 0) cur = avg;    // no quote right now — do not report a zero price
      int dg = (int)MarketInfo(sym, MODE_DIGITS);
      if(dg <= 0) dg = 2;
      g_grossPositionValue += MathAbs(netShares[p]) * cur;

      if(!first) json += ",";
      first = false;
      json += "{\"ticker\":\"" + JsonEscape(sym) + "\"";
      json += ",\"qty\":"               + JsonNum(netShares[p], 8);
      json += ",\"avg_entry_price\":"   + JsonNum(avg, dg);
      json += ",\"current_price\":"     + JsonNum(cur, dg);
      json += "}";
     }
   json += "],";
  }

// Sell orders that are placed but not yet filled. The bot subtracts these from
// the held quantity before sizing an exit — without it, repeated exit attempts
// while an order is still working sell the same shares twice and go short.
void BuildPendingSells(string &json)
  {
   json += "\"pending_sells\":{";
   bool first = true;
   for(int i = 0; i < OrdersTotal(); i++)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      int t = OrderType();
      if(t != OP_SELLLIMIT && t != OP_SELLSTOP) continue;
      if(!first) json += ",";
      first = false;
      json += "\"" + JsonEscape(OrderSymbol()) + "\":"
            + JsonNum(LotsToShares(OrderSymbol(), OrderLots()), 8);
     }
   json += "},";
  }

// Symbols the bot may ask about via get_asset(). Only those named in the
// SymbolsToReport input are included — walking the whole Market Watch on every
// tick would bloat the state file for no benefit.
void BuildSymbols(string &json)
  {
   json += "\"symbols\":{";
   if(SymbolsToReport != "")
     {
      string parts[];
      int cnt = StringSplit(SymbolsToReport, ',', parts);
      bool first = true;
      for(int i = 0; i < cnt; i++)
        {
         string sym = parts[i];
         StringTrimLeft(sym); StringTrimRight(sym);
         if(sym == "") continue;
         bool tradable = (MarketInfo(sym, MODE_BID) > 0);
         if(!first) json += ",";
         first = false;
         json += "\"" + JsonEscape(sym) + "\":{\"tradable\":"
               + (tradable ? "true" : "false")
               + ",\"description\":\"" + JsonEscape(sym) + "\"}";
        }
     }
   json += "},";
  }

void WriteState()
  {
   // Positions are built first: BuildPositions() computes g_grossPositionValue,
   // which the account block below reports.
   string positionsJson = "";
   BuildPositions(positionsJson);

   string json = "{";
   // Seconds since the Unix epoch, in UTC. TimeCurrent() is server time, whose
   // offset varies by broker; TimeGMT() keeps this comparable with Python's
   // time.time() so the staleness check means what it says.
   json += "\"ts\":" + IntegerToString((int)TimeGMT()) + ",";

   json += "\"account\":{";
   json += "\"equity\":"           + JsonNum(AccountEquity(), 2);
   json += ",\"buying_power\":"    + JsonNum(AccountFreeMargin(), 2);
   json += ",\"cash\":"            + JsonNum(AccountBalance(), 2);
   json += ",\"portfolio_value\":" + JsonNum(g_grossPositionValue, 2);
   json += ",\"currency\":\""      + JsonEscape(AccountCurrency()) + "\"";
   json += ",\"account\":\""       + JsonEscape(IntegerToString(AccountNumber())) + "\"";
   json += ",\"is_demo\":"         + (IsDemo() ? "true" : "false");
   json += "},";

   json += positionsJson;
   BuildPendingSells(json);
   BuildSymbols(json);

   json += "\"trade_allowed\":" + (IsTradeAllowed() ? "true" : "false") + ",";
   json += "\"market_open\":"   + (IsTradeAllowed() && !IsTradeContextBusy() ? "true" : "false");
   json += "}";

   // Write to a temp name then rename, so Python never reads a half-written
   // file. FileMove with FILE_REWRITE replaces atomically enough for a reader
   // that retries once, which broker_colmex.py does.
   string tmp = STATE_FILE + ".tmp";
   int h = FileOpen(tmp, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE)
     {
      Print("ColmexBridge: cannot write state (", GetLastError(), ")");
      return;
     }
   FileWriteString(h, json);
   FileClose(h);
   FileDelete(STATE_FILE);
   FileMove(tmp, 0, STATE_FILE, 0);
  }

//+------------------------------------------------------------------+
//| Order execution                                                   |
//+------------------------------------------------------------------+
void WriteResponse(string id, bool ok, string errorMsg, int ticket,
                   double shares, double price, string status)
  {
   string json = "{\"id\":\"" + JsonEscape(id) + "\"";
   json += ",\"ok\":" + (ok ? "true" : "false");
   if(ok)
     {
      json += ",\"order_id\":\"" + IntegerToString(ticket) + "\"";
      json += ",\"shares\":"     + JsonNum(shares, 8);
      json += ",\"price\":"      + JsonNum(price, 5);
      json += ",\"status\":\""   + JsonEscape(status) + "\"";
     }
   else
      json += ",\"error\":\"" + JsonEscape(errorMsg) + "\"";
   json += "}";

   string name = RES_PREFIX + id + ".json";
   int h = FileOpen(name, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE)
     {
      Print("ColmexBridge: cannot write response for ", id, " (", GetLastError(), ")");
      return;
     }
   FileWriteString(h, json);
   FileClose(h);
  }

// Close existing long tickets for `symbol` up to `shares`. A SELL from the bot
// always means "exit this long", never "open a short" — the bot is long-only,
// and OrderSend(OP_SELL) on MT4 would open a fresh short ticket alongside the
// long rather than closing it.
void CloseLong(string id, string symbol, double shares)
  {
   double remaining = shares;
   int    lastTicket = 0;
   double lastPrice  = 0;
   double closed     = 0;

   for(int i = OrdersTotal() - 1; i >= 0 && remaining > 0.0000001; i--)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderSymbol() != symbol) continue;
      if(OrderType() != OP_BUY)   continue;

      double ticketShares = LotsToShares(symbol, OrderLots());
      double takeShares   = MathMin(ticketShares, remaining);
      double takeLots     = SharesToLots(symbol, takeShares);

      string bad = ValidateLots(symbol, takeLots);
      if(bad != "")
        {
         WriteResponse(id, false, "partial close rejected: " + bad, 0, 0, 0, "");
         return;
        }

      double bid = MarketInfo(symbol, MODE_BID);
      if(!OrderClose(OrderTicket(), takeLots, bid, MaxSlippage))
        {
         WriteResponse(id, false,
                       StringFormat("OrderClose failed on ticket %d: error %d",
                                    OrderTicket(), GetLastError()),
                       0, 0, 0, "");
         return;
        }
      lastTicket = OrderTicket();
      lastPrice  = bid;
      closed    += takeShares;
      remaining -= takeShares;
     }

   if(closed <= 0)
     {
      WriteResponse(id, false, "no open long position to close for " + symbol, 0, 0, 0, "");
      return;
     }
   // A short fill is still a real fill; report what actually closed so the bot
   // records the true size rather than the size it asked for.
   WriteResponse(id, true, "", lastTicket, closed, lastPrice, "filled");
  }

void HandleRequest(string filename)
  {
   int h = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE) return;
   string body = "";
   while(!FileIsEnding(h)) body += FileReadString(h);
   FileClose(h);
   FileDelete(filename);

   string id     = JsonGetString(body, "id");
   string action = JsonGetString(body, "action");
   string symbol = JsonGetString(body, "symbol");
   double shares = JsonGetNumber(body, "shares");

   if(id == "") { Print("ColmexBridge: request with no id, ignored"); return; }

   if(!IsTradeAllowed())
     { WriteResponse(id, false, "AutoTrading is disabled in MT4", 0, 0, 0, ""); return; }
   if(symbol == "" || shares <= 0)
     { WriteResponse(id, false, "malformed request: missing symbol or shares", 0, 0, 0, ""); return; }
   if(MarketInfo(symbol, MODE_BID) <= 0)
     {
      WriteResponse(id, false,
                    "symbol '" + symbol + "' is not quoted by Colmex — check "
                    "COLMEX_SYMBOL_SUFFIX / COLMEX_SYMBOL_MAP", 0, 0, 0, "");
      return;
     }

   if(action == "SELL") { CloseLong(id, symbol, shares); return; }

   if(action != "BUY")
     { WriteResponse(id, false, "unsupported action '" + action + "'", 0, 0, 0, ""); return; }

   double lots = SharesToLots(symbol, shares);
   string bad  = ValidateLots(symbol, lots);
   if(bad != "") { WriteResponse(id, false, bad, 0, 0, 0, ""); return; }

   double ask = MarketInfo(symbol, MODE_ASK);
   int ticket = OrderSend(symbol, OP_BUY, lots, ask, MaxSlippage, 0, 0,
                          "bot", MagicNumber, 0, clrNONE);
   if(ticket < 0)
     {
      WriteResponse(id, false,
                    StringFormat("OrderSend failed: error %d", GetLastError()),
                    0, 0, 0, "");
      return;
     }
   // Report the actual fill price, not the requested one — slippage is real and
   // the bot's P&L is only as honest as the entry price it records.
   double fill = ask;
   if(OrderSelect(ticket, SELECT_BY_TICKET)) fill = OrderOpenPrice();
   WriteResponse(id, true, "", ticket, shares, fill, "filled");
  }

void ProcessRequests()
  {
   string filename;
   long   handle = FileFindFirst(REQ_PREFIX + "*.json", filename);
   if(handle == INVALID_HANDLE) return;
   do
     {
      HandleRequest(filename);
     }
   while(FileFindNext(handle, filename));
   FileFindClose(handle);
  }
//+------------------------------------------------------------------+
