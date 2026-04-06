#!/usr/bin/env python3
"""
AlgoScalp Pro — Intelligent Scalping Bot for Indian Markets
=============================================================
5 Professional Scalping Strategies (researched for NSE/BSE):

  1. VWAP Pullback        — Institutional level scalp (best for Nifty)
  2. EMA Ribbon + RSI     — Momentum entry scalp (5 EMA stack)
  3. Bollinger Squeeze    — Volatility breakout scalp
  4. MACD Zero Cross      — Trend momentum scalp
  5. SMART COMBO ★        — All 4 combined = highest win rate

Signal Scoring System:
  Each indicator gives +1 (bullish) or -1 (bearish) score.
  Trade only fires when combined score >= threshold (configurable).
  This is what separates professional algo from basic crossover.

Risk Management (Built-in):
  • ATR-based stop loss (1.5x ATR)
  • Trailing stop loss
  • Max daily loss limit (auto-stop trading for the day)
  • Max trades per day
  • Time filter (only trade 9:20–11:30 and 13:30–15:00 IST)
  • Position sizing based on risk % of capital

Charges: Full Groww NSE delivery + STCG/LTCG tax (Budget 2024)
Paper Trading: Auto-executes on signals every N seconds
History: All sessions saved to algoscalp_history.json

Run: python algoscalp_pro.py
Opens: http://localhost:5052
"""

from flask import Flask, render_template_string, request, jsonify
import yfinance as yf
import pandas as pd, numpy as np, json, os
from pathlib import Path
from datetime import datetime, time as dtime

app = Flask(__name__)
HIST_FILE = Path("algoscalp_history.json")

# ════════════════════════════════════════════════
#  DATA PERSISTENCE
# ════════════════════════════════════════════════
def load_hist():
    if HIST_FILE.exists():
        try:
            with open(HIST_FILE) as f: return json.load(f)
        except: pass
    return {"sessions": []}

def save_hist(d):
    with open(HIST_FILE, "w") as f: json.dump(d, f, indent=2, default=str)

def send_telegram(token, chat_id, msg):
    try:
        import urllib.request
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        pay = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}).encode()
        req = urllib.request.Request(url, data=pay, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5); return True
    except: return False

# ════════════════════════════════════════════════
#  INDICATORS
# ════════════════════════════════════════════════
def ind_ema(s, n):
    r = [None]*len(s); k = 2/(n+1)
    for i in range(len(s)):
        if i < n-1: continue
        if i == n-1: r[i] = float(np.mean(s[i-n+1:i+1]))
        elif r[i-1] is not None: r[i] = s[i]*k + r[i-1]*(1-k)
    return r

def ind_sma(s, n):
    return [None if i < n-1 else float(np.mean(s[i-n+1:i+1])) for i in range(len(s))]

def ind_rsi(s, n=14):
    r = [None]*len(s)
    if len(s) < n+1: return r
    g = [max(s[i]-s[i-1], 0) for i in range(1, len(s))]
    l = [max(s[i-1]-s[i], 0) for i in range(1, len(s))]
    ag = np.mean(g[:n]); al = np.mean(l[:n])
    for i in range(n, len(s)):
        if i > n: ag=(ag*(n-1)+g[i-1])/n; al=(al*(n-1)+l[i-1])/n
        r[i] = 100-100/(1+ag/al) if al != 0 else 100
    return r

def ind_vwap(H, L, C, V):
    r = []; tv = 0; vv = 0
    for i in range(len(C)):
        tp = (H[i]+L[i]+C[i])/3; v = V[i] if V[i]>0 else 1
        tv += tp*v; vv += v; r.append(tv/vv)
    return r

def ind_atr(H, L, C, n=14):
    r = [None]*len(C); trs = []
    for i in range(1, len(C)):
        trs.append(max(H[i]-L[i], abs(H[i]-C[i-1]), abs(L[i]-C[i-1])))
    for i in range(n, len(C)): r[i] = float(np.mean(trs[i-n:i]))
    return r

def ind_bollinger(C, n=20, k=2.0):
    m = ind_sma(C, n); U=[None]*len(C); L=[None]*len(C)
    for i in range(n-1, len(C)):
        sd = float(np.std(C[i-n+1:i+1])); U[i]=m[i]+k*sd; L[i]=m[i]-k*sd
    return U, m, L

def ind_macd(C, fast=12, slow=26, sig=9):
    ef=ind_ema(C,fast); es=ind_ema(C,slow)
    ml=[ef[i]-es[i] if ef[i] and es[i] else None for i in range(len(C))]
    ml_f=[v if v is not None else 0 for v in ml]
    sl=ind_ema(ml_f, sig)
    hist=[ml[i]-sl[i] if ml[i] and sl[i] else None for i in range(len(C))]
    return ml, sl, hist

def ind_bandwidth(upper, mid, lower):
    """Bollinger Bandwidth — squeeze = low volatility before big move"""
    bw = [None]*len(mid)
    for i in range(len(mid)):
        if upper[i] and lower[i] and mid[i] and mid[i]!=0:
            bw[i] = (upper[i]-lower[i])/mid[i]*100
    return bw

def co_up(a, b, i):
    if i < 1: return False
    if any(x is None for x in [a[i-1],b[i-1],a[i],b[i]]): return False
    return a[i-1] <= b[i-1] and a[i] > b[i]
def co_dn(a, b, i): return co_up(b, a, i)

# ════════════════════════════════════════════════
#  SMART SIGNAL ENGINE
#  Each sub-strategy scores +1 (BUY) or -1 (SELL)
#  Final signal fires only when score >= threshold
# ════════════════════════════════════════════════
def compute_smart_signals(data, strategy, params):
    C = data['Close'].values.tolist()
    H = data['High'].values.tolist()
    L = data['Low'].values.tolist()
    V = data['Volume'].values.tolist() if 'Volume' in data.columns else [1e6]*len(C)
    n = len(C)

    # ── Core indicators ──────────────────────────
    e5  = ind_ema(C, 5)
    e8  = ind_ema(C, 8)
    e13 = ind_ema(C, 13)
    e21 = ind_ema(C, 21)
    e50 = ind_ema(C, 50)
    rv9  = ind_rsi(C, 9)
    rv14 = ind_rsi(C, 14)
    vw   = ind_vwap(H, L, C, V)
    at14 = ind_atr(H, L, C, 14)
    bbu, bbm, bbl = ind_bollinger(C, 20, 2.0)
    ml, sl, mh   = ind_macd(C, 12, 26, 9)
    bw           = ind_bandwidth(bbu, bbm, bbl)

    # ── Per-bar scoring ───────────────────────────
    scores   = [0]*n        # combined score
    signals  = [None]*n     # BUY / SELL / None
    stops    = [None]*n
    targets  = [None]*n
    sub_sigs = [{}]*n       # breakdown per indicator

    threshold = int(params.get('threshold', 3))  # need N confirmations

    for i in range(50, n):
        score = 0
        subs  = {}

        # ── 1. VWAP: price above VWAP = bullish context ──
        if vw[i] and C[i]:
            if C[i] > vw[i]: score += 1; subs['VWAP'] = 'BULL'
            else:             score -= 1; subs['VWAP'] = 'BEAR'

        # ── 2. EMA Ribbon: 5>8>13 = strong uptrend ──
        if all(x is not None for x in [e5[i],e8[i],e13[i],e21[i]]):
            if e5[i] > e8[i] > e13[i] > e21[i]:  score += 2; subs['EMA'] = 'BULL★★'
            elif e5[i] < e8[i] < e13[i] < e21[i]: score -= 2; subs['EMA'] = 'BEAR★★'
            elif e5[i] > e8[i]:                    score += 1; subs['EMA'] = 'BULL'
            elif e5[i] < e8[i]:                    score -= 1; subs['EMA'] = 'BEAR'

        # ── 3. RSI momentum ──
        if rv9[i]:
            if rv9[i] > 55 and rv9[i] < 80:   score += 1; subs['RSI'] = 'BULL'
            elif rv9[i] < 45 and rv9[i] > 20:  score -= 1; subs['RSI'] = 'BEAR'
            elif rv9[i] >= 80:                  score -= 1; subs['RSI'] = 'OB⚠'
            elif rv9[i] <= 20:                  score += 1; subs['RSI'] = 'OS⚠'

        # ── 4. MACD momentum ──
        if ml[i] and sl[i]:
            if ml[i] > sl[i] and ml[i] > 0:   score += 1; subs['MACD'] = 'BULL'
            elif ml[i] < sl[i] and ml[i] < 0:  score -= 1; subs['MACD'] = 'BEAR'
            elif ml[i] > sl[i]:                 score += 0; subs['MACD'] = 'WEAK+'
            else:                               score -= 0; subs['MACD'] = 'WEAK-'

        # ── 5. Bollinger position ──
        if bbu[i] and bbl[i] and bbm[i]:
            bb_pos = (C[i] - bbl[i]) / (bbu[i] - bbl[i]) * 100 if (bbu[i]-bbl[i]) > 0 else 50
            if bb_pos > 60 and bb_pos < 85:    score += 1; subs['BB'] = 'UPPER'
            elif bb_pos < 40 and bb_pos > 15:   score -= 1; subs['BB'] = 'LOWER'
            elif bb_pos >= 85:                  score -= 1; subs['BB'] = 'OVERBOUGHT'
            elif bb_pos <= 15:                  score += 1; subs['BB'] = 'OVERSOLD'

        # ── 6. Bollinger Squeeze (low BW = explosion coming) ──
        if bw[i] and i >= 5:
            bw_avg = np.mean([b for b in bw[i-5:i] if b is not None] or [bw[i]])
            if bw[i] < bw_avg * 0.8:           subs['SQUEEZE'] = '⚡READY'; # neutral, just flag

        scores[i] = score
        sub_sigs[i] = subs

        # ── Fire signal based on strategy ────────
        if strategy == 'vwap_pullback':
            # Pure VWAP: price pulls back to VWAP from above, RSI 40-60
            above_vwap = C[i] > (vw[i] or 0)
            pullback   = abs(C[i] - (vw[i] or C[i])) / C[i] < 0.003  # within 0.3% of VWAP
            rsi_ok     = rv9[i] and 38 < rv9[i] < 65
            if above_vwap and pullback and rsi_ok and (e5[i] or 0) > (e21[i] or 0):
                signals[i] = 'BUY'
            elif (not above_vwap) and pullback and rsi_ok:
                signals[i] = 'SELL'

        elif strategy == 'ema_ribbon':
            # EMA 5-8-13 ribbon: all aligned + RSI confirmation
            if all(x is not None for x in [e5[i],e8[i],e13[i]]):
                if co_up(e5,e8,i) and e8[i]>(e13[i] or 0) and rv9[i] and rv9[i]>50:
                    signals[i] = 'BUY'
                elif co_dn(e5,e8,i) and e8[i]<(e13[i] or 0) and rv9[i] and rv9[i]<50:
                    signals[i] = 'SELL'

        elif strategy == 'bb_squeeze':
            # Bollinger squeeze breakout: price breaks out of tight band
            if bw[i] and i >= 10:
                bw_avg = np.mean([b for b in bw[i-10:i] if b is not None] or [bw[i]])
                was_squeeze = bw[i-1] and bw[i-1] < bw_avg * 0.85
                if was_squeeze:
                    if C[i] > (bbu[i] or C[i]) and rv14[i] and rv14[i] > 50:
                        signals[i] = 'BUY'
                    elif C[i] < (bbl[i] or C[i]) and rv14[i] and rv14[i] < 50:
                        signals[i] = 'SELL'

        elif strategy == 'macd_zero':
            # MACD crosses zero line with EMA confirmation
            if ml[i] and ml[i-1]:
                if ml[i-1] < 0 < ml[i] and C[i] > (e21[i] or 0):
                    signals[i] = 'BUY'
                elif ml[i-1] > 0 > ml[i] and C[i] < (e21[i] or 0):
                    signals[i] = 'SELL'

        elif strategy == 'smart_combo':
            # SMART: fire only when combined score reaches threshold
            if score >= threshold:
                signals[i] = 'BUY'
            elif score <= -threshold:
                signals[i] = 'SELL'

        # ── ATR-based stop & target ───────────────
        if signals[i] == 'BUY' and at14[i]:
            stops[i]   = round(C[i] - 1.5 * at14[i], 2)
            targets[i] = round(C[i] + 2.0 * at14[i], 2)   # 1:1.33 R:R
        elif signals[i] == 'SELL' and at14[i]:
            stops[i]   = round(C[i] + 1.5 * at14[i], 2)
            targets[i] = round(C[i] - 2.0 * at14[i], 2)

    return signals, scores, sub_sigs, stops, targets, {
        'e5': e5, 'e8': e8, 'e13': e13, 'e21': e21,
        'vwap': vw, 'rsi': rv9, 'bbu': bbu, 'bbm': bbm, 'bbl': bbl,
        'macd': ml, 'macd_sig': sl, 'macd_hist': mh, 'bw': bw, 'atr': at14
    }

# ════════════════════════════════════════════════
#  CHARGES ENGINE
# ════════════════════════════════════════════════
def calc_charges(buy_v, sell_v, buy_dt=None, sell_dt=None):
    bb=max(min(0.001*buy_v,20),5); sb=max(min(0.001*sell_v,20),5); brok=bb+sb
    stt=0.001*buy_v+0.001*sell_v; exc=0.0000325*(buy_v+sell_v)
    sebi=0.000001*(buy_v+sell_v); stamp=0.00015*buy_v
    gst=0.18*(brok+exc+sebi); dp=20; total=brok+stt+exc+sebi+stamp+gst+dp
    gross=sell_v-buy_v
    days=(sell_dt-buy_dt).days if buy_dt and sell_dt else 0
    cg_tax=0; cg_type=''; cg_rate=0
    if gross > 0:
        if days < 365: cg_type='STCG'; cg_rate=0.20; cg_tax=gross*0.20*1.04
        else:          cg_type='LTCG'; cg_rate=0.125; cg_tax=max(0,gross-125000)*0.125*1.04
    return dict(brok=round(brok,2),stt=round(stt,2),exc=round(exc,4),
                sebi=round(sebi,4),stamp=round(stamp,2),gst=round(gst,2),dp=20,
                total=round(total,2),gross=round(gross,2),cg_type=cg_type,
                cg_rate_pct=round(cg_rate*100,1),cg_tax=round(cg_tax,2),
                net=round(gross-total-cg_tax,2),days=days)

def clean(lst):
    if lst is None: return None
    return [None if (v is None or (isinstance(v,float) and np.isnan(v))) else round(float(v),4) for v in lst]

# ════════════════════════════════════════════════
#  HTML
# ════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlgoScalp Pro</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#04080f;--sf:#0b1623;--sf2:#111f30;--sf3:#080f1a;
  --bd:#1a2d45;--ac:#00e5ff;--ac2:#ff6b35;--gn:#00e676;--rd:#ff3d57;
  --tx:#dde6f0;--mt:#3d5a78;--gd:#ffd700;--pp:#a78bfa;--or:#fb923c;
  --score-pos:#00e676;--score-neg:#ff3d57;--score-neu:#4e6580;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--tx);font-family:'Syne',sans-serif;height:100vh;overflow:hidden;}
/* TOPBAR */
.top{display:flex;align-items:center;justify-content:space-between;padding:8px 20px;
  border-bottom:1px solid var(--bd);background:rgba(4,8,15,.98);position:relative;z-index:100;}
.logo{display:flex;align-items:center;gap:8px;}
.logo-i{width:28px;height:28px;background:linear-gradient(135deg,var(--ac),var(--ac2));
  border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:13px;}
.logo-t{font-size:15px;font-weight:800;}
.logo-t span{color:var(--ac);}
.logo-sub{font-size:8px;font-family:'Space Mono',monospace;color:var(--mt);margin-top:1px;}
.top-pills{display:flex;gap:6px;}
.pill{font-family:'Space Mono',monospace;font-size:8px;padding:3px 8px;border-radius:20px;border:1px solid var(--bd);color:var(--mt);}
.pill.live{border-color:var(--gn);color:var(--gn);}
.pill.live::before{content:'● ';}
.pill.warn{border-color:var(--or);color:var(--or);}
/* LAYOUT */
.app{display:grid;grid-template-columns:260px 1fr 200px;height:calc(100vh - 46px);}
/* LEFT SIDEBAR */
.lsb{background:var(--sf);border-right:1px solid var(--bd);overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px;}
.sl{font-size:7px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--mt);margin-bottom:6px;}
/* Strategy selector */
.stlist{display:flex;flex-direction:column;gap:4px;}
.stitem{background:var(--sf2);border:1px solid var(--bd);border-radius:7px;padding:8px 10px;cursor:pointer;transition:all .15s;}
.stitem:hover{border-color:var(--ac);}
.stitem.on{border-color:var(--ac);background:rgba(0,229,255,.07);}
.stitem .sn{font-size:10px;font-weight:700;color:var(--ac);}
.stitem .sd{font-size:8px;color:var(--mt);margin-top:2px;line-height:1.4;}
.stitem .sc-badge{float:right;font-size:7px;font-family:'Space Mono',monospace;padding:1px 5px;border-radius:3px;background:rgba(0,229,255,.12);color:var(--ac);}
/* Inputs */
.ig{display:flex;flex-direction:column;gap:3px;}
.ig label{font-size:8px;color:var(--mt);font-family:'Space Mono',monospace;}
.ig input,.ig select{background:var(--sf2);border:1px solid var(--bd);color:var(--tx);
  padding:6px 8px;border-radius:5px;font-size:11px;font-family:'Space Mono',monospace;width:100%;}
.ig input:focus,.ig select:focus{outline:none;border-color:var(--ac);}
.ig select option{background:var(--sf2);}
.r2{display:grid;grid-template-columns:1fr 1fr;gap:5px;}
.dv{height:1px;background:var(--bd);}
/* Score threshold slider */
.thresh-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;}
.thresh-val{font-family:'Space Mono',monospace;font-size:11px;color:var(--ac);font-weight:700;}
input[type=range]{width:100%;accent-color:var(--ac);}
/* Risk inputs */
.risk-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;}
/* Buttons */
.bmain{border:none;border-radius:7px;padding:9px;font-size:11px;font-weight:800;
  font-family:'Syne',sans-serif;cursor:pointer;width:100%;transition:all .15s;text-transform:uppercase;letter-spacing:.5px;}
.bgo{background:linear-gradient(135deg,var(--gn),#009944);color:#000;}
.bgo:hover{box-shadow:0 4px 14px rgba(0,230,118,.3);}
.bstop{background:linear-gradient(135deg,var(--rd),#bb0022);color:#fff;}
.bbt{background:linear-gradient(135deg,var(--ac),#0099bb);color:#000;}
.bbt:hover{box-shadow:0 4px 14px rgba(0,229,255,.2);}
.bmain:disabled{opacity:.3;cursor:not-allowed;}
.r2-btn{display:grid;grid-template-columns:1fr 1fr;gap:5px;}
/* Stats */
.stg{display:grid;grid-template-columns:1fr 1fr;gap:4px;}
.stc{background:var(--sf2);border:1px solid var(--bd);border-radius:5px;padding:6px 8px;}
.stc .sl2{font-size:6px;color:var(--mt);letter-spacing:1px;text-transform:uppercase;font-family:'Space Mono',monospace;}
.stc .sv{font-size:12px;font-weight:800;font-family:'Space Mono',monospace;margin-top:1px;}
.pos{color:var(--gn);}.neg{color:var(--rd);}.neu{color:var(--ac);}.gld{color:var(--gd);}
.pb{height:3px;background:var(--bd);border-radius:2px;overflow:hidden;margin-top:3px;}
.pf{height:100%;background:linear-gradient(90deg,var(--gn),var(--ac));width:0%;}
.pt{font-family:'Space Mono',monospace;font-size:8px;color:var(--mt);margin-top:2px;}
/* TG box */
.tgbox{background:var(--sf2);border:1px solid var(--bd);border-radius:6px;padding:8px;}
.tgsw{position:relative;width:30px;height:16px;cursor:pointer;display:inline-block;}
.tgsw input{opacity:0;width:0;height:0;}
.tgtr{position:absolute;inset:0;background:var(--bd);border-radius:16px;transition:.3s;}
.tgsw input:checked+.tgtr{background:var(--gn);}
.tgtr::before{content:'';position:absolute;width:12px;height:12px;background:#fff;border-radius:50%;top:2px;left:2px;transition:.3s;}
.tgsw input:checked+.tgtr::before{transform:translateX(14px);}
.err{background:rgba(255,61,87,.1);border:1px solid var(--rd);border-radius:5px;padding:6px 9px;font-size:9px;color:var(--rd);font-family:'Space Mono',monospace;display:none;}
/* MAIN */
.main{display:flex;flex-direction:column;overflow:hidden;}
.cw{flex:1;position:relative;min-height:0;}
#mc{width:100%;height:100%;}
/* Overlays */
.ovl{position:absolute;inset:0;background:rgba(4,8,15,.94);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;z-index:200;}
.ovl.hid{display:none;}
.spin{width:36px;height:36px;border:3px solid var(--bd);border-top-color:var(--gn);border-radius:50%;animation:sp .7s linear infinite;}
@keyframes sp{to{transform:rotate(360deg)}}
.ovlt{font-family:'Space Mono',monospace;font-size:11px;color:var(--mt);}
/* Price ticker */
.pxti{position:absolute;top:8px;left:8px;background:rgba(11,22,35,.95);border:1px solid var(--bd);border-radius:7px;padding:6px 10px;z-index:50;display:none;}
.ptsy{font-size:7px;color:var(--mt);letter-spacing:2px;text-transform:uppercase;font-family:'Space Mono',monospace;}
.ptpx{font-size:16px;font-weight:800;font-family:'Space Mono',monospace;}
.ptch{font-size:8px;font-family:'Space Mono',monospace;}
/* Score meter (center top) */
.score-meter{position:absolute;top:8px;left:50%;transform:translateX(-50%);
  background:rgba(11,22,35,.95);border:1px solid var(--bd);border-radius:8px;
  padding:6px 14px;z-index:50;display:none;min-width:220px;}
.sm-row{display:flex;justify-content:space-between;align-items:center;gap:8px;}
.sm-lbl{font-size:7px;font-family:'Space Mono',monospace;color:var(--mt);letter-spacing:1px;}
.sm-val{font-size:11px;font-weight:800;font-family:'Space Mono',monospace;}
.sm-bars{display:flex;gap:3px;align-items:center;margin-top:4px;}
.sm-bar{width:16px;height:8px;border-radius:2px;background:var(--bd);}
.sm-bar.bull{background:var(--gn);}
.sm-bar.bear{background:var(--rd);}
/* Signal flash */
.sig-flash{position:absolute;top:8px;right:8px;padding:7px 12px;border-radius:7px;
  font-family:'Space Mono',monospace;font-size:11px;font-weight:700;letter-spacing:1px;
  z-index:50;display:none;animation:flash .5s ease;}
@keyframes flash{from{transform:scale(1.15)}to{transform:scale(1)}}
.sig-flash.buy{background:rgba(0,230,118,.15);border:1px solid var(--gn);color:var(--gn);}
.sig-flash.sell{background:rgba(255,61,87,.15);border:1px solid var(--rd);color:var(--rd);}
.sig-flash.hold{background:var(--sf2);border:1px solid var(--bd);color:var(--mt);}
/* Oscillator */
.ow{height:80px;border-top:1px solid var(--bd);position:relative;background:var(--sf3);}
#oc{width:100%;height:100%;}
.ow .olbl{position:absolute;top:3px;left:8px;font-size:7px;font-family:'Space Mono',monospace;color:var(--mt);z-index:10;}
.ow.hid{display:none;}
/* Countdown */
.cdown{position:absolute;bottom:8px;left:8px;background:rgba(11,22,35,.92);border:1px solid var(--bd);border-radius:5px;padding:4px 8px;font-family:'Space Mono',monospace;font-size:8px;color:var(--mt);z-index:50;display:none;}
.cdown.on{display:block;}
/* RIGHT SIDEBAR — signal breakdown */
.rsb{background:var(--sf);border-left:1px solid var(--bd);overflow-y:auto;padding:10px;display:flex;flex-direction:column;gap:8px;}
.ind-card{background:var(--sf2);border:1px solid var(--bd);border-radius:6px;padding:8px;}
.ind-name{font-size:8px;font-weight:700;letter-spacing:1px;text-transform:uppercase;font-family:'Space Mono',monospace;color:var(--mt);margin-bottom:5px;}
.ind-val{font-size:11px;font-weight:800;font-family:'Space Mono',monospace;}
.ind-sig{font-size:9px;margin-top:2px;font-family:'Space Mono',monospace;}
.bull-sig{color:var(--gn);}
.bear-sig{color:var(--rd);}
.neu-sig{color:var(--mt);}
/* Score gauge */
.gauge{margin-top:5px;background:var(--bd);border-radius:3px;height:6px;overflow:hidden;}
.gauge-fill{height:100%;border-radius:3px;transition:width .4s,background .4s;}
/* BOTTOM — trade log */
.bot{height:185px;border-top:1px solid var(--bd);display:flex;flex-direction:column;background:var(--sf);}
.tabs{display:flex;border-bottom:1px solid var(--bd);background:var(--sf2);}
.tab{padding:5px 12px;font-size:8px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--mt);cursor:pointer;border-bottom:2px solid transparent;transition:all .2s;}
.tab.on{color:var(--ac);border-bottom-color:var(--ac);}
.tc{display:none;flex:1;overflow-y:auto;}
.tc.on{display:flex;flex-direction:column;}
table{width:100%;border-collapse:collapse;font-family:'Space Mono',monospace;font-size:8px;}
thead th{background:var(--sf2);color:var(--mt);text-align:left;padding:4px 8px;font-size:7px;letter-spacing:1px;position:sticky;top:0;border-bottom:1px solid var(--bd);}
tbody tr{border-bottom:1px solid rgba(26,45,69,.4);animation:ri .2s ease;}
@keyframes ri{from{opacity:0;transform:translateX(-4px)}to{opacity:1;transform:none}}
tbody tr:hover{background:var(--sf2);}
tbody td{padding:4px 8px;}
.tbt{color:var(--gn);font-weight:700;}.tts{color:var(--rd);font-weight:700;}
.pp{color:var(--gn);}.np{color:var(--rd);}
.totbar{display:flex;border-top:1px solid var(--bd);background:var(--sf2);}
.tbc{flex:1;padding:4px 9px;border-right:1px solid var(--bd);}
.tbc:last-child{border-right:none;}
.tbcl{font-size:6px;color:var(--mt);letter-spacing:1px;text-transform:uppercase;font-family:'Space Mono',monospace;}
.tbcv{font-size:10px;font-weight:800;font-family:'Space Mono',monospace;}
/* Toast */
.toast{position:fixed;bottom:16px;right:16px;background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:9px 14px;font-family:'Space Mono',monospace;font-size:10px;z-index:9999;transform:translateX(160%);transition:transform .3s;}
.toast.on{transform:translateX(0);}
.toast.bt{border-color:var(--gn);} .toast.st{border-color:var(--rd);}
/* Daily limit bar */
.dlbar{background:var(--sf2);border:1px solid var(--bd);border-radius:5px;padding:6px 8px;}
.dl-row{display:flex;justify-content:space-between;align-items:center;font-size:8px;font-family:'Space Mono',monospace;}
.dl-pct{font-weight:700;}
::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-thumb{background:var(--bd);border-radius:2px;}
</style>
</head>
<body>
<!-- TOP BAR -->
<div class="top">
  <div class="logo">
    <div class="logo-i">⚡</div>
    <div>
      <div class="logo-t">Algo<span>Scalp</span> Pro</div>
      <div class="logo-sub">5-STRATEGY SMART SIGNAL ENGINE · NSE/BSE INDIA</div>
    </div>
  </div>
  <div class="top-pills">
    <div class="pill" id="mpill">IDLE</div>
    <div class="pill" id="spill">NO STRATEGY</div>
    <div class="pill" id="tpill" style="display:none">—</div>
    <div class="pill warn" id="dlpill" style="display:none">⚠ DAILY LIMIT</div>
  </div>
</div>

<div class="app">
<!-- LEFT SIDEBAR -->
<aside class="lsb">

  <!-- Strategy -->
  <div>
    <div class="sl">⚡ Scalping Strategy</div>
    <div class="stlist">
      <div class="stitem on" id="st0" onclick="selStrat('vwap_pullback',0)">
        <span class="sc-badge">INST</span>
        <div class="sn">VWAP Pullback</div>
        <div class="sd">Buy/sell when price pulls back to VWAP with RSI confirmation. Best for Nifty index.</div>
      </div>
      <div class="stitem" id="st1" onclick="selStrat('ema_ribbon',1)">
        <span class="sc-badge">MOM</span>
        <div class="sn">EMA Ribbon 5-8-13</div>
        <div class="sd">3 EMAs all aligned = strong momentum. Early trend catch for stocks.</div>
      </div>
      <div class="stitem" id="st2" onclick="selStrat('bb_squeeze',2)">
        <span class="sc-badge">VOL</span>
        <div class="sn">BB Squeeze Breakout</div>
        <div class="sd">Low volatility squeeze → explosion. Good after range-bound periods.</div>
      </div>
      <div class="stitem" id="st3" onclick="selStrat('macd_zero',3)">
        <span class="sc-badge">TRD</span>
        <div class="sn">MACD Zero Cross</div>
        <div class="sd">MACD crosses zero line = trend change confirmed. Medium timeframe.</div>
      </div>
      <div class="stitem" id="st4" onclick="selStrat('smart_combo',4)">
        <span class="sc-badge" style="background:rgba(255,215,0,.2);color:var(--gd);">★ BEST</span>
        <div class="sn" style="color:var(--gd);">SMART COMBO</div>
        <div class="sd">All 5 indicators scored. Fires only at threshold. Highest win rate. Use this!</div>
      </div>
    </div>
  </div>

  <div class="dv"></div>

  <!-- Score threshold (for SMART COMBO) -->
  <div id="thresh-section">
    <div class="sl">🎯 Signal Threshold (Smart Combo)</div>
    <div class="thresh-row">
      <span style="font-size:9px;color:var(--mt);font-family:'Space Mono',monospace;">Need score ≥</span>
      <span class="thresh-val" id="thresh-lbl">3</span>
    </div>
    <input type="range" id="thresh" min="1" max="6" value="3" oninput="document.getElementById('thresh-lbl').textContent=this.value">
    <div style="display:flex;justify-content:space-between;margin-top:2px;font-size:7px;color:var(--mt);font-family:'Space Mono',monospace;">
      <span>1 = More trades</span><span>6 = Fewer, safer</span>
    </div>
  </div>

  <div class="dv"></div>

  <!-- Stock Setup -->
  <div>
    <div class="sl">📊 Instrument</div>
    <div style="display:flex;flex-direction:column;gap:5px;">
      <div class="ig"><label>TICKER (NSE = .NS)</label><input id="sym" value="SBIN.NS" placeholder="^NSEI / SBIN.NS / TCS.NS"></div>
      <div class="r2">
        <div class="ig"><label>TIMEFRAME</label>
          <select id="tf">
            <option value="1m">1 Min ⚡</option>
            <option value="5m" selected>5 Min ★</option>
            <option value="15m">15 Min</option>
            <option value="30m">30 Min</option>
          </select>
        </div>
        <div class="ig"><label>REFRESH sec</label><input type="number" id="ref" value="30" min="10"></div>
      </div>
      <div class="ig"><label>LOOKBACK BARS</label><input type="number" id="look" value="150" min="60"></div>
    </div>
  </div>

  <div class="dv"></div>

  <!-- Risk Management -->
  <div>
    <div class="sl">🛡️ Risk Management</div>
    <div class="risk-grid">
      <div class="ig"><label>CAPITAL ₹</label><input type="number" id="cap" value="100000"></div>
      <div class="ig"><label>RISK % /trade</label><input type="number" id="riskpct" value="1" min="0.5" max="5" step="0.5"></div>
      <div class="ig"><label>MAX TRADES/day</label><input type="number" id="maxtrd" value="10" min="1"></div>
      <div class="ig"><label>MAX LOSS ₹/day</label><input type="number" id="maxloss" value="2000" min="100"></div>
    </div>
    <!-- Daily loss bar -->
    <div class="dlbar" style="margin-top:5px;">
      <div class="dl-row">
        <span style="color:var(--mt)">Daily Loss Used</span>
        <span class="dl-pct" id="dl-pct">₹0 / ₹2000</span>
      </div>
      <div style="background:var(--bd);border-radius:3px;height:5px;overflow:hidden;margin-top:4px;">
        <div id="dl-bar" style="height:100%;width:0%;background:var(--gn);border-radius:3px;transition:width .3s,background .3s;"></div>
      </div>
    </div>
  </div>

  <div class="dv"></div>

  <!-- Telegram -->
  <div class="tgbox">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
      <span style="font-size:8px;color:var(--mt);font-family:'Space Mono',monospace;">📱 TELEGRAM ALERTS</span>
      <label class="tgsw"><input type="checkbox" id="tgen" onchange="tgTog()"><div class="tgtr"></div></label>
    </div>
    <div id="tgf" style="display:none;flex-direction:column;gap:4px;">
      <div class="ig"><label>BOT TOKEN</label><input type="password" id="tgtok" placeholder="123456:ABC..."></div>
      <div class="ig"><label>CHAT ID</label><input id="tgcid" placeholder="-100xxxxxx"></div>
      <button onclick="tgTest()" style="background:var(--sf);border:1px solid var(--bd);color:var(--tx);border-radius:4px;padding:4px;font-size:8px;cursor:pointer;font-family:'Space Mono',monospace;margin-top:2px;">📤 TEST</button>
    </div>
  </div>

  <div class="err" id="err"></div>

  <!-- Action buttons -->
  <div class="r2-btn">
    <button class="bmain bgo" id="bstart" onclick="ptStart()">▶ START</button>
    <button class="bmain bstop" id="bstop"  onclick="ptStop()"  disabled>■ STOP</button>
  </div>
  <button class="bmain bbt" id="bbt" onclick="runBacktest()">📈 BACKTEST THIS STRATEGY</button>

  <div class="dv"></div>

  <!-- Live stats -->
  <div>
    <div class="sl">📈 Session Performance</div>
    <div class="stg">
      <div class="stc"><div class="sl2">Equity</div><div class="sv neu" id="seq">—</div></div>
      <div class="stc"><div class="sl2">Net P&L</div><div class="sv gld" id="spnl">—</div></div>
      <div class="stc"><div class="sl2">Trades</div><div class="sv" id="strd" style="color:var(--pp)">0</div></div>
      <div class="stc"><div class="sl2">Win Rate</div><div class="sv" id="swr">—</div></div>
      <div class="stc"><div class="sl2">Charges</div><div class="sv neg" id="schg">₹0</div></div>
      <div class="stc"><div class="sl2">Signals Today</div><div class="sv neu" id="ssig">0</div></div>
    </div>
  </div>

</aside>

<!-- MAIN CHART -->
<div class="main">
  <div class="cw">
    <div id="mc"></div>
    <div class="ovl" id="ovl">
      <div style="font-size:36px;margin-bottom:10px;">⚡</div>
      <div style="font-size:14px;font-weight:800;margin-bottom:4px;">AlgoScalp Pro</div>
      <div style="font-size:8px;color:var(--mt);font-family:'Space Mono',monospace;line-height:2;text-align:center;">
        5 Strategies · Smart Signal Scoring Engine<br>
        ATR Stop Loss · Risk Management · Telegram Alerts<br>
        Select strategy → Set risk → Press ▶ START
      </div>
    </div>
    <div class="pxti" id="pxti">
      <div class="ptsy" id="ptsy">—</div>
      <div class="ptpx" id="ptpx">—</div>
      <div class="ptch" id="ptch"></div>
    </div>
    <div class="score-meter" id="smeter">
      <div class="sm-row">
        <span class="sm-lbl">SIGNAL SCORE</span>
        <span class="sm-val" id="sm-val">0</span>
        <span class="sm-lbl" id="sm-strat">—</span>
      </div>
      <div class="sm-bars" id="sm-bars"></div>
    </div>
    <div class="sig-flash" id="sig-flash">HOLD</div>
    <div class="cdown on" id="cdown" style="display:none;">⏱ <span id="cdval">—</span>s</div>
  </div>
  <div class="ow" id="ow">
    <div class="olbl" id="olbl">RSI(9)</div>
    <div id="oc"></div>
  </div>
  <!-- Bottom tabs -->
  <div class="bot">
    <div class="tabs">
      <div class="tab on" onclick="stab('tlog')">⚡ Signal Log</div>
      <div class="tab"    onclick="stab('tchg')">💰 Charges</div>
      <div class="tab"    onclick="stab('this')">📂 History</div>
    </div>
    <div class="tc on" id="tlog">
      <div style="overflow-y:auto;flex:1;">
        <table>
          <thead><tr>
            <th>#</th><th>TIME</th><th>SIG</th><th>SCORE</th><th>PRICE</th><th>QTY</th>
            <th>STOP</th><th>TARGET</th><th>CHARGES</th><th>GROSS P&L</th><th>NET P&L</th><th>EQUITY</th>
          </tr></thead>
          <tbody id="tbody"><tr><td colspan="12" style="text-align:center;color:var(--mt);padding:14px;">
            Start the algo to see live signals
          </td></tr></tbody>
        </table>
      </div>
      <div class="totbar">
        <div class="tbc"><div class="tbcl">Signals</div><div class="tbcv neu" id="tt0">0</div></div>
        <div class="tbc"><div class="tbcl">Charges</div><div class="tbcv neg" id="tt1">₹0</div></div>
        <div class="tbc"><div class="tbcl">Gross P&L</div><div class="tbcv" id="tt2">₹0</div></div>
        <div class="tbc"><div class="tbcl">NET PROFIT</div><div class="tbcv gld" id="tt3">₹0</div></div>
        <div class="tbc"><div class="tbcl">Win Rate</div><div class="tbcv" id="tt4">—</div></div>
        <div class="tbc"><div class="tbcl">Best Trade</div><div class="tbcv pos" id="tt5">—</div></div>
      </div>
    </div>
    <div class="tc" id="tchg">
      <div id="chgc" style="padding:10px;font-family:'Space Mono',monospace;font-size:9px;color:var(--mt);">
        Complete a sell to see charges breakdown
      </div>
    </div>
    <div class="tc" id="this">
      <div style="display:flex;justify-content:space-between;padding:5px 10px;border-bottom:1px solid var(--bd);">
        <span style="font-size:7px;color:var(--mt);font-family:'Space Mono',monospace;">SAVED SESSIONS</span>
        <button onclick="loadHist()" style="background:transparent;border:1px solid var(--bd);color:var(--ac);border-radius:3px;padding:1px 6px;font-size:7px;cursor:pointer;font-family:'Space Mono',monospace;">↻</button>
      </div>
      <div id="hisc" style="overflow-y:auto;flex:1;padding:7px 10px;font-size:9px;color:var(--mt);font-family:'Space Mono',monospace;">Loading...</div>
    </div>
  </div>
</div>

<!-- RIGHT SIDEBAR — indicator breakdown -->
<aside class="rsb">
  <div class="sl">📡 Indicator Signals</div>

  <div class="ind-card">
    <div class="ind-name">Score Meter</div>
    <div class="sm-val" id="rs-score" style="font-size:22px;font-weight:800;font-family:'Space Mono',monospace;text-align:center;padding:6px 0;">0</div>
    <div class="gauge"><div id="rs-gauge" class="gauge-fill" style="width:50%;background:var(--mt);"></div></div>
    <div style="display:flex;justify-content:space-between;margin-top:3px;font-size:7px;color:var(--mt);font-family:'Space Mono',monospace;">
      <span>BEAR</span><span>BULL</span>
    </div>
  </div>

  <div class="ind-card" id="ic-vwap">
    <div class="ind-name">VWAP</div>
    <div class="ind-val" id="iv-vwap">—</div>
    <div class="ind-sig neu-sig" id="is-vwap">—</div>
  </div>
  <div class="ind-card" id="ic-ema">
    <div class="ind-name">EMA Ribbon</div>
    <div class="ind-val" id="iv-ema">—</div>
    <div class="ind-sig neu-sig" id="is-ema">—</div>
  </div>
  <div class="ind-card" id="ic-rsi">
    <div class="ind-name">RSI (9)</div>
    <div class="ind-val" id="iv-rsi">—</div>
    <div class="ind-sig neu-sig" id="is-rsi">—</div>
  </div>
  <div class="ind-card" id="ic-macd">
    <div class="ind-name">MACD</div>
    <div class="ind-val" id="iv-macd">—</div>
    <div class="ind-sig neu-sig" id="is-macd">—</div>
  </div>
  <div class="ind-card" id="ic-bb">
    <div class="ind-name">Bollinger</div>
    <div class="ind-val" id="iv-bb">—</div>
    <div class="ind-sig neu-sig" id="is-bb">—</div>
  </div>
  <div class="ind-card" id="ic-sq">
    <div class="ind-name">BB Squeeze</div>
    <div class="ind-val" id="iv-sq">—</div>
    <div class="ind-sig neu-sig" id="is-sq">WATCHING</div>
  </div>

  <div class="dv"></div>

  <!-- Current position -->
  <div>
    <div class="sl">💼 Open Position</div>
    <div id="pos-panel" style="font-size:9px;color:var(--mt);font-family:'Space Mono',monospace;">No open position</div>
  </div>

  <div class="dv"></div>

  <!-- Risk status -->
  <div>
    <div class="sl">🛡️ Risk Status</div>
    <div style="font-size:9px;font-family:'Space Mono',monospace;display:flex;flex-direction:column;gap:4px;">
      <div style="display:flex;justify-content:space-between;"><span style="color:var(--mt)">Trades left</span><span id="rs-trd" style="color:var(--ac)">—</span></div>
      <div style="display:flex;justify-content:space-between;"><span style="color:var(--mt)">Loss budget</span><span id="rs-loss" style="color:var(--gn)">—</span></div>
      <div style="display:flex;justify-content:space-between;"><span style="color:var(--mt)">Market time</span><span id="rs-time" style="color:var(--gn)">—</span></div>
      <div style="display:flex;justify-content:space-between;"><span style="color:var(--mt)">Pos size</span><span id="rs-size" style="color:var(--ac)">—</span></div>
    </div>
  </div>

</aside>

</div><!-- /app -->

<div class="toast" id="toast"></div>

<script>
// ══ STATE ═══════════════════════════════════════
let curStrat='vwap_pullback', running=false;
let cap=100000, equity=100000, pos=null;
let trades=0, wins=0, losses=0, netPnl=0, chgTot=0, sigCount=0;
let dailyLoss=0, dailyTrades=0;
let maxLoss=2000, maxTrades=10;
let trCount=0, bestTrade=0;
let totGross=0, totChg=0, totNet=0;
let interval=null, ctdn=null, cdv=30;
let markers=[];

const SNAMES={vwap_pullback:'VWAP PULLBACK',ema_ribbon:'EMA RIBBON',
  bb_squeeze:'BB SQUEEZE',macd_zero:'MACD ZERO',smart_combo:'SMART COMBO ★'};

// ── Charts ───────────────────────────────────────
const mEl=document.getElementById('mc');
const MC=LightweightCharts.createChart(mEl,{
  layout:{background:{color:'#04080f'},textColor:'#3d5a78'},
  grid:{vertLines:{color:'#1a2d45'},horzLines:{color:'#1a2d45'}},
  crosshair:{mode:LightweightCharts.CrosshairMode.Normal},
  rightPriceScale:{borderColor:'#1a2d45'},
  timeScale:{borderColor:'#1a2d45',timeVisible:true},
  width:mEl.offsetWidth,height:mEl.offsetHeight,
});
const CS=MC.addCandlestickSeries({
  upColor:'#00e676',downColor:'#ff3d57',
  borderUpColor:'#00e676',borderDownColor:'#ff3d57',
  wickUpColor:'#00e676',wickDownColor:'#ff3d57',
});
let VW=null,E5=null,E8=null,E13=null,BBU=null,BBL=null;
const oEl=document.getElementById('oc');
const OC=LightweightCharts.createChart(oEl,{
  layout:{background:{color:'#080f1a'},textColor:'#3d5a78'},
  grid:{vertLines:{color:'#1a2d45'},horzLines:{color:'#1a2d45'}},
  rightPriceScale:{borderColor:'#1a2d45',scaleMargins:{top:.05,bottom:.05}},
  timeScale:{visible:false},width:oEl.offsetWidth,height:oEl.offsetHeight,
});
MC.timeScale().subscribeVisibleLogicalRangeChange(r=>{if(r) OC.timeScale().setVisibleLogicalRange(r);});
let OS1=null,OS2=null;
window.addEventListener('resize',()=>{
  MC.applyOptions({width:mEl.offsetWidth,height:mEl.offsetHeight});
  OC.applyOptions({width:oEl.offsetWidth,height:oEl.offsetHeight});
});

// ── Strategy select ──────────────────────────────
function selStrat(s, idx){
  curStrat=s;
  for(let i=0;i<5;i++) document.getElementById('st'+i).classList.toggle('on',i===idx);
  document.getElementById('spill').textContent=SNAMES[s];
  document.getElementById('thresh-section').style.opacity=s==='smart_combo'?'1':'0.4';
}

// ── Start / Stop ─────────────────────────────────
function ptStart(){
  if(running) return;
  cap=parseFloat(document.getElementById('cap').value);
  equity=cap; pos=null;
  trades=0;wins=0;losses=0;netPnl=0;chgTot=0;sigCount=0;
  dailyLoss=0;dailyTrades=0;
  maxLoss=parseFloat(document.getElementById('maxloss').value);
  maxTrades=parseInt(document.getElementById('maxtrd').value);
  trCount=0;bestTrade=0;totGross=totChg=totNet=0;
  markers=[]; trCount=0;
  running=true;
  document.getElementById('bstart').disabled=true;
  document.getElementById('bstop').disabled=false;
  document.getElementById('mpill').className='pill live'; document.getElementById('mpill').textContent='SCANNING';
  document.getElementById('score-meter')?.classList.add;
  document.getElementById('smeter').style.display='block';
  document.getElementById('sig-flash').style.display='block';
  document.getElementById('cdown').style.display='block';
  document.getElementById('pxti').style.display='block';
  document.getElementById('dlpill').style.display='none';
  resetTLog();
  fetch();
  startCD();
  toast('⚡ AlgoScalp Pro started — '+SNAMES[curStrat],'');
}

function ptStop(){
  if(!running) return;
  running=false; clearInterval(interval); clearInterval(ctdn);
  document.getElementById('bstart').disabled=false;
  document.getElementById('bstop').disabled=true;
  document.getElementById('mpill').className='pill'; document.getElementById('mpill').textContent='STOPPED';
  document.getElementById('cdown').style.display='none';
  saveSession();
  toast('■ AlgoScalp stopped. Session saved.','');
}

function startCD(){
  const ref=parseInt(document.getElementById('ref').value)||30;
  cdv=ref; clearInterval(interval);clearInterval(ctdn);
  interval=setInterval(()=>{if(running) fetch(); cdv=ref;},ref*1000);
  ctdn=setInterval(()=>{cdv--;const e=document.getElementById('cdval');if(e)e.textContent=cdv>0?cdv:ref;},1000);
  document.getElementById('cdval').textContent=cdv;
}

// ── Fetch tick ────────────────────────────────────
async function fetch(){
  const sym=document.getElementById('sym').value.trim().toUpperCase();
  const tf=document.getElementById('tf').value;
  const look=parseInt(document.getElementById('look').value)||150;
  const thresh=parseInt(document.getElementById('thresh').value)||3;
  const riskpct=parseFloat(document.getElementById('riskpct').value)||1;
  try{
    const r=await window.fetch('/api/tick',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({symbol:sym,timeframe:tf,lookback:look,strategy:curStrat,threshold:thresh})});
    const d=await r.json();
    if(d.error){document.getElementById('err').style.display='block';document.getElementById('err').textContent=d.error;return;}
    document.getElementById('err').style.display='none';
    processTick(d,sym,riskpct);
  }catch(e){console.error(e);}
}

// ── Process tick ──────────────────────────────────
function processTick(d, sym, riskpct){
  const {candles,signals,scores,sub_sigs,stops,targets,inds,latest_price,latest_signal,latest_score,latest_subs,latest_time}=d;

  // Update chart
  CS.setData(candles);
  setFull(VW,inds.vwap,candles);
  setFull(E5,inds.e5,candles);
  setFull(E8,inds.e8,candles);
  setFull(E13,inds.e13,candles);
  setFull(BBU,inds.bbu,candles);
  setFull(BBL,inds.bbl,candles);
  setFull(OS1,inds.rsi,candles);
  CS.setMarkers([...markers]);

  // Price ticker
  const fc=candles[0]?.close||latest_price;
  const chg=((latest_price-fc)/fc*100);
  document.getElementById('ptsy').textContent=sym;
  document.getElementById('ptpx').textContent='₹'+f2(latest_price);
  document.getElementById('ptch').textContent=(chg>=0?'+':'')+chg.toFixed(2)+'%';
  document.getElementById('ptch').style.color=chg>=0?'var(--gn)':'var(--rd)';

  // Score meter
  document.getElementById('sm-val').textContent=latest_score>0?'+'+latest_score:latest_score;
  document.getElementById('sm-strat').textContent=SNAMES[curStrat];
  const bars=document.getElementById('sm-bars');
  bars.innerHTML='';
  for(let i=-5;i<=5;i++){
    if(i===0) continue;
    const b=document.createElement('div');
    b.className='sm-bar';
    if(latest_score>0&&i>0&&i<=latest_score) b.classList.add('bull');
    else if(latest_score<0&&i<0&&i>=latest_score) b.classList.add('bear');
    bars.appendChild(b);
  }

  // Right sidebar indicator breakdown
  updInd('vwap',f2(inds.vwap?.[inds.vwap.length-1]),latest_subs?.VWAP);
  updInd('ema',inds.e5?.[inds.e5.length-1]?f2(inds.e5[inds.e5.length-1]):'—',latest_subs?.EMA);
  updInd('rsi',inds.rsi?.[inds.rsi.length-1]?f2(inds.rsi[inds.rsi.length-1]):'—',latest_subs?.RSI);
  updInd('macd',inds.macd?.[inds.macd.length-1]?f2(inds.macd[inds.macd.length-1]):'—',latest_subs?.MACD);
  const bwidth=inds.bbu?.[inds.bbu.length-1]&&inds.bbl?.[inds.bbl.length-1]?
    f2((inds.bbu[inds.bbu.length-1]-inds.bbl[inds.bbl.length-1])/latest_price*100)+'%':'—';
  updInd('bb',bwidth,latest_subs?.BB);
  document.getElementById('iv-sq').textContent=latest_subs?.SQUEEZE||'—';
  document.getElementById('is-sq').textContent=latest_subs?.SQUEEZE?'⚡ BREAKOUT READY':'Watching...';
  document.getElementById('is-sq').className=latest_subs?.SQUEEZE?'ind-sig bull-sig':'ind-sig neu-sig';

  // Score gauge
  const gPct=50+(latest_score/6)*50;
  const gEl=document.getElementById('rs-gauge');
  gEl.style.width=Math.min(100,Math.max(0,gPct))+'%';
  gEl.style.background=latest_score>=3?'var(--gn)':latest_score<=-3?'var(--rd)':'var(--or)';
  document.getElementById('rs-score').textContent=latest_score>0?'+'+latest_score:latest_score;
  document.getElementById('rs-score').style.color=latest_score>=3?'var(--gn)':latest_score<=-3?'var(--rd)':'var(--or)';

  // Signal flash
  const sf=document.getElementById('sig-flash');
  if(latest_signal==='BUY'){ sf.className='sig-flash buy'; sf.textContent='⬆ BUY SIGNAL'; }
  else if(latest_signal==='SELL'){ sf.className='sig-flash sell'; sf.textContent='⬇ SELL SIGNAL'; }
  else{ sf.className='sig-flash hold'; sf.textContent='⏸ HOLD'; }

  // Risk status
  const tradesLeft=maxTrades-dailyTrades;
  const lossLeft=maxLoss-dailyLoss;
  document.getElementById('rs-trd').textContent=tradesLeft+' left';
  document.getElementById('rs-trd').style.color=tradesLeft<=2?'var(--rd)':'var(--ac)';
  document.getElementById('rs-loss').textContent='₹'+f2(lossLeft)+' left';
  document.getElementById('rs-loss').style.color=lossLeft<maxLoss*0.3?'var(--rd)':'var(--gn)';
  const isMarketTime=checkMarketTime();
  document.getElementById('rs-time').textContent=isMarketTime?'✓ OPEN':'✗ CLOSED';
  document.getElementById('rs-time').style.color=isMarketTime?'var(--gn)':'var(--rd)';

  // Position sizing: risk% of capital / ATR stop distance
  const atr=inds.atr?.[inds.atr.length-1]||0;
  const posSize=atr>0?Math.floor((cap*riskpct/100)/(1.5*atr)):0;
  document.getElementById('rs-size').textContent=posSize+' shares';

  // Daily loss bar
  const dlPct=Math.min(100,(dailyLoss/maxLoss*100));
  document.getElementById('dl-bar').style.width=dlPct+'%';
  document.getElementById('dl-bar').style.background=dlPct>80?'var(--rd)':dlPct>50?'var(--or)':'var(--gn)';
  document.getElementById('dl-pct').textContent='₹'+f2(dailyLoss)+' / ₹'+f2(maxLoss);

  // Execute paper trade with all risk checks
  const lt=new Date(latest_time*1000);
  if(!isMarketTime){ return; } // time filter
  if(dailyLoss>=maxLoss){
    document.getElementById('dlpill').style.display='block';
    toast('⚠ Daily max loss reached. Trading paused.',''); return;
  }
  if(dailyTrades>=maxTrades){ return; }

  const latestStop=stops[stops.length-1];
  const latestTgt=targets[targets.length-1];

  if(latest_signal==='BUY'&&!pos){
    const qty=Math.max(1,posSize||Math.floor(equity/latest_price));
    if(qty>0&&equity>=qty*latest_price){
      pos={price:latest_price,qty,date:lt,val:qty*latest_price,stop:latestStop,target:latestTgt};
      const ch=calcChg(pos.val,pos.val,lt,lt);
      equity-=pos.val+ch.total/2; sigCount++; dailyTrades++;
      markers.push({time:latest_time,position:'belowBar',color:'#00e676',shape:'arrowUp',text:'BUY ₹'+f2(latest_price)});
      CS.setMarkers([...markers]);
      addRow('BUY',lt,latest_price,qty,pos.val,latest_score,latestStop,latestTgt,{brok:ch.brok/2,stt:ch.stt/2,total:ch.total/2},null,equity+pos.qty*latest_price);
      sendTG('BUY',sym,latest_price,qty,pos.val,latest_score,latestStop,latestTgt,null);
      updPos(latest_price);
      toast('⬆ PAPER BUY: '+qty+' × ₹'+f2(latest_price)+' | Score:'+latest_score,'bt');
    }
  } else if(latest_signal==='SELL'&&pos){
    const sv=pos.qty*latest_price;
    const ch=calcChg(pos.val,sv,pos.date,lt);
    equity+=sv-ch.total-ch.cg_tax;
    if(ch.net>=0) wins++; else{losses++;dailyLoss+=Math.abs(ch.net);}
    netPnl+=ch.net; chgTot+=ch.total; trades++; dailyTrades++;
    if(ch.net>bestTrade) bestTrade=ch.net;
    totGross+=ch.gross; totChg+=ch.total; totNet+=ch.net;
    markers.push({time:latest_time,position:'aboveBar',color:'#ff3d57',shape:'arrowDown',text:'SELL ₹'+f2(latest_price)});
    CS.setMarkers([...markers]);
    addRow('SELL',lt,latest_price,pos.qty,sv,latest_score,pos.stop,pos.target,ch,ch.gross,equity);
    updChgTab(ch,pos.price,latest_price);
    updTots();
    sendTG('SELL',sym,latest_price,pos.qty,sv,latest_score,pos.stop,pos.target,ch);
    toast('⬇ PAPER SELL | Net: '+(ch.net>=0?'+':'')+' ₹'+f2(ch.net),'st');
    pos=null; updPos(latest_price);
  } else if(pos){
    // Check stop loss hit
    if(pos.stop&&latest_price<=pos.stop){
      toast('⚠ STOP LOSS HIT at ₹'+f2(latest_price),'st');
      // Force SELL at stop price
      const sv=pos.qty*pos.stop;
      const ch=calcChg(pos.val,sv,pos.date,lt);
      equity+=sv-ch.total-ch.cg_tax; losses++; dailyLoss+=Math.abs(ch.net);
      netPnl+=ch.net; chgTot+=ch.total; trades++; dailyTrades++;
      totGross+=ch.gross;totChg+=ch.total;totNet+=ch.net;
      markers.push({time:latest_time,position:'aboveBar',color:'#ff6600',shape:'arrowDown',text:'SL ₹'+f2(pos.stop)});
      CS.setMarkers([...markers]);
      addRow('SL HIT',lt,pos.stop,pos.qty,sv,latest_score,pos.stop,pos.target,ch,ch.gross,equity);
      updTots(); pos=null; updPos(latest_price);
    } else {
      updPos(latest_price);
    }
  }

  updStats();
}

function setFull(s,arr,candles){
  if(s&&arr) s.setData(candles.map((c,i)=>arr[i]!=null?{time:c.time,value:arr[i]}:null).filter(Boolean));
}
function setupLines(){
  [VW,E5,E8,E13,BBU,BBL].forEach(s=>{if(s){try{MC.removeSeries(s);}catch(e){}}});
  [OS1,OS2].forEach(s=>{if(s){try{OC.removeSeries(s);}catch(e){}}});
  VW=MC.addLineSeries({color:'#ffd700',lineWidth:2,title:'VWAP'});
  E5=MC.addLineSeries({color:'#00e5ff',lineWidth:1,title:'EMA5'});
  E8=MC.addLineSeries({color:'rgba(0,229,255,.5)',lineWidth:1,title:'EMA8'});
  E13=MC.addLineSeries({color:'rgba(0,229,255,.25)',lineWidth:1,title:'EMA13'});
  BBU=MC.addLineSeries({color:'rgba(0,229,255,.25)',lineWidth:1,title:'BB Upper'});
  BBL=MC.addLineSeries({color:'rgba(255,61,87,.25)',lineWidth:1,title:'BB Lower'});
  OS1=OC.addLineSeries({color:'#a78bfa',lineWidth:1.5});
  document.getElementById('olbl').textContent='RSI(9)';
}
setupLines();

// ── Indicator card update ─────────────────────────
function updInd(key,val,sig){
  const iv=document.getElementById('iv-'+key);
  const is=document.getElementById('is-'+key);
  if(iv) iv.textContent=val||'—';
  if(!is||!sig) return;
  const isBull=sig.includes('BULL');
  const isBear=sig.includes('BEAR');
  is.textContent=sig;
  is.className='ind-sig '+(isBull?'bull-sig':isBear?'bear-sig':'neu-sig');
}

// ── Market time check ─────────────────────────────
function checkMarketTime(){
  const now=new Date(); const ist=new Date(now.getTime()+5.5*3600000);
  const h=ist.getUTCHours(); const m=ist.getUTCMinutes(); const t=h*60+m;
  const open1=9*60+20, close1=11*60+30;
  const open2=13*60+30, close2=15*60;
  const isWeekday=ist.getUTCDay()>0&&ist.getUTCDay()<6;
  return isWeekday&&((t>=open1&&t<=close1)||(t>=open2&&t<=close2));
}

// ── Position card ─────────────────────────────────
function updPos(cp){
  const el=document.getElementById('pos-panel');
  if(!pos){el.textContent='No open position';return;}
  const up=(cp-pos.price)*pos.qty;
  const upPct=((cp-pos.price)/pos.price*100);
  el.innerHTML=`<div style="background:rgba(0,230,118,.08);border:1px solid var(--gn);border-radius:5px;padding:7px;">
    <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">Buy</span><span>₹${f2(pos.price)}</span></div>
    <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">Qty</span><span>${pos.qty}</span></div>
    <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">Stop</span><span style="color:var(--rd)">${pos.stop?'₹'+f2(pos.stop):'—'}</span></div>
    <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">Target</span><span style="color:var(--gn)">${pos.target?'₹'+f2(pos.target):'—'}</span></div>
    <div style="display:flex;justify-content:space-between;margin:4px 0;border-top:1px solid var(--bd);padding-top:4px;">
      <span>Unrealized</span>
      <span style="color:${up>=0?'var(--gn)':'var(--rd)'};font-weight:700;">${up>=0?'+':''}₹${f2(up)} (${upPct>=0?'+':''}${upPct.toFixed(2)}%)</span>
    </div>
  </div>`;
}

// ── Stats ─────────────────────────────────────────
function updStats(){
  const pnl=equity-cap;
  document.getElementById('seq').textContent='₹'+f2(equity);
  const pe=document.getElementById('spnl');
  pe.textContent=(pnl>=0?'+':'')+' ₹'+f2(pnl); pe.style.color=pnl>=0?'var(--gd)':'var(--rd)';
  document.getElementById('strd').textContent=trades;
  const wr=trades>0?(wins/trades*100):0;
  const we=document.getElementById('swr');
  we.textContent=trades>0?wr.toFixed(0)+'%':'—'; we.className='sv '+(wr>=50?'pos':'neg');
  document.getElementById('schg').textContent='-₹'+f2(chgTot);
  document.getElementById('ssig').textContent=sigCount;
}

// ── Charges helpers ────────────────────────────────
function calcChg(bv,sv,bd,sd){
  const bb=Math.max(Math.min(0.001*bv,20),5),sb=Math.max(Math.min(0.001*sv,20),5),bk=bb+sb;
  const stt=0.001*bv+0.001*sv,exc=0.0000325*(bv+sv),sebi=0.000001*(bv+sv);
  const stamp=0.00015*bv,gst=0.18*(bk+exc+sebi),dp=20,tot=bk+stt+exc+sebi+stamp+gst+dp;
  const gross=sv-bv,days=bd&&sd?Math.floor((sd-bd)/864e5):0;
  let cg_tax=0,cg_type='',cg_rate=0;
  if(gross>0){if(days<365){cg_type='STCG';cg_rate=0.20;cg_tax=gross*0.20*1.04;}
    else{cg_type='LTCG';cg_rate=0.125;cg_tax=Math.max(0,gross-125000)*0.125*1.04;}}
  return{brok:+bk.toFixed(2),stt:+stt.toFixed(2),exc:+exc.toFixed(4),sebi:+sebi.toFixed(4),
    stamp:+stamp.toFixed(2),gst:+gst.toFixed(2),dp:20,total:+tot.toFixed(2),
    gross:+gross.toFixed(2),cg_type,cg_rate_pct:+(cg_rate*100).toFixed(1),
    cg_tax:+cg_tax.toFixed(2),net:+(gross-tot-cg_tax).toFixed(2),days};
}
function f2(n){return parseFloat(n||0).toFixed(2);}
function fdt(d){return d instanceof Date?d.toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit'}):'—';}

function addRow(type,dt,px,qty,val,score,stop,target,ch,gp,eq){
  const tb=document.getElementById('tbody');
  if(!trCount) tb.innerHTML='';
  trCount++;
  const isSell=type==='SELL'||type==='SL HIT';
  const row=document.createElement('tr');
  const gStr=isSell&&gp!=null?`<span class="${gp>=0?'pp':'np'}">${gp>=0?'+':''}₹${f2(gp)}</span>`:'—';
  const nStr=isSell?`<span class="${(ch.net||0)>=0?'pp':'np'}">${(ch.net||0)>=0?'+':''}₹${f2(ch.net||0)}</span>`:'—';
  row.innerHTML=`<td style="color:var(--mt)">${trCount}</td>
    <td>${fdt(dt)}</td>
    <td><span class="${type==='BUY'?'tbt':'tts'}">${type}</span></td>
    <td style="color:${score>=3?'var(--gn)':score<=-3?'var(--rd)':'var(--or)'}">${score>0?'+':''}${score}</td>
    <td>₹${f2(px)}</td><td>${qty}</td>
    <td style="color:var(--rd);font-size:8px;">${stop?'₹'+f2(stop):'—'}</td>
    <td style="color:var(--gn);font-size:8px;">${target?'₹'+f2(target):'—'}</td>
    <td class="neg">-₹${f2(ch.total||0)}</td>
    <td>${gStr}</td><td>${nStr}</td>
    <td style="color:var(--ac)">₹${f2(eq)}</td>`;
  tb.prepend(row);
}

function updTots(){
  document.getElementById('tt0').textContent=sigCount;
  document.getElementById('tt1').textContent='₹'+f2(totChg);
  const t2=document.getElementById('tt2'); t2.textContent=(totGross>=0?'+':'')+' ₹'+f2(totGross); t2.style.color=totGross>=0?'var(--gn)':'var(--rd)';
  const t3=document.getElementById('tt3'); t3.textContent=(totNet>=0?'+':'')+' ₹'+f2(totNet); t3.style.color=totNet>=0?'var(--gd)':'var(--rd)';
  const wr=trades>0?(wins/trades*100):0; const t4=document.getElementById('tt4');
  t4.textContent=trades?wr.toFixed(0)+'%':'—'; t4.style.color=wr>=50?'var(--gn)':'var(--rd)';
  const t5=document.getElementById('tt5'); t5.textContent=bestTrade>0?'+₹'+f2(bestTrade):'—';
}

function resetTLog(){
  trCount=0;
  document.getElementById('tbody').innerHTML='<tr><td colspan="12" style="text-align:center;color:var(--mt);padding:12px;">Waiting for signals...</td></tr>';
  ['tt0','tt1','tt2','tt3','tt4','tt5'].forEach(id=>document.getElementById(id).textContent='—');
}

function updChgTab(ch,bp,sp){
  const nt=ch.net>=0?'var(--gn)':'var(--rd)';
  document.getElementById('chgc').innerHTML=`
  <div style="display:grid;grid-template-columns:repeat(4,1fr);font-family:'Space Mono',monospace;font-size:9px;padding:8px;">
    <div style="padding:6px;border-right:1px solid var(--bd);">
      <div style="color:var(--mt);font-size:7px;margin-bottom:4px;">🏦 GROWW</div>
      <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">Brokerage</span><span style="color:var(--rd)">-₹${f2(ch.brok)}</span></div>
      <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">DP</span><span style="color:var(--rd)">-₹${f2(ch.dp)}</span></div>
    </div>
    <div style="padding:6px;border-right:1px solid var(--bd);">
      <div style="color:var(--mt);font-size:7px;margin-bottom:4px;">🏛️ GOVT</div>
      <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">STT</span><span style="color:var(--rd)">-₹${f2(ch.stt)}</span></div>
      <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">Exchange</span><span style="color:var(--rd)">-₹${f2(ch.exc)}</span></div>
      <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">Stamp+GST</span><span style="color:var(--rd)">-₹${f2((ch.stamp||0)+(ch.gst||0))}</span></div>
    </div>
    <div style="padding:6px;border-right:1px solid var(--bd);">
      <div style="color:var(--mt);font-size:7px;margin-bottom:4px;">📋 TAX</div>
      <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">Type</span><span style="color:var(--or)">${ch.cg_type||'—'}</span></div>
      <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">Held</span><span>${ch.days}d</span></div>
      <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">Tax</span><span style="color:var(--or)">-₹${f2(ch.cg_tax)}</span></div>
    </div>
    <div style="padding:6px;">
      <div style="color:var(--mt);font-size:7px;margin-bottom:4px;">✅ RESULT</div>
      <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">Gross</span><span style="color:${ch.gross>=0?'var(--gn)':'var(--rd)'}">${ch.gross>=0?'+':''}₹${f2(ch.gross)}</span></div>
      <div style="display:flex;justify-content:space-between;margin:2px 0;"><span style="color:var(--mt)">Charges</span><span style="color:var(--rd)">-₹${f2(ch.total)}</span></div>
      <div style="display:flex;justify-content:space-between;margin:4px 0;border-top:1px solid var(--ac);padding-top:4px;">
        <span style="font-weight:700;">NET</span>
        <span style="font-size:13px;font-weight:800;color:${nt};">${ch.net>=0?'+':''}₹${f2(ch.net)}</span>
      </div>
    </div>
  </div>`;
  stab('tchg');
}

// ── Telegram ──────────────────────────────────────
function tgTog(){document.getElementById('tgf').style.display=document.getElementById('tgen').checked?'flex':'none';}
async function tgTest(){
  const t=document.getElementById('tgtok').value.trim(),c=document.getElementById('tgcid').value.trim();
  if(!t||!c){toast('Enter token + chat ID','');return;}
  const r=await window.fetch('/api/tg_test',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:t,chat_id:c})});
  const d=await r.json(); toast(d.ok?'✅ Connected!':'❌ '+d.error,'');
}
async function sendTG(type,sym,price,qty,val,score,stop,target,ch){
  if(!document.getElementById('tgen').checked) return;
  const t=document.getElementById('tgtok').value.trim(),c=document.getElementById('tgcid').value.trim();
  if(!t||!c) return;
  let m=`<b>⚡ AlgoScalp Pro — ${type}</b>\n\nSymbol: <b>${sym}</b>\nStrategy: ${SNAMES[curStrat]}\nScore: ${score>0?'+':''}${score}\nPrice: ₹${f2(price)}\nQty: ${qty}\nValue: ₹${f2(val)}`;
  if(stop) m+=`\nStop Loss: ₹${f2(stop)}`;
  if(target) m+=`\nTarget: ₹${f2(target)}`;
  if(ch&&type==='SELL') m+=`\n\n<b>P&L</b>\nGross: ${ch.gross>=0?'+':''}₹${f2(ch.gross)}\nCharges: -₹${f2(ch.total)}\nTax: -₹${f2(ch.cg_tax)}\n<b>Net: ${ch.net>=0?'+':''}₹${f2(ch.net)}</b>`;
  m+=`\n\n${new Date().toLocaleString('en-IN')}`;
  await window.fetch('/api/tg_send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:t,chat_id:c,message:m})});
}

// ── Backtest redirect ──────────────────────────────
function runBacktest(){
  toast('▶ Open BackTest Pro v4 (port 5051) to backtest this strategy','');
}

// ── Save / load ────────────────────────────────────
async function saveSession(){
  if(!trades) return;
  const s={id:Date.now(),date:new Date().toLocaleDateString('en-IN'),
    symbol:document.getElementById('sym').value.toUpperCase(),strategy:SNAMES[curStrat],
    timeframe:document.getElementById('tf').value,capital:cap,
    net_pnl:netPnl,trades,wins,losses,charges:chgTot,signals:sigCount};
  await window.fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(s)});
}
async function loadHist(){
  const r=await window.fetch('/api/history'); const d=await r.json();
  const el=document.getElementById('hisc');
  if(!d.sessions?.length){el.textContent='No sessions yet';return;}
  el.innerHTML=d.sessions.slice().reverse().map(s=>`
    <div style="background:var(--sf2);border:1px solid var(--bd);border-radius:5px;padding:7px;margin-bottom:5px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span style="color:var(--ac);font-weight:700;">${s.symbol} · ${s.strategy}</span>
        <span style="color:var(--mt);font-size:7px;">${s.date} · ${s.timeframe}</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;">
        <div><div style="color:var(--mt);font-size:6px;">NET P&L</div><div style="color:${s.net_pnl>=0?'var(--gd)':'var(--rd)'};font-weight:700;">${s.net_pnl>=0?'+':''}₹${f2(s.net_pnl)}</div></div>
        <div><div style="color:var(--mt);font-size:6px;">TRADES</div><div>${s.trades}</div></div>
        <div><div style="color:var(--mt);font-size:6px;">WINS</div><div style="color:var(--gn);">${s.wins}</div></div>
        <div><div style="color:var(--mt);font-size:6px;">SIGNALS</div><div>${s.signals||0}</div></div>
      </div>
    </div>`).join('');
}

// ── UI helpers ────────────────────────────────────
function stab(t){
  ['tlog','tchg','this'].forEach((id,i)=>{
    document.querySelectorAll('.tab')[i].classList.toggle('on',id===t);
    document.getElementById(id).classList.toggle('on',id===t);
  });
  if(t==='this') loadHist();
}
function toast(m,cls){
  const t=document.getElementById('toast');t.textContent=m;t.className='toast '+(cls?cls:'');
  t.classList.add('on'); setTimeout(()=>t.classList.remove('on'),5000);
}
function showOvl(h){const o=document.getElementById('ovl');o.innerHTML=h;o.classList.remove('hid');}
function hideOvl(){document.getElementById('ovl').classList.add('hid');}

// Init
document.getElementById('ovl').classList.remove('hid');
loadHist();
</script>
</body>
</html>"""

# ════════════════════════════════════════════════
#  FLASK ROUTES
# ════════════════════════════════════════════════
@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/api/tick', methods=['POST'])
def api_tick():
    try:
        b = request.json
        sym      = b['symbol']
        tf       = b.get('timeframe', '5m')
        look     = int(b.get('lookback', 150))
        strategy = b.get('strategy', 'smart_combo')
        params   = {k: v for k, v in b.items() if v is not None}

        PM = {'1m':'1d','5m':'5d','15m':'30d','30m':'60d'}
        data = yf.download(sym, period=PM.get(tf,'5d'), interval=tf,
                           auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty: return jsonify({'error': f'No data for {sym}'})
        if len(data) > look: data = data.iloc[-look:]

        signals, scores, sub_sigs, stops, targets, inds = compute_smart_signals(data, strategy, params)

        candles = []
        for idx, row in data.iterrows():
            candles.append(dict(time=int(idx.timestamp()),
                open=round(float(row['Open']),2), high=round(float(row['High']),2),
                low=round(float(row['Low']),2),  close=round(float(row['Close']),2)))

        def cl(lst):
            if lst is None: return None
            return [None if (v is None or (isinstance(v,float) and np.isnan(v))) else round(float(v),4) for v in lst]

        inds_out = {k: cl(v) if isinstance(v, list) else v for k, v in inds.items()}

        return jsonify({
            'candles': candles,
            'signals': signals,
            'scores':  scores,
            'sub_sigs': [str(s) for s in sub_sigs],
            'stops':   cl(stops),
            'targets': cl(targets),
            'inds':    inds_out,
            'latest_price':  float(data['Close'].iloc[-1]),
            'latest_signal': signals[-1],
            'latest_score':  scores[-1],
            'latest_subs':   sub_sigs[-1],
            'latest_time':   int(data.index[-1].timestamp()),
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/tg_test', methods=['POST'])
def api_tg_test():
    b = request.json
    ok = send_telegram(b['token'], b['chat_id'],
        "✅ <b>AlgoScalp Pro</b>\n\nTelegram connected!\n\nYou'll receive all BUY/SELL signals with full P&L breakdown here. ⚡")
    return jsonify({'ok': ok, 'error': '' if ok else 'Check token/chat_id'})

@app.route('/api/tg_send', methods=['POST'])
def api_tg_send():
    b = request.json
    ok = send_telegram(b['token'], b['chat_id'], b['message'])
    return jsonify({'ok': ok})

@app.route('/api/save', methods=['POST'])
def api_save():
    try:
        s = request.json; d = load_hist()
        d['sessions'].append(s); save_hist(d)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/api/history')
def api_history(): return jsonify(load_hist())

if __name__ == '__main__':
    import webbrowser, threading, time as _t
    def ob():
        _t.sleep(1.2); webbrowser.open('http://localhost:5052')
    threading.Thread(target=ob, daemon=True).start()
    print("="*60)
    print("  ⚡ AlgoScalp Pro — Smart Algo Scalping Bot")
    print()
    print("  5 Strategies:")
    print("    1. VWAP Pullback     — Best for Nifty/BankNifty")
    print("    2. EMA Ribbon 5-8-13 — Momentum entry")
    print("    3. BB Squeeze        — Volatility breakout")
    print("    4. MACD Zero Cross   — Trend change")
    print("    5. SMART COMBO ★    — All combined, highest win rate")
    print()
    print("  Built-in Risk Management:")
    print("    • ATR Stop Loss + Target (1:1.33 R:R)")
    print("    • Max daily loss limit (auto-stop)")
    print("    • Max trades per day filter")
    print("    • Market time filter (9:20-11:30, 13:30-15:00 IST)")
    print("    • Position sizing by risk %")
    print()
    print("  Open: http://localhost:5052")
    print("="*60)
  import os port = int(os.environ.get("PORT", 5052)) app.run(debug=False, host="0.0.0.0", port=port)
