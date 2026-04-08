#!/usr/bin/env python3
"""
Stock Screener — scans S&P 500, Dow, Nasdaq-100, Russell 2000/3000 for VCP coils.

Replicates the Pine Script VCP/coil detection logic:
  - ATR contraction detection (Strict/Balanced/Loose)
  - MA spread tightness (EMA9 vs EMA21, EMA21 vs HMA40)
  - Price proximity to key EMAs
  - RMV (Relative Magnitude of Volatility) calculation
  - Coil box tracking with breakout detection
  - Inside-box detection (price in bottom 80% of coil range)

Outputs screener.json with results for each screener mode.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ─── Hardcoded stock universe (reliable, no API/scraping needed) ──────────────
# Updated March 2026. Update periodically when index reconstitutions occur.

SP500_TICKERS = [
    "AAPL","ABBV","ABT","ACN","ADBE","ADI","ADM","ADP","ADSK","AEE","AEP","AES","AFL","AIG","AIZ","AJG",
    "AKAM","ALB","ALGN","ALK","ALL","ALLE","AMAT","AMCR","AMD","AME","AMGN","AMP","AMT","AMZN","ANET",
    "ANSS","AON","AOS","APA","APD","APH","APTV","ARE","ARES","ATO","AVGO","AVY","AWK","AXP","AZO","BA",
    "BAC","BAX","BBY","BDX","BEN","BG","BIIB","BIO","BK","BKNG","BKR","BLDR","BLK","BMY","BR","BRK-B",
    "BRO","BSX","BWA","BX","BXP","C","CAG","CAH","CARR","CAT","CB","CBOE","CBRE","CCI","CCL","CDNS",
    "CDW","CE","CEG","CF","CFG","CHD","CHRW","CHTR","CI","CINF","CL","CLX","CNC","CNP","COF","COO",
    "COP","COR","COST","CPAY","CPB","CPRT","CPT","CRL","CRM","CRWD","CSCO","CSGP","CSX","CTAS","CTRA",
    "CTSH","CTVA","CVS","CVX","D","DAL","DAY","DD","DE","DECK","DFS","DG","DGX","DHI","DHR","DIS",
    "DLTR","DOV","DOW","DPZ","DRI","DT","DUK","DVA","DVN","DXCM","EA","EBAY","ECL","ED","EFX","EIX",
    "EL","EMN","EMR","ENPH","EOG","EPAM","EQIX","EQR","EQT","ES","ESS","ETN","ETR","EVRG","EW","EXC",
    "EXPD","EXPE","EXR","F","FANG","FAST","FCX","FDS","FDX","FE","FFIV","FI","FICO","FIS","FISV","FITB",
    "FMC","FOX","FOXA","FRT","FSLR","FTNT","FTV","GD","GDDY","GE","GEHC","GEN","GEV","GILD","GIS","GL",
    "GLW","GM","GNRC","GOOG","GOOGL","GPC","GPN","GRMN","GS","GWW","HAL","HAS","HBAN","HCA","HD","HOLX",
    "HON","HPE","HPQ","HRL","HSIC","HST","HSY","HUBB","HUM","HWM","IBM","ICE","IDXX","IEX","IFF","INCY",
    "INTC","INTU","INVH","IP","IPG","IQV","IR","IRM","ISRG","IT","ITW","IVZ","J","JBHT","JBL","JCI",
    "JKHY","JNJ","JNPR","JPM","K","KDP","KEY","KEYS","KHC","KIM","KKR","KLAC","KMB","KMI","KMX","KO",
    "KR","KVUE","L","LDOS","LEN","LH","LHX","LIN","LKQ","LLY","LMT","LNT","LOW","LRCX","LULU","LUV",
    "LVS","LW","LYB","LYV","MA","MAA","MAR","MAS","MCD","MCHP","MCK","MCO","MDLZ","MDT","MET","META",
    "MGM","MHK","MKC","MLM","MMC","MMM","MNST","MO","MOH","MOS","MPC","MPWR","MRK","MRNA","MRVL","MS",
    "MSCI","MSFT","MSI","MTB","MTCH","MTD","MU","NCLH","NDAQ","NDSN","NEE","NEM","NFLX","NI","NKE","NOC",
    "NOW","NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR","NWS","NWSA","NXPI","O","ODFL","OKE","OMC","ON",
    "ORCL","ORLY","OTIS","OXY","PANW","PARA","PAYC","PAYX","PCAR","PCG","PEG","PEP","PFE","PFG","PG",
    "PGR","PH","PHM","PKG","PLD","PLTR","PM","PNC","PNR","PNW","POOL","PPG","PPL","PRU","PSA","PSX",
    "PTC","PVH","PWR","PYPL","QCOM","QRVO","RCL","REG","REGN","RF","RHI","RJF","RL","RMD","ROK","ROL",
    "ROP","ROST","RSG","RTX","RVTY","SBAC","SBUX","SCHW","SEE","SHW","SJM","SLB","SMCI","SNA","SNPS",
    "SO","SOLV","SPG","SPGI","SRE","STE","STLD","STT","STX","STZ","SWK","SWKS","SYF","SYK","SYY","T",
    "TAP","TDG","TDY","TECH","TEL","TER","TFC","TFX","TGT","TJX","TMO","TMUS","TPR","TRGP","TRMB",
    "TROW","TRV","TSCO","TSLA","TSN","TT","TTWO","TXN","TXT","TYL","UAL","UBER","UDR","UHS","ULTA",
    "UNH","UNP","UPS","URI","USB","V","VFC","VICI","VLO","VLTO","VMC","VRSK","VRSN","VRTX","VST","VTR",
    "VTRS","VZ","WAB","WAT","WBA","WBD","WDC","WEC","WELL","WFC","WHR","WM","WMB","WMT","WRB","WST",
    "WTW","WY","WYNN","XEL","XOM","XRAY","XYL","YUM","ZBH","ZBRA","ZTS",
]

NASDAQ100_EXTRA = [
    # Tickers in Nasdaq-100 but not already in S&P 500
    "ABNB","APP","ARM","ASML","AZN","DASH","DDOG","DELL","GFS","KHC","LINU",
    "MDB","MELI","PDD","TEAM","TTD","WDAY","ZS",
]

# Top ~500 Russell 2000 / IWM holdings (most liquid small/mid caps not in S&P 500)
RUSSELL2000_TICKERS = [
    "AAOI","ACLX","ACLS","ACVA","ADMA","AEIS","AEVA","AGI","AGYS","AHCO","AI","AIT","ALGT","ALKS",
    "ALLO","ALRM","ALSN","AM","AMKR","AMPH","AMPL","AMR","AMSF","ANDE","ANIP","AORT","APAM","APGE",
    "APPF","ARAY","ARCB","ARCO","ARIS","AROC","ARQQ","ARRY","ARVN","ASGN","ASTS","ATEC","ATGE","ATI",
    "AVAV","AVNT","AVPT","AXNX","AXSM","AY","AZZ","B","BBIO","BBWI","BC","BCO","BE","BELFB",
    "BGC","BHVN","BKH","BL","BLKB","BNL","BOOT","BOX","BRBR","BRSP","BSM","BTDR","BTU","BUR",
    "BWXT","BYD","CAKE","CALM","CARG","CARS","CAVA","CBRE","CBSH","CC","CCS","CDE","CEIX","CELH",
    "CENX","CERS","CHRD","CHS","CIFR","CIVI","CLDX","CLSK","CMC","CMPO","CNK","CNM","CNS","COCO",
    "COHR","COMP","CORT","CPA","CPK","CPRX","CRAI","CRDO","CRI","CRK","CRSP","CRS","CRVL","CSGS",
    "CW","CWK","CWST","CXT","CYAD","CYTK","CZR","DCGO","DFH","DINO","DIOD","DK","DLB","DORM",
    "DOCS","DV","DXC","DY","EAT","EFSC","EHC","ELF","EMBC","EME","ENSG","ENVA","ENV","EPRT",
    "ERJ","ESTA","EVTC","EXEL","EXLS","EXPI","EYE","FAF","FARO","FBIN","FCEL","FCNCA","FDMT",
    "FG","FHI","FIGS","FIVE","FIX","FL","FLNC","FLYW","FNA","FNB","FNF","FOLD","FORM","FOXF",
    "FRHC","FRPT","FRSH","FSS","FTRE","FTDR","FULT","FUN","FYBR","G","GATX","GBCI","GBX","GEO",
    "GFF","GH","GHC","GLBE","GLNG","GLP","GMS","GNRC","GO","GPI","GSHD","GTLS","GTES","GTN",
    "GBTC","GPCR","HAYW","HBI","HEES","HGV","HI","HLMN","HLNE","HLI","HLIO","HL","HNI","HOPE",
    "HP","HQY","HRMY","HSAI","HUN","HURN","HXL","HYLN","IAC","IBKR","IBP","IBTX","ICL","ICUI",
    "IDCC","IFS","IGMS","IMAX","INDI","INSM","INST","INTA","INVA","IONQ","IOVA","IRDM","IRON",
    "ITCI","ITRI","IVT","JACK","JANX","JBT","JJSF","JOE","JOBY","KALU","KAR","KBH","KFY",
    "KGS","KN","KNSL","KNX","KNTK","KRYS","KTOS","KWR","LADR","LANC","LAUR","LBRT","LEA","LGIH",
    "LHCG","LITE","LIVN","LKFN","LMAT","LMND","LNTH","LNW","LOB","LPRO","LSCC","LSTR","LTCH",
    "MATX","MBIN","MCRI","MDGL","MEDP","MGEE","MGY","MHO","MIDD","MKSI","MLI","MMSI","MNTK",
    "MOD","MOG-A","MORF","MPLN","MQ","MRCY","MSTR","MTDR","MTSI","MTX","MUR","MWA","MXCT",
    "NABL","NATL","NBHC","NBR","NBTB","NEO","NEOG","NFE","NGS","NKLA","NNI","NOVT","NR","NSA",
    "NSIT","NUVB","NWE","NWL","NXT","OGN","OGS","OI","OLED","OLPX","OMF","ONB","ONTO","ORA",
    "ORI","OSH","OSIS","OUST","OUT","OVV","PAG","PARR","PAYO","PBF","PCRX","PDCO","PDFS","PEB",
    "PENN","PGNY","PIPR","PLAY","PLMR","PLXS","PNFP","PNM","POR","POST","POWL","PRCT","PRGS",
    "PRGO","PRK","PRMW","PROK","PTGX","PVH","QTWO","QUBT","R","RAMP","RBC","RCKT","RDFN",
    "RDN","RDNT","REZI","RGR","RGTI","RH","RHP","RIG","RIOT","RIVN","RKT","RLJ","RMBS","RNG",
    "RNST","ROAD","ROCK","ROG","RPD","RPM","RRC","RVLV","RXO","RWT","RXRX","SAIA","SATS","SBCF",
    "SBI","SBR","SCSC","SDGR","SEDG","SFBS","SGH","SHAK","SHOO","SIG","SITC","SITM","SKT","SKWD",
    "SKYT","SLVM","SM","SMMT","SN","SND","SNEX","SNV","SONO","SPHR","SPNT","SPSC","SPT","SPXC",
    "SRC","SRCL","SRRK","SSD","SSNC","SSRM","STEP","STNG","STRL","STWD","SUM","SUPN","SWIM",
    "SWX","TALO","TASK","TBBK","TCBI","TCBK","TDOC","TDS","TENB","TH","THC","TIGO","TKO","TKR",
    "TMHC","TNET","TOST","TREX","TRIP","TRUP","TSAT","TTGT","TTMI","TXRH","UCTT","UDMY","UFO",
    "UMBF","URBN","USM","USLM","VCEL","VCTR","VCYT","VEL","VERX","VIRT","VKTX","VMI","VNDA",
    "VNT","VPG","VRN","VRNS","VRNT","VSEC","VTOL","WD","WDFC","WERN","WGO","WHD","WINA","WING",
    "WK","WLK","WNS","WOLF","WOR","WRBY","WSC","WSFS","WULF","XNCR","XPO","XPOF","YETI","YELP",
    "YOU","ZETA","ZI","ZWS",
]

DOW30_TICKERS = [
    "AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS",
    "GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM","MRK",
    "MSFT","NKE","NVDA","PG","SHW","TRV","UNH","V","VZ","WMT",
]


def get_all_tickers():
    """Combine all hardcoded lists into one deduplicated universe."""
    all_tickers = sorted(set(SP500_TICKERS + NASDAQ100_EXTRA + DOW30_TICKERS + RUSSELL2000_TICKERS))
    return all_tickers


# ─── Technical helpers ────────────────────────────────────────────────────────

def compute_ema(closes, period):
    if len(closes) < period:
        return [None] * len(closes)
    mult = 2 / (period + 1)
    out = [None] * len(closes)
    out[period - 1] = np.mean(closes[:period])
    for i in range(period, len(closes)):
        out[i] = (closes[i] - out[i-1]) * mult + out[i-1]
    return out


def compute_sma(closes, period):
    out = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        out[i] = np.mean(closes[i - period + 1:i + 1])
    return out


def compute_wma(values, period):
    out = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        if any(v is None for v in window):
            continue
        weights = list(range(1, period + 1))
        out[i] = sum(w * v for w, v in zip(weights, window)) / sum(weights)
    return out


def compute_hma(closes, period):
    """Hull Moving Average."""
    half_p = max(period // 2, 1)
    half = compute_wma(closes, half_p)
    full = compute_wma(closes, period)
    diff = [None] * len(closes)
    for i in range(len(closes)):
        if half[i] is not None and full[i] is not None:
            diff[i] = 2.0 * half[i] - full[i]
    sqrt_period = max(int(np.sqrt(period)), 1)
    return compute_wma(diff, sqrt_period)


def compute_atr_series(highs, lows, closes, period=14):
    n = len(closes)
    out = [0.0] * n
    trs = []
    for i in range(1, n):
        try:
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        except (TypeError, ValueError):
            tr = 0.0
        trs.append(tr)
        if len(trs) <= period:
            out[i] = np.mean(trs)
        else:
            out[i] = (out[i-1] * (period - 1) + tr) / period
    return out


def compute_atr_avg(atr_series, period=50):
    """SMA of ATR series."""
    return compute_sma(atr_series, period)


# ─── VCP Coil Detection ──────────────────────────────────────────────────────

def scan_ticker(closes, highs, lows, opens, volumes):
    """
    Run VCP coil detection on a single ticker.
    Returns dict with booleans for each screener condition.
    """
    n = len(closes)
    if n < 210:  # Need enough data for SMA200
        return None

    # Moving averages
    ema3 = compute_ema(closes, 3)
    ema5 = compute_ema(closes, 5)
    ema9 = compute_ema(closes, 9)
    ema21 = compute_ema(closes, 21)
    sma10 = compute_sma(closes, 10)
    sma40 = compute_sma(closes, 40)
    sma50 = compute_sma(closes, 50)
    sma200 = compute_sma(closes, 200)
    hma40 = compute_hma(closes, 40)

    # ATR
    atr_series = compute_atr_series(highs, lows, closes, 14)
    atr_avg = compute_atr_avg(atr_series, 50)

    # Current bar values (last bar)
    i = n - 1
    if any(v is None for v in [ema9[i], ema21[i], sma50[i], sma200[i], hma40[i], atr_series[i], atr_avg[i]]):
        return None

    close = closes[i]
    high = highs[i]
    low = lows[i]
    opn = opens[i]
    atr = atr_series[i]
    atr_a = atr_avg[i]

    body_range = abs(close - opn)
    candle_range = abs(high - low)
    atr_buffer = 0.5 * atr

    # ATR contraction
    atr_contracted_str = (body_range < atr_a * 0.6) and (candle_range < atr_a * 0.85)
    atr_contracted_bal = candle_range < atr_a * 0.8
    atr_contracted_loo = body_range < atr_a * 1.5

    # MA spread
    spread_short = abs(ema9[i] - ema21[i]) / atr * 100 if atr > 0 else 999
    spread_long = abs(ema21[i] - hma40[i]) / atr * 100 if atr > 0 else 999
    ma_tight = spread_short < 50  # Balanced mode

    # Price proximity
    price_near_ema9 = abs(close - ema9[i]) <= atr_buffer
    price_near_ema21 = abs(close - ema21[i]) <= atr_buffer

    # RMV Calc #1
    atr3 = compute_atr_series(highs, lows, closes, 3)
    atr5 = compute_atr_series(highs, lows, closes, 5)
    atr8 = compute_atr_series(highs, lows, closes, 8)
    if atr3[i] is not None and atr5[i] is not None and atr8[i] is not None:
        short_avg = (atr3[i] + atr5[i] + atr8[i]) / 3
        # Look back 30 bars for highest/lowest
        short_avgs = []
        for j in range(max(0, i - 29), i + 1):
            if atr3[j] is not None and atr5[j] is not None and atr8[j] is not None:
                short_avgs.append((atr3[j] + atr5[j] + atr8[j]) / 3)
        if short_avgs:
            h_sa = max(short_avgs)
            l_sa = min(short_avgs)
            rmv = (short_avg - l_sa) / max(h_sa - l_sa, 0.001) * 100
        else:
            rmv = 50
    else:
        rmv = 50

    # RMV Calc #2 (alt)
    bar_ranges = [highs[j] - lows[j] for j in range(n)]
    if i >= 22:
        prev_ranges = bar_ranges[i-21:i]
        r_min = min(prev_ranges)
        r_max = max(prev_ranges)
        den = max(r_max - r_min, 1e-6)
        rmv_alt = max(0, min(100, 100 * (bar_ranges[i] - r_min) / den))
    else:
        rmv_alt = 50

    # ─── Coil conditions (Balanced mode — also includes strict) ───
    strict_coil = (atr_contracted_str
        and ((ema21[i] > sma50[i]) or (sma50[i] > sma200[i]))
        and (rmv < 21 or rmv_alt < 10)
        and (price_near_ema9 or price_near_ema21)
        and ((spread_short < 50 and spread_long < 50) or close > sma50[i] or ema5[i] > ema21[i]))

    balanced_coil = (atr_contracted_bal
        and ((ema21[i] > sma50[i]) or (sma50[i] > sma200[i]))
        and (rmv < 25 or rmv_alt < 20)
        and ma_tight
        and (price_near_ema9 or price_near_ema21))

    coil_detected = balanced_coil or strict_coil

    # ─── Coil box tracking (simplified — look back for recent coil start) ───
    # Find the most recent coil box by scanning backwards
    coil_high = None
    coil_low = None
    coil_active = False

    for j in range(i, max(i - 10, 0), -1):
        if j < 1:
            break
        bj = abs(closes[j] - opens[j])
        cj = abs(highs[j] - lows[j])
        aa = atr_avg[j] if atr_avg[j] is not None else atr_a

        contracted = cj < aa * 0.8
        e9 = ema9[j] if ema9[j] is not None else 0
        e21 = ema21[j] if ema21[j] is not None else 0
        s50 = sma50[j] if sma50[j] is not None else 0
        s200 = sma200[j] if sma200[j] is not None else 0
        atr_j = atr_series[j] if atr_series[j] is not None else atr

        sp = abs(e9 - e21) / atr_j * 100 if atr_j > 0 else 999
        near9 = abs(closes[j] - e9) <= 0.5 * atr_j
        near21 = abs(closes[j] - e21) <= 0.5 * atr_j
        rmv_ok = True  # simplified for lookback

        bal_j = contracted and ((e21 > s50) or (s50 > s200)) and sp < 50 and (near9 or near21)
        if bal_j:
            if coil_high is None:
                coil_high = highs[j]
                coil_low = lows[j]
            else:
                coil_high = max(coil_high, highs[j])
                coil_low = min(coil_low, lows[j])
            coil_active = True
        elif coil_active:
            break  # End of contiguous coil region

    # ─── Breakout detection (daily close > coil high, no 15m needed) ───
    breakout = False
    if coil_active and coil_high is not None:
        breakout = (close > coil_high) and (close > sma50[i]) and (close > sma200[i])

    # ─── Inside box detection ───
    inside_box = False
    if coil_active and coil_high is not None and coil_low is not None:
        box_80 = coil_low + 0.80 * (coil_high - coil_low)
        open_in = (opn >= coil_low) and (opn <= coil_high)
        close_in_80 = (close >= coil_low) and (close <= box_80)
        inside_box = open_in and close_in_80

    # ─── Filter conditions ───
    above_50sma = close > sma50[i]
    above_200sma = close > sma200[i]
    above_21ema = close > ema21[i]

    return {
        "coil": bool(coil_detected),
        "breakout": bool(breakout),
        "inside": bool(inside_box),
        "a50": bool(above_50sma),
        "a200": bool(above_200sma),
        "a21": bool(above_21ema),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Stock Screener — VCP Coil Detection")
    print("=" * 60)

    # Gather universe
    print("\nStep 1: Building stock universe...")
    all_tickers = get_all_tickers()
    print(f"  Total unique tickers: {len(all_tickers)}")

    if not all_tickers:
        print("ERROR: No tickers found")
        sys.exit(1)

    # Download data
    print(f"\nStep 2: Downloading price data for {len(all_tickers)} tickers...")
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=365)

    raw = yf.download(
        all_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        threads=True,
    )

    if raw.empty:
        print("ERROR: No data returned")
        sys.exit(1)

    def get_ohlcv(ticker):
        try:
            if len(all_tickers) == 1:
                df = raw.dropna(subset=["Close"])
            else:
                df = raw[ticker].dropna(subset=["Close"])
            if len(df) < 210:
                return None
            return {
                "c": df["Close"].values.tolist(),
                "h": df["High"].values.tolist(),
                "l": df["Low"].values.tolist(),
                "o": df["Open"].values.tolist(),
                "v": df["Volume"].values.tolist(),
            }
        except Exception:
            return None

    # Scan
    print(f"\nStep 3: Scanning {len(all_tickers)} tickers for VCP coils...")
    results = {"coil": [], "breakout": [], "inside": []}
    scanned = 0
    for i, tk in enumerate(all_tickers):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  {i+1}/{len(all_tickers)}...")
        data = get_ohlcv(tk)
        if data is None:
            continue
        scanned += 1
        try:
            scan = scan_ticker(data["c"], data["h"], data["l"], data["o"], data["v"])
        except Exception as ex:
            if (i + 1) % 200 == 0:
                print(f"    Error scanning {tk}: {ex}")
            continue
        if scan is None:
            continue

        entry = {"t": tk, "a50": scan["a50"], "a200": scan["a200"], "a21": scan["a21"]}

        if scan["coil"]:
            results["coil"].append(entry)
        if scan["breakout"]:
            results["breakout"].append(entry)
        if scan["inside"]:
            results["inside"].append(entry)

    print(f"\nResults:")
    print(f"  Scanned: {scanned}")
    print(f"  Coil Active: {len(results['coil'])}")
    print(f"  Coil Breakout: {len(results['breakout'])}")
    print(f"  Inside Box: {len(results['inside'])}")

    output = {
        "results": results,
        "meta": {
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "scanned": scanned,
            "universe": len(all_tickers),
        }
    }

    with open("screener.json", "w") as f:
        json.dump(output, f, separators=(",", ":"))

    print(f"\nWritten screener.json")


if __name__ == "__main__":
    main()
