#!/usr/bin/env python3
"""
Grade individual holdings by moving average structure and write grades.json.
Also fetches dynamic holdings for FFTY and BUZZ from web sources and writes
dynamic_holdings.json for use by fetch_data.py.

Grading criteria:
  Gold:   EMA9 > EMA21 > SMA50 AND Price > EMA21 AND Price > SMA200
  Silver: EMA9 > EMA21 but does not meet all gold criteria
  Bronze: EMA9 < EMA21 and/or does not meet gold or silver

Usage:
    pip install yfinance numpy requests beautifulsoup4
    python grade_holdings.py
"""

import json
import os
import time
import re
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf

# Use curl_cffi to bypass Yahoo Finance rate limiting (impersonates Chrome)
try:
    from curl_cffi import requests as cffi_requests
    _session = cffi_requests.Session(impersonate="chrome")
except ImportError:
    _session = None

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ─── All ETF holdings (must match fetch_data.py) ─────────────────────────

ETF_INFO = [
    {"t":"MAGS","h":"GOOGL,AMZN,AAPL,META,MSFT,NVDA,TSLA"},
    {"t":"XOP","h":"APA,AR,CLMT,CNX,COP,CTRA,CVX,DINO,DK,DVN,EOG,EQT,EXE,FANG,GPOR,MGY,MPC,MUR,OVV,OXY,PARR,PBF,PR,PSX,RRC,SHEL,VLO,VNOM,XOM"},
    {"t":"REMX","h":"ALB,SQM,LAC,MP,UUUU,USAR,SGML,DNN,UEC,CCJ,LAR,IDR,IPX,TROX"},
    {"t":"ITA","h":"ACHR,ATI,AVAV,AXON,BA,BWXT,CRS,CW,GD,GE,HEI,HII,HWM,HXL,KRMN,KTOS,LHX,LMT,NOC,RKLB,RTX,SARO,TDG,TXT,WWD"},
    {"t":"FCG","h":"AR,EXE,DVN,EQT,FANG,CTRA,APA,EOG,OXY,SM,COP,HESM"},
    {"t":"BOAT","h":"FRO,MATX,STNG,INSW,ZIM,SBLK,CMDB,DHT,CMRE,DAC,TNK,NAT"},
    {"t":"XLE","h":"APA,BKR,COP,CTRA,CVX,DVN,EOG,EQT,EXE,FANG,HAL,HES,KMI,MPC,OKE,OXY,PSX,SHEL,SLB,TPL,TRGP,VLO,WMB,XOM"},
    {"t":"FFTY","h":"ANAB,CLS,ARQT,STOK,FIX,MU,AU,TARS,APH,TVTX,AGI,KGC,GH,VRT,AEM,LLY,HWM,ARGX,ONC,WGS,ZYME,FN,TMDX"},
    {"t":"AMLP","h":"MPLX,WES,EPD,SUN,PAA,ET,HESM,CQP,USAC,GEL,SPH,GLP,DKL"},
    {"t":"IYZ","h":"AAOI,ANET,ASTS,CALX,CCOI,CHTR,CIEN,CMCSA,COMM,CSCO,DIS,EA,EXTR,FFIV,FOX,FOXA,FYBR,GLIBK,GOOGL,GSAT,IPG,IRDM,LITE,LUMN,LYV,META,MSI,MTCH,NFLX,NTCT,NWS,NWSA,NYT,OMC,ONDS,PARA,ROKU,T,TDS,TIGO,TKO,TMUS,TSAT,TTWO,USM,VIAV,VZ,WBD,ZBRA"},
    {"t":"XLI","h":"GE,CAT,RTX,UBER,GEV,BA,UNP,ETN,HON,DE,PH,ADP,TT,MMM,LMT,GD,HWM,WM,TDG,JCI,EMR,NOC,UPS,CMI,PWR,LUV,UAL,PAYX,J,FDX,HII,VLTO,ROK,EFX,VRSK,AME"},
    {"t":"FAN","h":"CWEN,NEE,ACA,LNT,TKR"},
    {"t":"IYT","h":"UBER,UNP,UPS,FDX,CSX,NSC,DAL,UAL,ODFL,EXPD,CHRW,LUV,XPO,JBHT,AAL,LYFT,SAIA,KNX,JOBY,R,KEX,GXO,ALK,LSTR,MATX,HUBG,SNDR,WERN"},
    {"t":"XES","h":"LBRT,HP,RIG,WFRD,SEI,NOV,HAL,FTI,VAL,PTEN,BKR,KGS,NE,SLB,WHD,AROC,TDW,SDRL,OII,XPRO,PUMP,AESI,NBR,HLX,WTTR"},
    {"t":"GDX","h":"PAAS,WPM,KGC,AGI,FNV,AEM,B,GFI,NEM"},
    {"t":"GDXJ","h":"PAAS,KGC,EQX,AGI,HMY,IAG,RGLD,USAU,GORO,CDE,GLDG"},
    {"t":"SIL","h":"EXK,AG,PAAS,HL,FSM,WPM,HCHDF,OR,SVM,SSRM,CDE"},
    {"t":"SILJ","h":"AG,HYMC,HL,PAAS,OR,FSM,EXK,GORO,SSRM,CDE"},
    {"t":"XLU","h":"NEE,SO,CEG,DUK,AEP,SRE,D,VST,EXC,XEL,ETR,PEG,ED,PCG,WEC,NRG,DTE,AEE,ATO"},
    {"t":"AIPO","h":"PWR,VRT,GEV,ETN,CCJ,AVGO,NVDA,CEG,BE,HUBB,FLNC,AMD,OKLO,MTZ,VST,NVT,EOSE,BWXT"},
    {"t":"XHS","h":"GH,PACS,BKD,NEO,CAH,WGS,UHS,BTSG,HCA,SEM,COR,MCK,CNC,ALHC,RDNT,THC,HQY,PGNY,ENSG,ADUS,HSIC,OPCH,ELV,CVS,PRVA"},
    {"t":"OIH","h":"RIG,WFRD,FTI,NE,BKR,NOV,VAL,TS,SLB,PUMP,HAL"},
    {"t":"KIE","h":"LMND,BHF,MCY,WTM,RYAN,ORI,L,RNR,AGO,CINF,MKL,AIZ,TRV,CB,PFG,AXS,ALL,THG,CNO,HIG,PLMR,ACGL,PRU,AFL,UNM"},
    {"t":"SLX","h":"VALE,STLD,NUE,MT,CLF,ATI,RIO,HCC,CRS,PKX"},
    {"t":"XHE","h":"TNDM,AXGN,INSP,GMED,HAE,ATEC,TMDX,ISRG,GKOS,SOLV,COO,IDXX,HOLX,OMCL,LNTH,MDT,LIVN,ALGN,UFPT,ICUI,EW,STE,PEN,NEOG,GEHC"},
    {"t":"XLV","h":"LLY,JNJ,ABBV,UNH,MRK,ABT,TMO,ISRG,AMGN,GILD,BSX,PFE,DHR,MDT,SYK,VRTX,MCK,CVS,BMY,HCA,REGN,ELV,CI,COR,IDXX,CNC,WST,BDX,EW,CAH,LH,ZBH,DXCM,DVA,STE,IQV,DGX"},
    {"t":"COPX","h":"ERO,TGB,FCX,HBM,IE,TECK,SCCO"},
    {"t":"DRNZ","h":"ACHR,AVAV,DPRO,DRSHF,EH,ELS,JOBY,KTOS,ONDS,PLTR,RCAT,UMAC"},
    {"t":"URA","h":"ASPI,BWXT,CCJ,CEG,DNN,EU,LEU,LTBR,MIR,NNE,NXE,OKLO,SBSW,SMR,SRUUF,TLN,UEC,URNJ,URNM,UROY,UUUU"},
    {"t":"UFO","h":"ASTS,BA,BKSY,CMCSA,FLY,GE,GEMI,GRMN,GSAT,HON,IRDM,LHX,LMT,LUNR,NOC,PL,RDW,RKLB,RTX,SATS,SIRI,TRMB,VOYG,VSAT"},
    {"t":"PAVE","h":"HWM,PWR,PH,CRH,SRE,NSC,FAST,TT,CSX,ROK,URI,EMR,DE,ETN,UNP,VMC,MLM,NUE,EME,STLD,HUBB,TRMB,FTV,WWD,PNR"},
    {"t":"LIT","h":"RIO,ALB,LAR,SEI,MVST,ENS,EOSE,SQM,AMPX,TSLA,BLDP,SLDP,ABAT,SGML,SLI,LAC"},
    {"t":"XLB","h":"LIN,NEM,SHW,ECL,NUE,FCX,MLM,VMC,APD,CTVA,STLD,PPG,IP,AMCR,PKG,IFF,DOW,DD,ALB,AVY,BALL,CF,LYB,MOS,CE,EMN,FMC"},
    {"t":"XPH","h":"LQDA,MBX,OGN,AXSM,EWTX,ELAN,VTRS,AMLX,AMRX,MRK,CRNX,XERS,LGND,JNJ,LLY,PBH,BMY,PRGO,SUPN,RPRX,ZTS,JAZZ,AVDL,PFE"},
    {"t":"PBJ","h":"MNST,HSY,KR,SYY,MDLZ,KHC,CTVA,DASH,HLF,UNFI,SEB,IMKTA,TSN,USFD,FDP,CART,CHEF,ADM,AGRO,TR,ACI,DPZ,POST,JBS,WMK"},
    {"t":"XLP","h":"ADM,CAG,CHD,CL,CLX,COST,CPB,DG,DLTR,EL,GIS,HRL,HSY,K,KDP,KHC,KMB,KO,KR,KVUE,MDLZ,MNST,MO,PEP,PG,PM,SJM,SYY,TGT,WMT"},
    {"t":"QQQE","h":"AAPL,ADI,AEP,AMAT,AMD,AMGN,ANET,ANSS,AVGO,AZN,BIIB,CDNS,CDW,CSCO,CTSH,DDOG,DELL,EA,FFIV,GILD,GLW,HPE,IDXX,INTC,ISRG,JBL,JNPR,KLAC,LRCX,MAR,MNST,MPWR,MRVL,MU,NVDA,PLTR,QCOM,REGN,ROST,SWKS,TDY,TER,TXN,VRSN,VRTX,WBD,XEL"},
    {"t":"ROBO","h":"SYM,TER,ISRG,SERV,IRBT,COHR,ROK,ILMN,RR,ARBE,AUR,GMED,PRCT,NOVT,PDYN,IPGP,NDSN,EMR"},
    {"t":"ARKK","h":"TSLA,ROKU,COIN,TEM,CRSP,SHOP,HOOD,RBLX,AMD,PLTR,BEAM,TER,CRCL,BMNR,ACHR,TXG,TWST,ILMN,AMZN,VCYT,BLSH,NVDA,NTRA,META,DKNG"},
    {"t":"HYDR","h":"BE,PLUG,BLDP,SLDP,FCEL,CMI,APD"},
    {"t":"PEJ","h":"WBD,LVS,SYY,CCL,DASH,LYV,RCL,FLUT,LYFT,LION,EXPE,FOXA,CNK,TKO,CPA,USFD,PSKY,CART,BYD,EAT,BH,RRR,DPZ,MGM,ACEL"},
    {"t":"EEM","h":"PDD,BABA,NU,MELI,DLO,JMIA"},
    {"t":"IWO","h":"CRDO,BE,FN,IONQ,GH,KTOS,BBIO,MDGL,ENSG,NXT,RMBS,SPXC,DY,STRL,GTLS,IDCC,HQY,MOD,AVAV,AEIS,WTS,RGTI,ZWS,HIMS,LUMN"},
    {"t":"MOO","h":"DE,ZTS,CTVA,ADM,TSN,CF,BG,ELAN,MOS,CNH,DAR,TTC,AGCO,CAT"},
    {"t":"XME","h":"HL,AA,HCC,STLD,NEM,LEU,CDE,NUE,CLF,CMC,RGLD,BTU,CNR,RS,FCX,UEC,MP,AMR,MTRN,CENX,IE,KALU,USAR,WS,MUX"},
    {"t":"ARKF","h":"SHOP,COIN,HOOD,PLTR,TOST,SOFI,XYZ,RBLX,ROKU,CRCL,MELI,AMD,DKNG,META,AMZN,BMNR,PINS,NU,KLAR,SE,BLSH,FUTU,Z"},
    {"t":"EWZ","h":"NU,MELI,DLO"},
    {"t":"XBI","h":"EXAS,RVMD,RNA,INSM,NTRA,BBIO,REGN,MDGL,IONS,BIIB,AMGN,UTHR,INCY,ROIV,EXEL,VRTX,GILD,NBIX,ABBV,BMRN,CRSP,MRNA,PTCT,ALNY,KRYS"},
    {"t":"SMH","h":"MU,INTC,RGTI,AMD,MRVL,MTSI,FSLR,RMBS,SITM,SMTC,MPWR,ADI,QCOM,CRUS,ON,AVGO,CRDO,LSCC,NVDA,QRVO,SLAB,TXN,NXPI,SWKS,OLED,TSEM,TSM,AMAT,ASML,LRCX,KLAC,CDNS,SNPS,TER,MCHP,STM"},
    {"t":"ARKG","h":"TEM,CRSP,PSNL,GH,TWST,TXG,NTRA,BEAM,ILMN,VCYT,RXRX,ADPT,IONS,ABSI,CDNA,SDGR,NTLA,NRIX,PACB,WGS,BFLY,PRME,ARCT,AMGN"},
    {"t":"GBTC","h":"IBIT,ETHA,MSTR,BMNR,SBET,COIN"},
    {"t":"IGV","h":"ADBE,ADSK,AGYS,APP,APPN,BBAI,CDNS,CIFR,CLSK,CRM,CRNC,CRWD,CTSH,DDOG,EA,EPAM,FICO,FTNT,GDYN,HUT,IBM,IDCC,INTU,JAMF,MSFT,MSTR,NOW,ORCL,PANW,PATH,PLTR,PRO,PTC,QBTS,ROP,SEMR,SNPS,TDC,TEAM,TTWO,WDAY,WK,WULF,ZM,ZS"},
    {"t":"PHO","h":"WAT,FERG,ECL,ROP,AWK,MLI,IEX,WMS,XYL,PNR,VLTO,AOS,ACM,CNM,VMI,WTRG,BMI,TTEK,ITRI,WTS,ZWS,MWA,SBS,FELE,HWKN"},
    {"t":"BLOK","h":"BBBY,BITB,BKKT,BLK,CAN,CIFR,CLSK,CME,CMPO,COIN,CORZ,CRCL,FBTC,FIGR,GLXY,HOOD,HUT,IBIT,IBM,MELI,NU,OPRA,PYPL,RBLX,WULF,XYZ"},
    {"t":"WCLD","h":"FSLY,SEMR,MDB,DOCN,FROG,PATH,SNOW,BILL,DDOG,CFLT,WK,TWLO,CRWD,PCOR,AGYS,IOT,BRZE,QLYS,CWAN,BL,CLBT,PANW,SHOP,INTA,NET"},
    {"t":"WGMI","h":"APLD,BITF,BTBT,BTDR,CAN,CANG,CIFR,CLSK,CORZ,CRWV,GLXY,HIVE,HUT,IREN,MARA,NBIS,NVDA,RIOT,TSM,WULF,XYZ"},
    {"t":"IBUY","h":"FIGS,LQDT,CVNA,UPWK,EXPE,CART,W,RVLV,CHWY,LYFT,MSM,EBAY,BKNG,AFRM,SPOT,TRIP,ABNB,PTON,UBER,AMZN,CPRT,PYPL,HIMS,SSTK,ETSY"},
    {"t":"TAN","h":"NXT,FSLR,RUN,ENPH,SEDG,HASI,CSIQ,CWEN,DQ,SHLS,ARRY,JKS"},
    {"t":"KCE","h":"AMG,IVZ,MS,CBOE,BK,SF,CME,STT,HOOD,LPLA,GS,NTRS,IBKR,STEP,SCHW,MSCI,PIPR,JHG,TPG,MCO,EVR,TROW,GLXY,FHI,NDAQ"},
    {"t":"IPAY","h":"AXP,V,MA,PYPL,CPAY,FIS,AFRM,XYZ,GPN,COIN,TOST,FISV,FOUR,WEX,QTWO,STNE,ACIW,EEFT,WU,RELY"},
    {"t":"ITB","h":"DHI,LEN,PHM,NVR,TOL,SHW,LOW,BLD,HD,LII,MAS,BLDR,TMHC,IBP,MTH,OC,SKY,CVCO,KBH,EXP,SSD,FND,MHO,FBIN,MHK,SGI,WMS,JCI,TT,ALLE,WSM,CARR,CSL"},
    {"t":"WOOD","h":"PCH,RYN,SLVM,WY,SW,IP,CLW"},
    {"t":"ICLN","h":"AEIS,ALB,AMRC,BE,BEPC,CSIQ,CWEN,ENPH,EOSE,FLNC,FSLR,JKS,LAC,LAR,MYRG,NVTS,NXT,ORA,PLUG,PWR,RIVN,RUN,SGML,SHLS,SLDP,SLI,SQM,TE"},
    {"t":"KRE","h":"CADE,VLY,COLB,CFG,TFC,PB,FNB,EWBC,FHN,FLG,WBS,ONB,MTB,PNFP,RF,HBAN,ZION,BPOP,WAL,UMBF,WTFC,SSB,CFR,HWC"},
    {"t":"CIBR","h":"AVGO,CRWD,CSCO,INFY,PANW,LDOS,FTNT,CYBR,CHKP,NET,ZS,GEN,AKAM,FFIV,OKTA,BAH,RBRK,S,CVLT,QLYS,SAIC,VRNS"},
    {"t":"XRT","h":"VSCO,REAL,KSS,M,ODP,EYE,DDS,ROST,GAP,DLTR,WMT,URBN,SBH,FIVE,PSMT,AEO,TJX,RVLV,ULTA,BOOT,ANF,CASY,SIG,CVNA,DG"},
    {"t":"XLF","h":"AFL,AIG,AIZ,AJG,ALL,AON,AXP,BAC,BK,BLK,BRK.B,BX,C,CB,CME,COF,EG,ERIE,FDS,FI,FICO,GL,GS,HOOD,ICE,JPM,KKR,L,MA,MET,MMC,MS,PGR,PNC,PRU,SCHW,SPGI,STT,USB,V,WFC,WTW"},
    {"t":"ESPO","h":"EA,NTES,TTWO,RBLX,SE,U,LOGI,GME,PLTK,CRSR,SONY"},
    {"t":"JETS","h":"LUV,AAL,DAL,UAL,ALGT,SNCY,SKYW,ULCC,JBLU,EXPE,GD,ALK,TXT,SABR,BKNG,TRIP,BA,RYAAY"},
    {"t":"KBE","h":"CMA,BKU,BANC,EBC,PFSI,CADE,BK,VLY,COLB,WFC,BAC,INDB,C,FBK,CFG,TCBI,BOKF,SBCF,TFC,PB,NTRS,ABCB,FIBK,JPM,WSBC"},
    {"t":"FXI","h":"BABA,TCEHY,NTES,BYDDF,TCOM,JD,BIDU,PTR,SNP,FUTU,KWEB"},
    {"t":"XLY","h":"ABNB,AMZN,APTV,AZO,BBY,BKNG,CMG,DASH,DECK,DHI,DRI,EBAY,EXPE,F,GM,GRMN,HD,HLT,KMX,LKQ,LOW,LULU,MAR,MCD,NKE,ORLY,RCL,RL,ROST,SBUX,TJX,TPR,TSLA,ULTA,YUM"},
    {"t":"MSOS","h":"VFF,MNMD,TLRY,MO,CRON,GTBIF,TCNNF"},
    {"t":"BJK","h":"LVS,FLUT,VICI,DKNG,WYNN,CHDN,GLPI,MGM,BYD"},
    # Custom baskets
    {"t":"NURS","h":"LH,FMS,ENSG,SOLV,GH,EHC,DVA,BTSG,RDNT,OPCH,HIMS,BLLN,CON,LFST,WGS"},
    {"t":"CHEMG","h":"NTR,CF,MOS,ICL,SMG,FMC,UAN,LXU"},
    {"t":"CHEMS","h":"LIN,ECL,APD,DOW,LYB,ALB,SQM,WLK,EMN,NEU,CE,MEOH,CBT,HWKN,CC,KWR,GEL,MTX,ROG,SHW,DD,SOLS,OLN,AXTA"},
    {"t":"QNTM","h":"SKYT,ARQQ,QSI,QUBT,QBTS,RGTI,IONQ"},
    {"t":"COOL","h":"FIX,EME,JCI,TT,WSO,VRT,CARR,LII,SPXC,AAON"},
    {"t":"AGENT","h":"GTLB,ASAN,PATH,BRZE,VERI,AI,SOUN,BBAI,LAW,CRNC"},
    {"t":"OPTIC","h":"LITE,COHR,AAOI,POET,ALMU,LWLG,MTSI,GLW,FN,GFS,TSEM"},
    {"t":"LIDAR","h":"OUST,AEVA,HSAI,TDY"},
    {"t":"IYR","h":"WELL,PLD,EQIX,O,AMT,SPG,DLR,PSA,CBRE,VTR,CCI,VICI,EXR,IRM"},
]


# ─── Dynamic Holdings Scraper ──────────────────────────────────────────────

def fetch_holdings_from_perplexity(ticker):
    """Try to fetch holdings from Perplexity Finance. Returns list of tickers or None."""
    if not HAS_REQUESTS:
        print(f"  requests/beautifulsoup4 not available, skipping {ticker} scrape")
        return None
    url = f"https://www.perplexity.ai/finance/{ticker.lower()}/holdings"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"  Perplexity returned {resp.status_code} for {ticker}")
            return None
        # Try to extract holdings from HTML
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text()
        # Also try JSON in script tags
        for script in soup.find_all("script"):
            if script.string and "holdings" in script.string.lower():
                # Try to parse embedded JSON
                try:
                    data = json.loads(script.string)
                    # Look for holdings array
                    if isinstance(data, dict):
                        for key in data:
                            if "holding" in key.lower():
                                return _extract_tickers_from_data(data[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        print(f"  Could not parse holdings from Perplexity for {ticker}")
        return None
    except Exception as e:
        print(f"  Error fetching {ticker} from Perplexity: {e}")
        return None


def fetch_holdings_from_yfinance(ticker, min_weight=1.0):
    """Fallback: try yfinance to get ETF holdings."""
    try:
        etf = yf.Ticker(ticker, session=_session)
        # Try the holdings property (available in newer yfinance)
        try:
            holdings = etf.get_institutional_holders()
        except Exception:
            holdings = None

        if holdings is not None and not holdings.empty:
            tickers = []
            for _, row in holdings.iterrows():
                if hasattr(row, "Symbol") and row.get("% Out", 0) > min_weight:
                    tickers.append(row["Symbol"])
            if tickers:
                return tickers

        # Try fund_holding_info
        try:
            info = etf.get_fund_holding_info()
            if info and "holdings" in info:
                return [h["symbol"] for h in info["holdings"]
                        if h.get("holdingPercent", 0) > min_weight / 100]
        except Exception:
            pass

        return None
    except Exception as e:
        print(f"  yfinance fallback failed for {ticker}: {e}")
        return None


def update_dynamic_holdings():
    """Fetch current FFTY and BUZZ holdings and write dynamic_holdings.json."""
    dynamic = {}
    for ticker in ["FFTY", "BUZZ"]:
        print(f"\nFetching dynamic holdings for {ticker}...")

        # Try Perplexity first
        holdings = fetch_holdings_from_perplexity(ticker)

        # Fallback to yfinance
        if not holdings:
            print(f"  Trying yfinance fallback for {ticker}...")
            holdings = fetch_holdings_from_yfinance(ticker, min_weight=1.0)

        if holdings:
            # Filter: keep only valid-looking tickers (1-5 uppercase letters)
            clean = [h.upper().strip() for h in holdings
                     if re.match(r'^[A-Z]{1,5}$', h.strip().upper())]
            if clean:
                dynamic[ticker] = ",".join(sorted(set(clean)))
                print(f"  Got {len(clean)} holdings for {ticker}: {dynamic[ticker][:80]}...")
            else:
                print(f"  No valid tickers found for {ticker}")
        else:
            print(f"  Could not fetch holdings for {ticker} - keeping existing")

    # Write dynamic_holdings.json
    with open("dynamic_holdings.json", "w") as f:
        json.dump(dynamic, f, indent=2)
    print(f"\nWritten dynamic_holdings.json with {len(dynamic)} entries")
    return dynamic


# ─── MA Helpers ────────────────────────────────────────────────────────────

def compute_ema(closes, period):
    if len(closes) < period:
        return None
    mult = 2 / (period + 1)
    ema = np.mean(closes[:period])
    for c in closes[period:]:
        ema = (c - ema) * mult + ema
    return ema


def compute_sma(closes, period):
    if len(closes) < period:
        return None
    return np.mean(closes[-period:])


def grade_holding(price, ema9, ema21, sma50, sma200):
    if any(v is None for v in [price, ema9, ema21, sma50, sma200]):
        return "b"
    # Gold: EMA9 > EMA21 > SMA50 AND Price > EMA21 AND Price > SMA200
    if ema9 > ema21 and ema21 > sma50 and price > ema21 and price > sma200:
        return "g"
    # Silver path 1: EMA9 > EMA21 but doesn't meet all gold criteria
    if ema9 > ema21:
        return "s"
    # Silver path 2: EMA9 within 2% of EMA21 (just below) AND Price > SMA50 AND Price > SMA200 AND EMA21 > SMA50
    if ema21 > 0 and (ema21 - ema9) / ema21 <= 0.02 and price > sma50 and price > sma200 and ema21 > sma50:
        return "s"
    return "b"


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    # Step 1: Update FFTY/BUZZ holdings
    print("=" * 60)
    print("Step 1: Fetching dynamic holdings for FFTY/BUZZ")
    print("=" * 60)
    update_dynamic_holdings()

    # Step 2: Collect all unique holdings (including from dynamic file)
    print("\n" + "=" * 60)
    print("Step 2: Grading all holdings")
    print("=" * 60)

    # Load dynamic holdings to include in grading
    dyn = {}
    try:
        with open("dynamic_holdings.json") as f:
            dyn = json.load(f)
    except Exception:
        pass

    all_holdings = set()
    for e in ETF_INFO:
        h_str = e.get("h", "")
        # Override with dynamic if available
        if e["t"] in dyn and dyn[e["t"]]:
            h_str = dyn[e["t"]]
        if h_str:
            for hh in h_str.split(","):
                hh = hh.strip()
                if hh:
                    all_holdings.add(hh)

    holding_tickers = sorted(list(all_holdings))
    print(f"Grading {len(holding_tickers)} unique holdings...")

    end = datetime.now()
    h_start = end - timedelta(days=365)

    holding_grades = {}
    holding_names = {}
    graded = 0
    failed = 0
    fmp_fallback_count = 0

    # FMP API key for fallback (optional — from environment)
    fmp_api_key = os.environ.get("FMP_API_KEY", "")
    if fmp_api_key:
        print(f"FMP API key found — will use as fallback for insufficient yfinance data")

    def fetch_closes_from_fmp(ticker, days=400):
        """Fetch adjusted close prices from FMP as fallback."""
        if not fmp_api_key:
            return None
        try:
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries={days}&apikey={fmp_api_key}"
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                return None
            data = resp.json()
            hist = data.get("historical", [])
            if not hist:
                return None
            closes = [d["adjClose"] for d in reversed(hist) if "adjClose" in d]
            return closes if len(closes) >= 50 else None
        except Exception:
            return None

    # ─── Bulk download all tickers at once (avoids rate limiting) ──
    print(f"Bulk downloading {len(holding_tickers)} tickers from Yahoo Finance...")
    raw = yf.download(
        holding_tickers,
        start=h_start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        threads=False,
        session=_session,
    )

    def get_closes(ticker):
        try:
            if len(holding_tickers) == 1:
                df = raw.dropna(subset=["Close"])
            else:
                df = raw[ticker].dropna(subset=["Close"])
            return df["Close"].values.tolist()
        except Exception:
            return []

    # ─── Fetch names individually (uses yf.Ticker.info, slower) ──
    print("Fetching company names and descriptions...")
    for i, tk in enumerate(holding_tickers):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Names: {i+1}/{len(holding_tickers)}...")
        try:
            ticker_obj = yf.Ticker(tk, session=_session)
            info = ticker_obj.info
            name = info.get("shortName") or info.get("longName") or ""
            summary = info.get("longBusinessSummary") or ""
            if summary:
                sentences = summary.split('. ')
                short = ''
                for s in sentences:
                    candidate = short + s + '. ' if short else s + '. '
                    if len(candidate) > 500:
                        break
                    short = candidate
                if not short:
                    short = summary[:497] + '...'
                summary = short.strip()
            if name:
                holding_names[tk] = {"n": name, "d": summary} if summary else {"n": name}
        except Exception:
            pass
        if (i + 1) % 10 == 0:
            time.sleep(0.3)

    # ─── Grade each ticker ──
    print("Grading tickers...")
    for i, tk in enumerate(holding_tickers):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Grading: {i+1}/{len(holding_tickers)}...")

        closes = get_closes(tk)

        # If yfinance returned insufficient data for SMA200, try FMP fallback
        if len(closes) < 200:
            fmp_closes = fetch_closes_from_fmp(tk)
            if fmp_closes and len(fmp_closes) > len(closes):
                print(f"    FMP fallback for {tk}: yf={len(closes)} pts -> fmp={len(fmp_closes)} pts")
                closes = fmp_closes
                fmp_fallback_count += 1

        if len(closes) < 50:
            holding_grades[tk] = "b"
            failed += 1
            continue

        ema9 = compute_ema(closes, 9)
        ema21 = compute_ema(closes, 21)
        sma50 = compute_sma(closes, 50)
        sma200 = compute_sma(closes, 200)
        price = closes[-1]

        grade = grade_holding(price, ema9, ema21, sma50, sma200)
        holding_grades[tk] = grade
        graded += 1

        if tk in ("T", "CMCSA", "BAC", "AAPL", "XOM", "WES", "CWEN", "ATI"):
            print(f"    DEBUG {tk}: Price={price:.2f} EMA9={ema9:.2f} EMA21={ema21:.2f} "
                  f"SMA50={sma50:.2f} SMA200={sma200 if sma200 is not None else 'N/A'} pts={len(closes)} -> {grade}")

    with open("grades.json", "w") as f:
        json.dump(holding_grades, f, separators=(",", ":"))

    with open("names.json", "w") as f:
        json.dump(holding_names, f, separators=(",", ":"))

    g_count = sum(1 for v in holding_grades.values() if v == "g")
    s_count = sum(1 for v in holding_grades.values() if v == "s")
    b_count = sum(1 for v in holding_grades.values() if v == "b")

    print(f"\nWritten grades.json — {g_count} gold, {s_count} silver, {b_count} bronze")
    print(f"Written names.json — {len(holding_names)} company names")
    print(f"Graded: {graded}, Failed/insufficient: {failed}, FMP fallbacks: {fmp_fallback_count}")


if __name__ == "__main__":
    main()
