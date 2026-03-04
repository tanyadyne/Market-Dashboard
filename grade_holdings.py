#!/usr/bin/env python3
"""
Grade individual holdings by moving average structure and write grades.json.
Runs once daily after market close — results cached for the fast ETF refresh.

Grading criteria:
  Gold:   EMA9 > EMA21 > SMA50 AND Price > EMA21 AND Price > SMA200
  Silver: EMA9 > EMA21 but does not meet all gold criteria
  Bronze: EMA9 < EMA21 and/or does not meet gold or silver

Usage:
    pip install yfinance numpy
    python grade_holdings.py
"""

import json
import time
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf

# ─── All ETF holdings (same list as fetch_data.py) ─────────────────────────

ETF_INFO = [
    {"t":"XTL","h":"LITE,CIEN,ASTS,ONDS,LUMN,VIAV,COMM,GSAT,VSAT,CSCO,TDS,FYBR,VZ,UI,IRDM,AAOI,T,CALX,TMUS,ANET,MSI,FFIV,EXTR,NTCT,CCOI"},
    {"t":"XOP","h":"CNX,EXE,MUR,GPOR,EQT,VLO,RRC,FANG,CTRA,AR,APA,MPC,DVN,PSX,XOM,DINO,PR,OVV,VNOM,MGY,CVX,COP,PBF,OXY,EOG,SHEL"},
    {"t":"REMX","h":"ALB,SQM,LAC,MP,UUUU,USAR,SGML,DNN,UEC,CCJ,LAR,IDR,IPX,TROX"},
    {"t":"CRAK","h":"MPC,PSX,VLO,DINO,PBF,DK,PARR,CLMT"},
    {"t":"ITA","h":"GE,RTX,BA,HWM,TDG,GD,LHX,LMT,NOC,AXON,CW,WWD,RKLB,BWXT,CRS,TXT,ATI,HEI,KTOS,HII,AVAV,HXL,KRMN"},
    {"t":"FCG","h":"AR,EXE,DVN,EQT,FANG,CTRA,APA,EOG,OXY,SM,COP,HESM"},
    {"t":"XAR","h":"ATI,CRS,WWD,HII,KTOS,CW,RTX,HWM,HXL,BWXT,TDG,AVAV,GD,GE,TXT,LHX,HEI,LMT,NOC,ACHR,RKLB,BA,KRMN,AXON,SARO"},
    {"t":"BOAT","h":"FRO,MATX,STNG,INSW,ZIM,SBLK,CMDB,DHT,CMRE,DAC,TNK,NAT"},
    {"t":"RSPG","h":"KMI,OXY,APA,WMB,EQT,CTRA,OKE,FANG,BKR,TRGP,XOM,COP,TPL,EOG,CVX,HAL,HES,SLB,PSX,DVN,VLO,MPC,SHEL"},
    {"t":"XLE","h":"XOM,CVX,COP,WMB,MPC,EOG,PSX,VLO,SLB,KMI,BKR,OKE,TRGP,EQT,OXY,FANG,EXE,DVN,HAL,CTRA,TPL,APA"},
    {"t":"FFTY","h":"ANAB,CLS,ARQT,STOK,FIX,MU,AU,TARS,APH,TVTX,AGI,KGC,GH,VRT,AEM,LLY,HWM,ARGX,ONC,WGS,ZYME,FN,TMDX"},
    {"t":"AMLP","h":"MPLX,WES,EPD,SUN,PAA,ET,HESM,CQP,USAC,GEL,SPH,GLP,DKL"},
    {"t":"IYZ","h":"CSCO,T,VZ,LITE,ASTS,CIEN,ANET,TMUS,TIGO,CMCSA,FYBR,ROKU,IRDM,MSI,UI,CHTR,GLIBK"},
    {"t":"XLI","h":"GE,CAT,RTX,UBER,GEV,BA,UNP,ETN,HON,DE,PH,ADP,TT,MMM,LMT,GD,HWM,WM,TDG,JCI,EMR,NOC,UPS,CMI,PWR"},
    {"t":"RSPN","h":"BA,TDG,LUV,GE,UAL,MMM,GD,HON,PAYX,J,UBER,RTX,FDX,HII,NOC,ADP,UNP,VLTO,ROK,UPS,EFX,VRSK,GEV,LMT,AME"},
    {"t":"GNR","h":"XOM,NEM,CVX,CTVA,FCX,ADM,VALE,AEM"},
    {"t":"FAN","h":"CWEN,NEE,ACA,LNT,TKR"},
    {"t":"RSPC","h":"MTCH,TKO,FYBR,WBD,LYV,TTWO,T,DIS,NFLX,NYT,CMCSA,TMUS,VZ,META,OMC,IPG,PARA,CHTR,EA,NWSA,FOXA,GOOGL,FOX,NWS"},
    {"t":"IYT","h":"UBER,UNP,UPS,FDX,CSX,NSC,DAL,UAL,ODFL,EXPD,CHRW,LUV,XPO,JBHT,AAL,LYFT,SAIA,KNX,JOBY,R,KEX,GXO,ALK,LSTR"},
    {"t":"XES","h":"LBRT,HP,RIG,WFRD,SEI,NOV,HAL,FTI,VAL,PTEN,BKR,KGS,NE,SLB,WHD,AROC,TDW,SDRL,OII,XPRO,PUMP,AESI,NBR,HLX,WTTR"},
    {"t":"RSP","h":"WBD,ALB,WDC,MU,INTC,TER,AMAT,LRCX,STX,LLY,CAH,REGN,WAT,AMD,CAT,FSLR,UHS,HCA,GM,BIIB,ISRG,JBHT,TECH,RVTY,STLD"},
    {"t":"GDX","h":"PAAS,WPM,KGC,AGI,FNV,AEM,B,GFI,NEM"},
    {"t":"GDXJ","h":"PAAS,KGC,EQX,AGI,HMY,IAG,RGLD,USAU,GORO,CDE,GLDG"},
    {"t":"SIL","h":"EXK,AG,PAAS,HL,FSM,WPM,HCHDF,OR,SVM,SSRM,CDE"},
    {"t":"SILJ","h":"AG,HYMC,HL,PAAS,OR,FSM,EXK,GORO,SSRM,CDE"},
    {"t":"RSPU","h":"CNP,ETR,EXC,PCG,SRE,CMS,AEE,DTE,NI,SO,XEL,ATO,D,PPL,PEG,WEC,EVRG,FE,DUK,AEP,LNT,EIX,NEE,ED,NRG"},
    {"t":"AIPO","h":"PWR,VRT,GEV,ETN,CCJ,AVGO,NVDA,CEG,BE,HUBB,FLNC,AMD,OKLO,MTZ,VST,NVT,EOSE,BWXT"},
    {"t":"XHS","h":"GH,PACS,BKD,NEO,CAH,WGS,UHS,BTSG,HCA,SEM,COR,MCK,CNC,ALHC,RDNT,THC,HQY,PGNY,ENSG,ADUS,HSIC,OPCH,ELV,CVS,PRVA"},
    {"t":"OIH","h":"RIG,WFRD,FTI,NE,BKR,NOV,VAL,TS,SLB,PUMP,HAL"},
    {"t":"RSPH","h":"CNC,PFE,WST,BDX,EW,ABBV,GILD,BMY,CAH,ABT,LH,BSX,MCK,MRK,ZBH,DXCM,DVA,STE,IQV,JNJ,TMO,LLY,ISRG,DHR,DGX"},
    {"t":"KIE","h":"LMND,BHF,MCY,WTM,RYAN,ORI,L,RNR,AGO,CINF,MKL,AIZ,TRV,CB,PFG,AXS,ALL,THG,CNO,HIG,PLMR,ACGL,PRU,AFL,UNM"},
    {"t":"SLX","h":"VALE,STLD,NUE,MT,CLF,ATI,RIO,HCC,CRS,PKX"},
    {"t":"XHE","h":"TNDM,AXGN,INSP,GMED,HAE,ATEC,TMDX,ISRG,GKOS,SOLV,COO,IDXX,HOLX,OMCL,LNTH,MDT,LIVN,ALGN,UFPT,ICUI,EW,STE,PEN,NEOG,GEHC"},
    {"t":"XLV","h":"LLY,JNJ,ABBV,UNH,MRK,ABT,TMO,ISRG,AMGN,GILD,BSX,PFE,DHR,MDT,SYK,VRTX,MCK,CVS,BMY,HCA,REGN,ELV,CI,COR,IDXX"},
    {"t":"COPX","h":"ERO,TGB,FCX,HBM,IE,TECK,SCCO"},
    {"t":"DRNZ","h":"ONDS,AVAV,ELS,DRSHF,EH,RCAT,UMAC,PLTR,DPRO,KTOS"},
    {"t":"URA","h":"CCJ,OKLO,UEC,URNM,NXE,LEU,LTBR,UUUU,SMR,SBSW,DNN,URNJ,SRUUF,NNE,EU,ASPI,UROY,BWXT,MIR,CEG"},
    {"t":"UFO","h":"ASTS,GSAT,PL,VSAT,TRMB,SATS,SIRI,RKLB,GRMN,IRDM,RTX,GE,LHX,VOYG,NOC,LMT,HON,LUNR,BA,FLY,CMCSA"},
    {"t":"PAVE","h":"HWM,PWR,PH,CRH,SRE,NSC,FAST,TT,CSX,ROK,URI,EMR,DE,ETN,UNP,VMC,MLM,NUE,EME,STLD,HUBB,TRMB,FTV,WWD,PNR"},
    {"t":"LIT","h":"RIO,ALB,LAR,SEI,MVST,ENS,EOSE,SQM,AMPX,TSLA,BLDP,SLDP,ABAT,SGML,SLI,LAC"},
    {"t":"XLB","h":"LIN,NEM,SHW,ECL,NUE,FCX,MLM,VMC,APD,CTVA,STLD,PPG,IP,AMCR,PKG,IFF,DOW,DD,ALB,AVY,BALL,CF,LYB,MOS"},
    {"t":"XPH","h":"LQDA,MBX,OGN,AXSM,EWTX,ELAN,VTRS,AMLX,AMRX,MRK,CRNX,XERS,LGND,JNJ,LLY,PBH,BMY,PRGO,SUPN,RPRX,ZTS,JAZZ,AVDL,PFE"},
    {"t":"PBJ","h":"MNST,HSY,KR,SYY,MDLZ,KHC,CTVA,DASH,HLF,UNFI,SEB,IMKTA,TSN,USFD,FDP,CART,CHEF,ADM,AGRO,TR,ACI,DPZ,POST,JBS,WMK"},
    {"t":"RSPS","h":"DLTR,KR,KMB,MNST,K,CHD,CAG,KO,TGT,PG,CLX,CL,KHC,SJM,GIS,PEP,CPB,HSY,KDP,KVUE,WMT,HRL,MO,SYY"},
    {"t":"XLP","h":"WMT,COST,PG,KO,PM,PEP,CL,MDLZ,MO,MNST,TGT,KR,KDP,SYY,KMB,KVUE,ADM,HSY,GIS,DG,EL,K,KHC,DLTR,CHD"},
    {"t":"QQQE","h":"MU,AMD,REGN,AMAT,ISRG,BIIB,INTC,WBD,AZN,LRCX,ROST,AMGN,MRVL,MNST,EA,AVGO,IDXX,CTSH,AEP,DDOG,MAR,AAPL,VRTX,XEL,GILD"},
    {"t":"ROBO","h":"SYM,TER,ISRG,SERV,IRBT,COHR,ROK,ILMN,RR,ARBE,AUR,GMED,ROK,PRCT,NOVT,PDYN,IPGP,NDSN,EMR"},
    {"t":"ARKK","h":"TSLA,ROKU,COIN,TEM,CRSP,SHOP,HOOD,RBLX,AMD,PLTR,BEAM,TER,CRCL,BMNR,ACHR,TXG,TWST,ILMN,AMZN,VCYT,BLSH,NVDA,NTRA,META,DKNG"},
    {"t":"IWM","h":"CRDO,BE,FN,IONQ,NXT,GH,TSLA,KTOS,BBIO,CDE,MDGL,ENSG,HL,RMBS,SPXC,SATS,DY,STRL,OKLO,GTLS,IDCC,HQY,MOD,UMBF,AVAV"},
    {"t":"HYDR","h":"BE,PLUG,BLDP,SLDP,FCEL,CMI,APD"},
    {"t":"PEJ","h":"WBD,LVS,SYY,CCL,DASH,LYV,RCL,FLUT,LYFT,LION,EXPE,FOXA,CNK,TKO,CPA,USFD,PSKY,CART,BYD,EAT,BH,RRR,DPZ,MGM,ACEL"},
    {"t":"EEM","h":"PDD,BABA,NU,MELI,DLO,JMIA"},
    {"t":"IWO","h":"CRDO,BE,FN,IONQ,GH,KTOS,BBIO,MDGL,ENSG,NXT,RMBS,SPXC,DY,STRL,GTLS,IDCC,HQY,MOD,AVAV,AEIS,WTS,RGTI,ZWS,HIMS,LUMN"},
    {"t":"MOO","h":"DE,ZTS,CTVA,ADM,TSN,CF,BG,ELAN,MOS,CNH,DAR,TTC,AGCO,CAT"},
    {"t":"XME","h":"HL,AA,HCC,STLD,NEM,LEU,CDE,NUE,CLF,CMC,RGLD,BTU,CNR,RS,FCX,UEC,MP,AMR,MTRN,CENX,IE,KALU,USAR,WS,MUX"},
    {"t":"ARKF","h":"SHOP,COIN,HOOD,PLTR,TOST,SOFI,XYZ,RBLX,ROKU,CRCL,MELI,AMD,DKNG,META,AMZN,BMNR,PINS,NU,KLAR,SE,BLSH,FUTU,Z"},
    {"t":"EWZ","h":"NU,MELI,DLO"},
    {"t":"RSPM","h":"CE,IP,PPG,BALL,LYB,ECL,DOW,IFF,LIN,CTVA,AVY,PKG,MLM,CF,AMCR,VMC,APD,DD,EMN,SHW,FCX,NEM,MOS,FMC"},
    {"t":"RSPT","h":"AVGO,JBL,PLTR,TER,ANET,AAPL,VRSN,CSCO,INTC,SWKS,JNPR,GLW,ADI,TXN,KLAC,TDY,DELL,HPE,CDNS,ANSS,CDW,QCOM,NVDA,FFIV,MPWR"},
    {"t":"XBI","h":"EXAS,RVMD,RNA,INSM,NTRA,BBIO,REGN,MDGL,IONS,BIIB,AMGN,UTHR,INCY,ROIV,EXEL,VRTX,GILD,NBIX,ABBV,BMRN,CRSP,MRNA,PTCT,ALNY,KRYS"},
    {"t":"VERS","h":"GOOGL,AAPL,QCOM,EXPI,U,AMZN,NVDA,MU,MSFT,META,KOPN,RBLX,HIMX,SNAP,STGW,VUZI,EA,ARM,AMBA,ADSK,PTC,AMKR,TTWO,WSM,NOK"},
    {"t":"XSD","h":"MU,INTC,RGTI,AMD,MRVL,MTSI,FSLR,RMBS,SITM,SMTC,MPWR,ADI,QCOM,CRUS,ON,AVGO,CRDO,LSCC,NVDA,QRVO,SLAB,TXN,NXPI,SWKS,OLED,TSEM"},
    {"t":"ARKG","h":"TEM,CRSP,PSNL,GH,TWST,TXG,NTRA,BEAM,ILMN,VCYT,RXRX,ADPT,IONS,ABSI,CDNA,SDGR,NTLA,NRIX,PACB,WGS,BFLY,PRME,ARCT,AMGN"},
    {"t":"GBTC","h":"IBIT,ETHA,MSTR,BMNR,SBET,COIN"},
    {"t":"IGV","h":"PLTR,MSFT,CRM,INTU,NOW,ORCL,APP,ADBE,CRWD,PANW,CDNS,SNPS,ADSK,FTNT,DDOG,ROP,WDAY,EA,MSTR,TTWO,FICO,TEAM,ZS,ZM,PTC"},
    {"t":"XTN","h":"JBHT,KEX,CHRW,FDX,EXPD,KNX,UPS,LYFT,LUV,XPO,AAL,CSX,MATX,UNP,NSC,DAL,HUBG,LSTR,JOBY,GXO,SNDR,ODFL,SAIA,UAL,WERN"},
    {"t":"PHO","h":"WAT,FERG,ECL,ROP,AWK,MLI,IEX,WMS,XYL,PNR,VLTO,AOS,ACM,CNM,VMI,WTRG,BMI,TTEK,ITRI,WTS,ZWS,MWA,SBS,FELE,HWKN"},
    {"t":"LABU","h":"EXAS,RVMD,RNA,INSM,REGN,NTRA,MDGL,BBIO,BIIB,IONS,UTHR,AMGN,INCY,ROIV,EXEL"},
    {"t":"BLOK","h":"HOOD,CIFR,HUT,GLXY,CLSK,WULF,COIN,IBM,NU,BBBY,PYPL,CMPO,CORZ,FBTC,OPRA,RBLX,BLK,FIGR,XYZ,CME,IBIT,BITB,MELI,CAN,BKKT"},
    {"t":"BUZZ","h":"APLD,INTC,GME,NBIS,TSLA,META,SOFI,RGTI,IREN,AMZN,PLTR,NVDA,HOOD,ASTS,OPEN,AMD,MSTR,HIMS,GOOGL,AAPL,UNH,SOUN,SMCI,DKNG,PYPL"},
    {"t":"XSW","h":"CIFR,SEMR,PRO,WULF,HUT,CLSK,TDC,QBTS,JAMF,APPN,BBAI,EPAM,PATH,WK,EA,IBM,CRWD,IDCC,GDYN,FICO,CRNC,DDOG,SNPS,CTSH,AGYS"},
    {"t":"SMH","h":"NVDA,TSM,AVGO,MU,INTC,AMAT,AMD,ASML,LRCX,KLAC,ADI,QCOM,TXN,CDNS,SNPS,MRVL,NXPI,MPWR,TER,MCHP,STM,ON,SWKS,QRVO,OLED,TSEM"},
    {"t":"WCLD","h":"FSLY,SEMR,MDB,DOCN,FROG,PATH,SNOW,BILL,DDOG,CFLT,WK,TWLO,CRWD,PCOR,AGYS,IOT,BRZE,QLYS,CWAN,BL,CLBT,PANW,SHOP,INTA,NET"},
    {"t":"DRIV","h":"GOOGL,BE,TSLA,INTC,NVDA,MSFT,RIO,QCOM,GM,ALB,NBIS,COHR,HON,SQM,ENS,F,BIDU,AMPX,SITM"},
    {"t":"WGMI","h":"CIFR,IREN,BITF,WULF,RIOT,HUT,CORZ,APLD,CLSK,HIVE,BTDR,MARA,NVDA,CANG,GLXY,XYZ,BTBT,TSM,CAN"},
    {"t":"IBUY","h":"FIGS,LQDT,CVNA,UPWK,EXPE,CART,W,RVLV,CHWY,LYFT,MSM,EBAY,BKNG,AFRM,SPOT,TRIP,ABNB,PTON,UBER,AMZN,CPRT,PYPL,HIMS,SSTK,ETSY"},
    {"t":"XHB","h":"SKY,SGI,WMS,CVCO,BLD,JCI,IBP,TT,KBH,TOL,ALLE,LEN,PHM,MTH,LOW,TMHC,NVR,WSM,DHI,MAS,LII,CARR,HD,CSL,BLDR"},
    {"t":"TAN","h":"NXT,FSLR,RUN,ENPH,SEDG,HASI,CSIQ,CWEN,DQ,SHLS,ARRY,JKS"},
    {"t":"KCE","h":"AMG,IVZ,MS,CBOE,BK,SF,CME,STT,HOOD,LPLA,GS,NTRS,IBKR,STEP,SCHW,MSCI,PIPR,JHG,TPG,MCO,EVR,TROW,GLXY,FHI,NDAQ"},
    {"t":"IPAY","h":"AXP,V,MA,PYPL,CPAY,FIS,AFRM,XYZ,GPN,COIN,TOST,FISV,FOUR,WEX,QTWO,STNE,ACIW,EEFT,WU,RELY"},
    {"t":"ITB","h":"DHI,LEN,PHM,NVR,TOL,SHW,LOW,BLD,HD,LII,MAS,BLDR,TMHC,IBP,MTH,OC,SKY,CVCO,KBH,EXP,SSD,FND,MHO,FBIN,MHK"},
    {"t":"RSPF","h":"GL,ERIE,MET,WTW,FI,V,AJG,CB,ALL,L,CME,MMC,MA,AXP,AON,MS,EG,WFC,STT,FDS,AFL,PRU,AIZ,AIG,JPM,FICO"},
    {"t":"WOOD","h":"PCH,RYN,SLVM,WY,SW,IP,CLW"},
    {"t":"COAL","h":"HCC,AMR,BTU,NRP,ARLP,METC,BHP,AREC,NC"},
    {"t":"ICLN","h":"BE,FSLR,NXT,ORA,ENPH,RUN,CWEN,PLUG"},
    {"t":"KRE","h":"CADE,VLY,COLB,CFG,TFC,PB,FNB,EWBC,FHN,FLG,WBS,ONB,MTB,PNFP,RF,HBAN,ZION,BPOP,WAL,UMBF,WTFC,SSB,CFR,HWC"},
    {"t":"CIBR","h":"AVGO,CRWD,CSCO,INFY,PANW,LDOS,FTNT,CYBR,CHKP,NET,ZS,GEN,AKAM,FFIV,OKTA,BAH,RBRK,S,CVLT,QLYS,SAIC,VRNS"},
    {"t":"PBW","h":"CSIQ,TE,FLNC,SGML,LAC,LAR,ALB,SQM,EOSE,NXT,FSLR,ORA,SLI,BE,NVTS,BEPC,AEIS,MYRG,JKS,PWR,SLDP,RUN,SHLS,RIVN,AMRC"},
    {"t":"XRT","h":"VSCO,REAL,KSS,M,ODP,EYE,DDS,ROST,GAP,DLTR,WMT,URBN,SBH,FIVE,PSMT,AEO,TJX,RVLV,ULTA,BOOT,ANF,CASY,SIG,CVNA,DG"},
    {"t":"RSPD","h":"DRI,TPR,GM,ULTA,APTV,TSLA,BBY,DECK,RL,EBAY,ROST,MCD,EXPE,TJX,YUM,HLT,MAR,AMZN,NKE,LULU,AZO,F,KMX,ABNB,LKQ"},
    {"t":"XLF","h":"BRK.B,JPM,V,MA,BAC,WFC,GS,MS,AXP,C,SCHW,SPGI,BLK,COF,PGR,CB,BX,CME,HOOD,MMC,ICE,KKR,BK,USB,PNC"},
    {"t":"ESPO","h":"EA,NTES,TTWO,RBLX,SE,U,LOGI,GME,PLTK,CRSR,SONY"},
    {"t":"JETS","h":"LUV,AAL,DAL,UAL,ALGT,SNCY,SKYW,ULCC,JBLU,EXPE,GD,ALK,TXT,SABR,BKNG,TRIP,BA,RYAAY"},
    {"t":"KBE","h":"CMA,BKU,BANC,EBC,PFSI,CADE,BK,VLY,COLB,WFC,BAC,INDB,C,FBK,CFG,TCBI,BOKF,SBCF,TFC,PB,NTRS,ABCB,FIBK,JPM,WSBC"},
    {"t":"FXI","h":"BABA,TCEHY,NTES,BYDDF,TCOM,JD,BIDU,PTR,SNP,FUTU,KWEB"},
    {"t":"XLY","h":"AMZN,TSLA,HD,MCD,TJX,BKNG,LOW,SBUX,ORLY,NKE,DASH,GM,MAR,RCL,HLT,AZO,ROST,F,ABNB,CMG,DHI,YUM,EBAY,GRMN,EXPE"},
    {"t":"MSOS","h":"VFF,MNMD,TLRY,MO,CRON,GTBIF,TCNNF"},
]


# ─── Math Helpers ───────────────────────────────────────────────────────────

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
    """
    Gold:   EMA9 > EMA21 > SMA50 AND Price > EMA21 AND Price > SMA200
    Silver: EMA9 > EMA21 but does not meet all gold criteria
    Bronze: EMA9 < EMA21 and/or does not meet gold or silver
    """
    if any(v is None for v in [price, ema9, ema21, sma50, sma200]):
        return "b"
    if ema9 > ema21:
        if ema21 > sma50 and price > ema21 and price > sma200:
            return "g"
        return "s"
    return "b"


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    # Collect all unique holdings
    all_holdings = set()
    for e in ETF_INFO:
        if e.get("h"):
            for hh in e["h"].split(","):
                hh = hh.strip()
                if hh:
                    all_holdings.add(hh)

    holding_tickers = sorted(list(all_holdings))
    print(f"Grading {len(holding_tickers)} unique holdings...")

    end = datetime.now()
    h_start = end - timedelta(days=365)

    holding_grades = {}
    graded = 0
    failed = 0

    for i, tk in enumerate(holding_tickers):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i+1}/{len(holding_tickers)}...")
        try:
            ticker_obj = yf.Ticker(tk)
            hist = ticker_obj.history(
                start=h_start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
            if hist.empty or "Close" not in hist.columns:
                holding_grades[tk] = "b"
                failed += 1
                continue

            closes = hist["Close"].dropna().values.tolist()

            if len(closes) < 200:
                holding_grades[tk] = "b"
                failed += 1
                if tk in ("T", "CMCSA", "BAC", "AAPL", "XOM"):
                    print(f"    DEBUG {tk}: only {len(closes)} pts (need 200) -> b")
                continue

            ema9 = compute_ema(closes, 9)
            ema21 = compute_ema(closes, 21)
            sma50 = compute_sma(closes, 50)
            sma200 = compute_sma(closes, 200)
            price = closes[-1]

            grade = grade_holding(price, ema9, ema21, sma50, sma200)
            holding_grades[tk] = grade
            graded += 1

            if tk in ("T", "CMCSA", "BAC", "AAPL", "XOM"):
                print(f"    DEBUG {tk}: Price={price:.2f} EMA9={ema9:.2f} EMA21={ema21:.2f} "
                      f"SMA50={sma50:.2f} SMA200={sma200:.2f} pts={len(closes)} -> {grade}")

        except Exception as ex:
            holding_grades[tk] = "b"
            failed += 1
            if tk in ("T", "CMCSA", "BAC", "AAPL", "XOM"):
                print(f"    DEBUG {tk}: FAILED ({ex}) -> b")

        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    # Write grades.json
    with open("grades.json", "w") as f:
        json.dump(holding_grades, f, separators=(",", ":"))

    g_count = sum(1 for v in holding_grades.values() if v == "g")
    s_count = sum(1 for v in holding_grades.values() if v == "s")
    b_count = sum(1 for v in holding_grades.values() if v == "b")

    print(f"\nWritten grades.json — {g_count} gold, {s_count} silver, {b_count} bronze")
    print(f"Graded: {graded}, Failed/insufficient: {failed}")


if __name__ == "__main__":
    main()

