#!/usr/bin/env python3
"""
Fetch ETF data from Yahoo Finance, compute standard VARS
(Volatility-Adjusted Relative Strength) vs SPY, and write data.json.

Supports both real ETFs (price from Yahoo) and custom baskets
(equal-weighted synthetic metrics from component stocks).

Reads pre-computed holding grades from grades.json and dynamic
holdings for FFTY/BUZZ from dynamic_holdings.json.

Usage:
    pip install yfinance numpy
    python fetch_data.py
"""

import json
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf

# Use curl_cffi to bypass Yahoo Finance rate limiting (impersonates Chrome)
try:
    from curl_cffi import requests as cffi_requests
    _session = cffi_requests.Session(impersonate="chrome")
except ImportError:
    _session = None

BENCHMARK = "SPY"
LOOKBACK = 25
ATR_PERIOD = 14

ETF_INFO = [
    {"t":"XOP","n":"Oil Refining/Exploration","fn":"SS SPDR S&P Oil&Gas Exp","h":"APA,AR,CLMT,CNX,COP,CTRA,CVX,DINO,DK,DVN,EOG,EQT,EXE,FANG,GPOR,MGY,MPC,MUR,OVV,OXY,PARR,PBF,PR,PSX,RRC,SHEL,VLO,VNOM,XOM"},
    {"t":"REMX","n":"Rare Earth Metals","fn":"VanEck:RE & Str Metals","h":"ALB,SQM,LAC,MP,UUUU,USAR,SGML,DNN,UEC,CCJ,LAR,IDR,IPX,TROX"},
    {"t":"ITA","n":"Aerospace & Defense","fn":"iShares:US Aer&Def ETF","h":"ACHR,ATI,AVAV,AXON,BA,BWXT,CRS,CW,GD,GE,HEI,HII,HWM,HXL,KRMN,KTOS,LHX,LMT,NOC,RKLB,RTX,SARO,TDG,TXT,WWD"},
    {"t":"FCG","n":"Natural Gas","fn":"FT:Natural Gas","h":"AR,EXE,DVN,EQT,FANG,CTRA,APA,EOG,OXY,SM,COP,HESM"},
    {"t":"BOAT","n":"Maritime & Shipping","fn":"Tidal:SS Glb Ship","h":"FRO,MATX,STNG,INSW,ZIM,SBLK,CMDB,DHT,CMRE,DAC,TNK,NAT"},
    {"t":"XLE","n":"Energy","fn":"Sel Sector:Enrgy SS SPDR","h":"APA,BKR,COP,CTRA,CVX,DVN,EOG,EQT,EXE,FANG,HAL,HES,KMI,MPC,OKE,OXY,PSX,SHEL,SLB,TPL,TRGP,VLO,WMB,XOM"},
    {"t":"FFTY","n":"IBD 50","fn":"Innovator IBD 50","h":"ANAB,CLS,ARQT,STOK,FIX,MU,AU,TARS,APH,TVTX,AGI,KGC,GH,VRT,AEM,LLY,HWM,ARGX,ONC,WGS,ZYME,FN,TMDX"},
    {"t":"AMLP","n":"Energy Infrastructure","fn":"Alerian MLP","h":"MPLX,WES,EPD,SUN,PAA,ET,HESM,CQP,USAC,GEL,SPH,GLP,DKL"},
    {"t":"IYZ","n":"Telecom","fn":"iShares:US Telecom ETF","h":"AAOI,ANET,ASTS,CALX,CCOI,CHTR,CIEN,CMCSA,COMM,CSCO,DIS,EA,EXTR,FFIV,FOX,FOXA,FYBR,GLIBK,GOOGL,GSAT,IPG,IRDM,LITE,LUMN,LYV,META,MSI,MTCH,NFLX,NTCT,NWS,NWSA,NYT,OMC,ONDS,PARA,ROKU,T,TDS,TIGO,TKO,TMUS,TSAT,TTWO,USM,VIAV,VZ,WBD,ZBRA"},
    {"t":"XLI","n":"Industrials","fn":"Sel Sector:Ind SS SPDR","h":"GE,CAT,RTX,UBER,GEV,BA,UNP,ETN,HON,DE,PH,ADP,TT,MMM,LMT,GD,HWM,WM,TDG,JCI,EMR,NOC,UPS,CMI,PWR,LUV,UAL,PAYX,J,FDX,HII,VLTO,ROK,EFX,VRSK,AME"},
    {"t":"FAN","n":"Wind Energy","fn":"FT II:Global Wind Energy","h":"CWEN,NEE,ACA,LNT,TKR"},
    {"t":"IYT","n":"Transportation","fn":"iShares:US Transportatn","h":"UBER,UNP,UPS,FDX,CSX,NSC,DAL,UAL,ODFL,EXPD,CHRW,LUV,XPO,JBHT,AAL,LYFT,SAIA,KNX,JOBY,R,KEX,GXO,ALK,LSTR,MATX,HUBG,SNDR,WERN"},
    {"t":"XES","n":"Oil & Gas Equipment","fn":"SS SPDR S&P Oil&Gas E&S","h":"LBRT,HP,RIG,WFRD,SEI,NOV,HAL,FTI,VAL,PTEN,BKR,KGS,NE,SLB,WHD,AROC,TDW,SDRL,OII,XPRO,PUMP,AESI,NBR,HLX,WTTR"},
    {"t":"GDX","n":"Gold Miners","fn":"VanEck:Gold Miners","h":"PAAS,WPM,KGC,AGI,FNV,AEM,B,GFI,NEM"},
    {"t":"GDXJ","n":"Jr Gold Miners","fn":"VanEck:Jr Gold Miners","h":"PAAS,KGC,EQX,AGI,HMY,IAG,RGLD,USAU,GORO,CDE,GLDG"},
    {"t":"SIL","n":"Silver Miners","fn":"Glbl X Silver Miners ETF","h":"EXK,AG,PAAS,HL,FSM,WPM,HCHDF,OR,SVM,SSRM,CDE"},
    {"t":"SILJ","n":"Jr Silver Miners","fn":"Amplify Jr Slvr Miners","h":"AG,HYMC,HL,PAAS,OR,FSM,EXK,GORO,SSRM,CDE"},
    {"t":"XLU","n":"Utilities","fn":"Sel Sector:Utl SS SPDR","h":"NEE,SO,CEG,DUK,AEP,SRE,D,VST,EXC,XEL,ETR,PEG,ED,PCG,WEC,NRG,DTE,AEE,ATO"},
    {"t":"AIPO","n":"AI & Power Infra","fn":"Defiance AI & Pow Infra","h":"PWR,VRT,GEV,ETN,CCJ,AVGO,NVDA,CEG,BE,HUBB,FLNC,AMD,OKLO,MTZ,VST,NVT,EOSE,BWXT"},
    {"t":"XHS","n":"Healthcare Services","fn":"SS SPDR S&P Hlth Cr Svc","h":"GH,PACS,BKD,NEO,CAH,WGS,UHS,BTSG,HCA,SEM,COR,MCK,CNC,ALHC,RDNT,THC,HQY,PGNY,ENSG,ADUS,HSIC,OPCH,ELV,CVS,PRVA"},
    {"t":"OIH","n":"Oil Services","fn":"VanEck:Oil Services","h":"RIG,WFRD,FTI,NE,BKR,NOV,VAL,TS,SLB,PUMP,HAL"},
    {"t":"KIE","n":"Insurance","fn":"SS SPDR S&P Insurance","h":"LMND,BHF,MCY,WTM,RYAN,ORI,L,RNR,AGO,CINF,MKL,AIZ,TRV,CB,PFG,AXS,ALL,THG,CNO,HIG,PLMR,ACGL,PRU,AFL,UNM"},
    {"t":"SLX","n":"Steel","fn":"VanEck:Steel","h":"VALE,STLD,NUE,MT,CLF,ATI,RIO,HCC,CRS,PKX"},
    {"t":"XHE","n":"Healthcare Equipment","fn":"SS SPDR S&P Hlth Care Eq","h":"TNDM,AXGN,INSP,GMED,HAE,ATEC,TMDX,ISRG,GKOS,SOLV,COO,IDXX,HOLX,OMCL,LNTH,MDT,LIVN,ALGN,UFPT,ICUI,EW,STE,PEN,NEOG,GEHC"},
    {"t":"XLV","n":"Healthcare","fn":"Sel Sector:HC SS SPDR","h":"LLY,JNJ,ABBV,UNH,MRK,ABT,TMO,ISRG,AMGN,GILD,BSX,PFE,DHR,MDT,SYK,VRTX,MCK,CVS,BMY,HCA,REGN,ELV,CI,COR,IDXX,CNC,WST,BDX,EW,CAH,LH,ZBH,DXCM,DVA,STE,IQV,DGX"},
    {"t":"COPX","n":"Copper Miners","fn":"Glbl X Copper Miners ETF","h":"ERO,TGB,FCX,HBM,IE,TECK,SCCO"},
    {"t":"DRNZ","n":"Drones","fn":"REX Drone ETF","h":"ACHR,AVAV,DPRO,DRSHF,EH,ELS,JOBY,KTOS,ONDS,PLTR,RCAT,UMAC"},
    {"t":"URA","n":"Uranium / Nuclear","fn":"Glbl X Uranium ETF","h":"ASPI,BWXT,CCJ,CEG,DNN,EU,LEU,LTBR,MIR,NNE,NXE,OKLO,SBSW,SMR,SRUUF,TLN,UEC,URNJ,URNM,UROY,UUUU"},
    {"t":"UFO","n":"Space","fn":"Procure Space","h":"ASTS,BA,BKSY,CMCSA,FLY,GE,GEMI,GRMN,GSAT,HON,IRDM,LHX,LMT,LUNR,NOC,PL,RDW,RKLB,RTX,SATS,SIRI,TRMB,VOYG,VSAT"},
    {"t":"PAVE","n":"Infrastructure Dev","fn":"Glbl X US Infra Dev","h":"HWM,PWR,PH,CRH,SRE,NSC,FAST,TT,CSX,ROK,URI,EMR,DE,ETN,UNP,VMC,MLM,NUE,EME,STLD,HUBB,TRMB,FTV,WWD,PNR"},
    {"t":"LIT","n":"Lithium & Battery","fn":"Glbl X Lith & Bat Tech","h":"RIO,ALB,LAR,SEI,MVST,ENS,EOSE,SQM,AMPX,TSLA,BLDP,SLDP,ABAT,SGML,SLI,LAC"},
    {"t":"XLB","n":"Basic Materials","fn":"Sel Sector:Mat SS SPDR","h":"LIN,NEM,SHW,ECL,NUE,FCX,MLM,VMC,APD,CTVA,STLD,PPG,IP,AMCR,PKG,IFF,DOW,DD,ALB,AVY,BALL,CF,LYB,MOS,CE,EMN,FMC"},
    {"t":"XPH","n":"Pharmaceuticals","fn":"SS SPDR S&P Pharm","h":"LQDA,MBX,OGN,AXSM,EWTX,ELAN,VTRS,AMLX,AMRX,MRK,CRNX,XERS,LGND,JNJ,LLY,PBH,BMY,PRGO,SUPN,RPRX,ZTS,JAZZ,AVDL,PFE"},
    {"t":"PBJ","n":"Food & Beverage","fn":"Invesco Food & Beverage","h":"MNST,HSY,KR,SYY,MDLZ,KHC,CTVA,DASH,HLF,UNFI,SEB,IMKTA,TSN,USFD,FDP,CART,CHEF,ADM,AGRO,TR,ACI,DPZ,POST,JBS,WMK"},
    {"t":"XLP","n":"Consumer Staples","fn":"Sel Sector:C SSS SPDR I","h":"ADM,CAG,CHD,CL,CLX,COST,CPB,DG,DLTR,EL,GIS,HRL,HSY,K,KDP,KHC,KMB,KO,KR,KVUE,MDLZ,MNST,MO,PEP,PG,PM,SJM,SYY,TGT,WMT"},
    {"t":"QQQE","n":"Nasdaq-100 (EW)","fn":"Direxion:NASDAQ-100 EWI","h":"AAPL,ADI,AEP,AMAT,AMD,AMGN,ANET,ANSS,AVGO,AZN,BIIB,CDNS,CDW,CSCO,CTSH,DDOG,DELL,EA,FFIV,GILD,GLW,HPE,IDXX,INTC,ISRG,JBL,JNPR,KLAC,LRCX,MAR,MNST,MPWR,MRVL,MU,NVDA,PLTR,QCOM,REGN,ROST,SWKS,TDY,TER,TXN,VRSN,VRTX,WBD,XEL"},
    {"t":"ROBO","n":"Robotics & Automation","fn":"Robo Glbl Robots & Auto","h":"SYM,TER,ISRG,SERV,IRBT,COHR,ROK,ILMN,RR,ARBE,AUR,GMED,PRCT,NOVT,PDYN,IPGP,NDSN,EMR"},
    {"t":"ARKK","n":"Innovation / Growth","fn":"ARK Innovation","h":"TSLA,ROKU,COIN,TEM,CRSP,SHOP,HOOD,RBLX,AMD,PLTR,BEAM,TER,CRCL,BMNR,ACHR,TXG,TWST,ILMN,AMZN,VCYT,BLSH,NVDA,NTRA,META,DKNG"},
    {"t":"HYDR","n":"Hydrogen","fn":"Glbl X Hydrogen ETF","h":"BE,PLUG,BLDP,SLDP,FCEL,CMI,APD"},
    {"t":"PEJ","n":"Leisure & Ent","fn":"Invesco Leisure and Ent","h":"WBD,LVS,SYY,CCL,DASH,LYV,RCL,FLUT,LYFT,LION,EXPE,FOXA,CNK,TKO,CPA,USFD,PSKY,CART,BYD,EAT,BH,RRR,DPZ,MGM,ACEL"},
    {"t":"EEM","n":"Emerging Markets","fn":"iShares:MSCI Em Mkts","h":"PDD,BABA,NU,MELI,DLO,JMIA"},
    {"t":"IWO","n":"Small Cap Growth","fn":"iShares:Russ 2000 Gr","h":"CRDO,BE,FN,IONQ,GH,KTOS,BBIO,MDGL,ENSG,NXT,RMBS,SPXC,DY,STRL,GTLS,IDCC,HQY,MOD,AVAV,AEIS,WTS,RGTI,ZWS,HIMS,LUMN"},
    {"t":"MOO","n":"Agribusiness","fn":"VanEck:Agribusiness","h":"DE,ZTS,CTVA,ADM,TSN,CF,BG,ELAN,MOS,CNH,DAR,TTC,AGCO,CAT"},
    {"t":"XME","n":"Metals & Mining","fn":"SS SPDR S&P Metals&Mng","h":"HL,AA,HCC,STLD,NEM,LEU,CDE,NUE,CLF,CMC,RGLD,BTU,CNR,RS,FCX,UEC,MP,AMR,MTRN,CENX,IE,KALU,USAR,WS,MUX"},
    {"t":"ARKF","n":"Fintech Innovation","fn":"ARK BC & Fintech Innov","h":"SHOP,COIN,HOOD,PLTR,TOST,SOFI,XYZ,RBLX,ROKU,CRCL,MELI,AMD,DKNG,META,AMZN,BMNR,PINS,NU,KLAR,SE,BLSH,FUTU,Z"},
    {"t":"EWZ","n":"Brazil","fn":"iShares:MSCI Brazil","h":"NU,MELI,DLO"},
    {"t":"XBI","n":"Biotechnology","fn":"SS SPDR S&P Biotech","h":"EXAS,RVMD,RNA,INSM,NTRA,BBIO,REGN,MDGL,IONS,BIIB,AMGN,UTHR,INCY,ROIV,EXEL,VRTX,GILD,NBIX,ABBV,BMRN,CRSP,MRNA,PTCT,ALNY,KRYS"},
    {"t":"SMH","n":"Semiconductors","fn":"SS SPDR S&P Semiconductr","h":"MU,INTC,RGTI,AMD,MRVL,MTSI,FSLR,RMBS,SITM,SMTC,MPWR,ADI,QCOM,CRUS,ON,AVGO,CRDO,LSCC,NVDA,QRVO,SLAB,TXN,NXPI,SWKS,OLED,TSEM,TSM,AMAT,ASML,LRCX,KLAC,CDNS,SNPS,TER,MCHP,STM","yt":"XSD"},
    {"t":"ARKG","n":"Genomics","fn":"ARK Genomic Revolution","h":"TEM,CRSP,PSNL,GH,TWST,TXG,NTRA,BEAM,ILMN,VCYT,RXRX,ADPT,IONS,ABSI,CDNA,SDGR,NTLA,NRIX,PACB,WGS,BFLY,PRME,ARCT,AMGN"},
    {"t":"GBTC","n":"Bitcoin","fn":"GRAYSCALE BITCOIN TRUST","h":"IBIT,ETHA,MSTR,BMNR,SBET,COIN"},
    {"t":"IGV","n":"Software","fn":"iShares:Expand Tch-Sftwr","h":"ADBE,ADSK,AGYS,APP,APPN,BBAI,CDNS,CIFR,CLSK,CRM,CRNC,CRWD,CTSH,DDOG,EA,EPAM,FICO,FTNT,GDYN,HUT,IBM,IDCC,INTU,JAMF,MSFT,MSTR,NOW,ORCL,PANW,PATH,PLTR,PRO,PTC,QBTS,ROP,SEMR,SNPS,TDC,TEAM,TTWO,WDAY,WK,WULF,ZM,ZS"},
    {"t":"PHO","n":"Water Infrastructure","fn":"Invesco Water Res","h":"WAT,FERG,ECL,ROP,AWK,MLI,IEX,WMS,XYL,PNR,VLTO,AOS,ACM,CNM,VMI,WTRG,BMI,TTEK,ITRI,WTS,ZWS,MWA,SBS,FELE,HWKN"},
    {"t":"BLOK","n":"Blockchain","fn":"Amplify Blockchain Tech","h":"BBBY,BITB,BKKT,BLK,CAN,CIFR,CLSK,CME,CMPO,COIN,CORZ,CRCL,FBTC,FIGR,GLXY,HOOD,HUT,IBIT,IBM,MELI,NU,OPRA,PYPL,RBLX,WULF,XYZ"},
    {"t":"WCLD","n":"Cloud Computing","fn":"WisdomTree:Cloud Cmptng","h":"FSLY,SEMR,MDB,DOCN,FROG,PATH,SNOW,BILL,DDOG,CFLT,WK,TWLO,CRWD,PCOR,AGYS,IOT,BRZE,QLYS,CWAN,BL,CLBT,PANW,SHOP,INTA,NET"},
    {"t":"WGMI","n":"Crypto Miners / Data Centers","fn":"CoinShares Btc Mining","h":"APLD,BITF,BTBT,BTDR,CAN,CANG,CIFR,CLSK,CORZ,CRWV,GLXY,HIVE,HUT,IREN,MARA,NBIS,NVDA,RIOT,TSM,WULF,XYZ"},
    {"t":"IBUY","n":"Online Retail","fn":"Amplify Online Retail","h":"FIGS,LQDT,CVNA,UPWK,EXPE,CART,W,RVLV,CHWY,LYFT,MSM,EBAY,BKNG,AFRM,SPOT,TRIP,ABNB,PTON,UBER,AMZN,CPRT,PYPL,HIMS,SSTK,ETSY"},
    {"t":"TAN","n":"Solar Energy","fn":"Invesco Solar","h":"NXT,FSLR,RUN,ENPH,SEDG,HASI,CSIQ,CWEN,DQ,SHLS,ARRY,JKS"},
    {"t":"KCE","n":"Capital Markets","fn":"SS SPDR S&P Cap Mkts","h":"AMG,IVZ,MS,CBOE,BK,SF,CME,STT,HOOD,LPLA,GS,NTRS,IBKR,STEP,SCHW,MSCI,PIPR,JHG,TPG,MCO,EVR,TROW,GLXY,FHI,NDAQ"},
    {"t":"IPAY","n":"Digital Payments","fn":"Amplify Digital Payments","h":"AXP,V,MA,PYPL,CPAY,FIS,AFRM,XYZ,GPN,COIN,TOST,FISV,FOUR,WEX,QTWO,STNE,ACIW,EEFT,WU,RELY"},
    {"t":"ITB","n":"Home Construction","fn":"iShares:US Home Cons ETF","h":"DHI,LEN,PHM,NVR,TOL,SHW,LOW,BLD,HD,LII,MAS,BLDR,TMHC,IBP,MTH,OC,SKY,CVCO,KBH,EXP,SSD,FND,MHO,FBIN,MHK,SGI,WMS,JCI,TT,ALLE,WSM,CARR,CSL"},
    {"t":"WOOD","n":"Timber & Forestry","fn":"iShares:Gl Timber","h":"PCH,RYN,SLVM,WY,SW,IP,CLW"},
    {"t":"ICLN","n":"Clean Energy","fn":"iShares:Gl Cl Energy","h":"AEIS,ALB,AMRC,BE,BEPC,CSIQ,CWEN,ENPH,EOSE,FLNC,FSLR,JKS,LAC,LAR,MYRG,NVTS,NXT,ORA,PLUG,PWR,RIVN,RUN,SGML,SHLS,SLDP,SLI,SQM,TE"},
    {"t":"KRE","n":"Regional Banks","fn":"SS SPDR S&P Reg Banking","h":"CADE,VLY,COLB,CFG,TFC,PB,FNB,EWBC,FHN,FLG,WBS,ONB,MTB,PNFP,RF,HBAN,ZION,BPOP,WAL,UMBF,WTFC,SSB,CFR,HWC"},
    {"t":"CIBR","n":"Cybersecurity","fn":"FT II:Nsdq Cybersecurity","h":"AVGO,CRWD,CSCO,INFY,PANW,LDOS,FTNT,CYBR,CHKP,NET,ZS,GEN,AKAM,FFIV,OKTA,BAH,RBRK,S,CVLT,QLYS,SAIC,VRNS"},
    {"t":"XRT","n":"Retail","fn":"SS SPDR S&P Retail","h":"VSCO,REAL,KSS,M,ODP,EYE,DDS,ROST,GAP,DLTR,WMT,URBN,SBH,FIVE,PSMT,AEO,TJX,RVLV,ULTA,BOOT,ANF,CASY,SIG,CVNA,DG"},
    {"t":"XLF","n":"Financials","fn":"Sel Sector:Fin SS SPDR I","h":"AFL,AIG,AIZ,AJG,ALL,AON,AXP,BAC,BK,BLK,BRK.B,BX,C,CB,CME,COF,EG,ERIE,FDS,FI,FICO,GL,GS,HOOD,ICE,JPM,KKR,L,MA,MET,MMC,MS,PGR,PNC,PRU,SCHW,SPGI,STT,USB,V,WFC,WTW"},
    {"t":"ESPO","n":"Esports & Gaming","fn":"VanEck:VG and eSports","h":"EA,NTES,TTWO,RBLX,SE,U,LOGI,GME,PLTK,CRSR,SONY"},
    {"t":"JETS","n":"Airlines & Travel","fn":"US Global Jets","h":"LUV,AAL,DAL,UAL,ALGT,SNCY,SKYW,ULCC,JBLU,EXPE,GD,ALK,TXT,SABR,BKNG,TRIP,BA,RYAAY"},
    {"t":"KBE","n":"Banks","fn":"SS SPDR S&P Bank ETF","h":"CMA,BKU,BANC,EBC,PFSI,CADE,BK,VLY,COLB,WFC,BAC,INDB,C,FBK,CFG,TCBI,BOKF,SBCF,TFC,PB,NTRS,ABCB,FIBK,JPM,WSBC"},
    {"t":"FXI","n":"China Large-Cap","fn":"iShares:China Large Cp","h":"BABA,TCEHY,NTES,BYDDF,TCOM,JD,BIDU,PTR,SNP,FUTU,KWEB"},
    {"t":"XLY","n":"Consumer Discretionary","fn":"Sel Sctr:C D SS SPDR In","h":"ABNB,AMZN,APTV,AZO,BBY,BKNG,CMG,DASH,DECK,DHI,DRI,EBAY,EXPE,F,GM,GRMN,HD,HLT,KMX,LKQ,LOW,LULU,MAR,MCD,NKE,ORLY,RCL,RL,ROST,SBUX,TJX,TPR,TSLA,ULTA,YUM"},
    {"t":"MSOS","n":"Cannabis","fn":"AdvsrShs Pure USCannabis","h":"VFF,MNMD,TLRY,MO,CRON,GTBIF,TCNNF"},
    {"t":"BJK","n":"Casinos & Gaming","fn":"VanEck Gaming ETF","h":"LVS,FLUT,VICI,DKNG,WYNN,CHDN,GLPI,MGM,BYD"},
    # ─── Custom Baskets (no ETF ticker — synthetic metrics from components) ──
    {"t":"NURS","n":"Medical/Nursing Services","fn":"Custom Basket","h":"LH,FMS,ENSG,SOLV,GH,EHC,DVA,BTSG,RDNT,OPCH,HIMS,BLLN,CON,LFST,WGS","basket":True},
    {"t":"CHEMG","n":"Chemicals (Agricultural)","fn":"Custom Basket","h":"NTR,CF,MOS,ICL,SMG,FMC,UAN,LXU","basket":True},
    {"t":"CHEMS","n":"Chemicals (Specialty)","fn":"Custom Basket","h":"LIN,ECL,APD,DOW,LYB,ALB,SQM,WLK,EMN,NEU,CE,MEOH,CBT,HWKN,CC,KWR,GEL,MTX,ROG,SHW,DD,SOLS,OLN,AXTA","basket":True},
    {"t":"QNTM","n":"Quantum","fn":"Custom Basket","h":"SKYT,ARQQ,QSI,QUBT,QBTS,RGTI,IONQ","basket":True},
    {"t":"COOL","n":"HVAC / Cooling","fn":"Custom Basket","h":"FIX,EME,JCI,TT,WSO,VRT,CARR,LII,SPXC,AAON","basket":True},
    {"t":"AGENT","n":"Agentic AI","fn":"Custom Basket","h":"GTLB,ASAN,PATH,BRZE,VERI,AI,SOUN,BBAI,LAW,CRNC","basket":True},
    {"t":"OPTIC","n":"Photonics","fn":"Custom Basket","h":"LITE,COHR,AAOI,POET,ALMU,LWLG,MTSI,GLW,FN,GFS,TSEM","basket":True},
    {"t":"LIDAR","n":"LiDAR","fn":"Custom Basket","h":"OUST,AEVA,HSAI,TDY","basket":True},
    {"t":"IYR","n":"US Real Estate","fn":"iShares:US Real Estate","h":"WELL,PLD,EQIX,O,AMT,SPG,DLR,PSA,CBRE,VTR,CCI,VICI,EXR,IRM"},
]


# ─── Math Helpers ───────────────────────────────────────────────────────────

def compute_atr(highs, lows, closes, period=14):
    trs = []
    for i in range(1, len(closes)):
        if closes[i] is None or closes[i-1] is None or highs[i] is None or lows[i] is None:
            continue
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
    if len(trs) < period:
        return np.mean(trs) if trs else 0
    atr = np.mean(trs[:period])
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def compute_atr_series(highs, lows, closes, period=14):
    n = len(closes)
    atr_out = [0.0] * n
    trs = []
    for i in range(1, n):
        if closes[i] is None or closes[i-1] is None or highs[i] is None or lows[i] is None:
            atr_out[i] = atr_out[i-1]
            continue
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
        if len(trs) <= period:
            atr_out[i] = np.mean(trs)
        else:
            atr_out[i] = (atr_out[i-1] * (period - 1) + tr) / period
    return atr_out


def percentrank_inc(values, x):
    n = len(values)
    if n <= 1:
        return None
    return sum(1 for v in values if v < x) / (n - 1)


def compute_ema_series(closes, period):
    """Compute full EMA series from closes array."""
    if len(closes) < period:
        return [None] * len(closes)
    mult = 2 / (period + 1)
    ema_out = [None] * len(closes)
    ema_out[period - 1] = np.mean(closes[:period])
    for i in range(period, len(closes)):
        ema_out[i] = (closes[i] - ema_out[i-1]) * mult + ema_out[i-1]
    return ema_out


def compute_sma_series(closes, period):
    """Compute full SMA series from closes array."""
    sma_out = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        sma_out[i] = np.mean(closes[i - period + 1:i + 1])
    return sma_out


def compute_ma_status(closes):
    """Compute MA status: price vs MAs and MA slopes for breadth panel."""
    if len(closes) < 50:
        return None
    ema9 = compute_ema_series(closes, 9)
    ema21 = compute_ema_series(closes, 21)
    sma50 = compute_sma_series(closes, 50)
    price = closes[-1]
    p_ema9 = bool(price > ema9[-1]) if ema9[-1] is not None else None
    p_ema21 = bool(price > ema21[-1]) if ema21[-1] is not None else None
    p_sma50 = bool(price > sma50[-1]) if sma50[-1] is not None else None
    lookback = 5
    def slope(series):
        if series[-1] is not None and len(series) > lookback and series[-1 - lookback] is not None:
            return bool(series[-1] > series[-1 - lookback])
        return None
    s_ema9 = slope(ema9)
    s_ema21 = slope(ema21)
    s_sma50 = slope(sma50)
    return {
        "price": round(price, 2),
        "pos": [p_ema9, p_ema21, p_sma50],
        "slope": [s_ema9, s_ema21, s_sma50],
    }


def compute_market_regime(closes):
    """
    Compute market regime (Bullish/Bearish/Neutral) using a composite trend score
    similar to the Pine Script MMTS indicator's smooth_trend logic.

    Components (weighted):
      - RSI(14) score: (RSI - 50) * 2, weight 0.25
      - MA crossover score: EMA(10) vs EMA(21) divergence, weight 0.35
      - Bollinger Band position: where price sits in BB(20,2), weight 0.20
      - Price momentum: 5-day change vs benchmark, weight 0.20

    smooth_trend > 20 = Bullish, < -20 = Bearish, else Neutral
    """
    n = len(closes)
    if n < 50:
        return None

    # RSI(14)
    period = 14
    gains = []
    losses = []
    for i in range(1, n):
        delta = closes[i] - closes[i-1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))

    if len(gains) < period:
        return None

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    rs = avg_gain / avg_loss if avg_loss > 0 else 100
    rsi = 100 - (100 / (1 + rs))
    rsi_score = (rsi - 50) * 2  # -100 to +100

    # MA crossover score: EMA(10) vs EMA(21)
    ema10 = compute_ema_series(closes, 10)
    ema21 = compute_ema_series(closes, 21)
    if ema10[-1] is not None and ema21[-1] is not None and ema21[-1] != 0:
        md = (ema10[-1] - ema21[-1]) / ema21[-1] * 100
        ma_score = max(-100, min(100, md * 5))
    else:
        ma_score = 0

    # Bollinger Band position: BB(20, 2)
    bb_period = 20
    if n >= bb_period:
        bb_slice = closes[-bb_period:]
        bb_basis = np.mean(bb_slice)
        bb_std = np.std(bb_slice)
        bb_upper = bb_basis + 2 * bb_std
        bb_lower = bb_basis - 2 * bb_std
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_position = (closes[-1] - bb_lower) / bb_range * 100
            bb_score = (bb_position - 50) * 2
        else:
            bb_score = 0
    else:
        bb_score = 0

    # Price momentum (5-day change, simple)
    if n >= 6:
        mom = (closes[-1] / closes[-6] - 1) * 100
        mom_score = max(-100, min(100, mom * 10))
    else:
        mom_score = 0

    # Composite trend score (same weights as Pine Script)
    trend_score = (rsi_score * 0.25) + (ma_score * 0.35) + (bb_score * 0.20) + (mom_score * 0.20)

    # Classify regime
    if trend_score > 20:
        regime = "Bullish"
    elif trend_score < -20:
        regime = "Bearish"
    else:
        regime = "Choppy"

    return {
        "regime": regime,
        "score": round(float(trend_score), 1),
    }


def load_dynamic_holdings():
    """Load dynamic holdings for FFTY/BUZZ from dynamic_holdings.json."""
    if os.path.exists("dynamic_holdings.json"):
        with open("dynamic_holdings.json") as f:
            return json.load(f)
    return {}


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    # Override FFTY/BUZZ holdings from dynamic file
    dyn = load_dynamic_holdings()
    for info in ETF_INFO:
        if info["t"] in dyn and dyn[info["t"]]:
            info["h"] = dyn[info["t"]]
            print(f"  Loaded dynamic holdings for {info['t']}: {len(info['h'].split(','))} stocks")

    # Separate regular ETFs and baskets
    regular_etfs = [e for e in ETF_INFO if not e.get("basket")]
    basket_etfs = [e for e in ETF_INFO if e.get("basket") and e.get("h")]

    # Collect basket component tickers for download
    basket_components = set()
    for b in basket_etfs:
        for h in b["h"].split(","):
            h = h.strip()
            if h:
                basket_components.add(h)

    regular_tickers = [e.get("yt", e["t"]) for e in regular_etfs]
    all_download = list(set(["SPY", "QQQ", "IWM"] + regular_tickers + sorted(basket_components)))

    print(f"Fetching data for {len(all_download)} tickers ({len(regular_tickers)} ETFs + {len(basket_components)} basket components)...")

    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=91)

    raw = yf.download(
        all_download,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        threads=False,
        session=_session,
    )

    if raw.empty:
        print("ERROR: No data returned from Yahoo Finance")
        sys.exit(1)

    # Also download weekly data for weekly VARS
    w_start = end - timedelta(days=365)
    raw_w = yf.download(
        all_download,
        start=w_start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1wk",
        group_by="ticker",
        auto_adjust=True,
        threads=False,
        session=_session,
    )
    print(f"Weekly data downloaded: {not raw_w.empty}")

    def flatten_columns(df):
        """Flatten MultiIndex columns introduced in yfinance >= 0.2.51."""
        if df is None:
            return None
        if hasattr(df.columns, 'levels'):
            df = df.copy()
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df

    def get_df(ticker):
        try:
            if len(all_download) == 1:
                df = flatten_columns(raw)
                return df.dropna(subset=["Close"]) if df is not None else None
            else:
                sub = raw[ticker]
                sub = flatten_columns(sub)
                return sub.dropna(subset=["Close"]) if sub is not None else None
        except Exception:
            return None

    def get_df_w(ticker):
        try:
            if raw_w.empty:
                return None
            if len(all_download) == 1:
                df = flatten_columns(raw_w)
                return df.dropna(subset=["Close"]) if df is not None else None
            else:
                sub = raw_w[ticker]
                sub = flatten_columns(sub)
                return sub.dropna(subset=["Close"]) if sub is not None else None
        except Exception:
            return None

    spy_df = get_df("SPY")
    if spy_df is None or len(spy_df) < LOOKBACK + 1:
        print("ERROR: Insufficient SPY data")
        sys.exit(1)

    spy_closes = spy_df["Close"].values
    spy_highs = spy_df["High"].values
    spy_lows = spy_df["Low"].values
    spy_atr = compute_atr(spy_highs, spy_lows, spy_closes, ATR_PERIOD)
    spy_atr_series = compute_atr_series(spy_highs, spy_lows, spy_closes, ATR_PERIOD)
    spy_price = float(spy_closes[-1])
    spy_chg = (spy_closes[-1] / spy_closes[-2] - 1) * 100 if len(spy_closes) >= 2 else 0
    spy_ts_map = {ts: i for i, ts in enumerate(spy_df.index)}

    # Weekly SPY baseline
    LOOKBACK_W = 12  # 12 weeks lookback for weekly VARS
    spy_df_w = get_df_w("SPY")
    spy_w_closes = spy_df_w["Close"].values if spy_df_w is not None else np.array([])
    spy_w_highs = spy_df_w["High"].values if spy_df_w is not None else np.array([])
    spy_w_lows = spy_df_w["Low"].values if spy_df_w is not None else np.array([])
    spy_w_atr_series = compute_atr_series(spy_w_highs, spy_w_lows, spy_w_closes, ATR_PERIOD) if len(spy_w_closes) > ATR_PERIOD else []
    spy_w_ts_map = {ts: i for i, ts in enumerate(spy_df_w.index)} if spy_df_w is not None else {}

    def process_ticker(ticker, df):
        """Process a single ticker (ETF or component) and return metrics dict."""
        if df is None or len(df) < 10:
            return None

        c = df["Close"].values
        h = df["High"].values
        l = df["Low"].values
        v = df["Volume"].values
        length = len(c)

        price = float(c[-1])
        change = (c[-1] / c[-2] - 1) * 100 if length >= 2 else 0
        c5 = (c[-1] / c[-6] - 1) * 100 if length >= 6 else None
        c20 = (c[-1] / c[-21] - 1) * 100 if length >= 21 else None

        atr = compute_atr(h, l, c, ATR_PERIOD)
        valid_c = [x for x in c if x is not None]
        sma50 = np.mean(valid_c[-50:]) if len(valid_c) >= 50 else np.mean(valid_c)

        # ATR Extension: how far price is from 50-day SMA in ATR units (same for D/W)
        atr_ext = abs(c[-1] - sma50) / atr if atr > 0 else 0

        # ATR Mult (D): today's move relative to daily ATR%
        atr_pct = (atr / c[-2] * 100) if length >= 2 and c[-2] != 0 else 1
        atr_mult = abs(change) / atr_pct if atr_pct > 0 else 0

        vols = [x for x in v if x is not None and x > 0]
        if len(vols) > 1:
            today_vol = vols[-1]
            avg_vol = np.mean(vols[:-1][-20:])
            rvol = today_vol / avg_vol if avg_vol > 0 else None
        else:
            rvol = None

        common = []
        for idx, ts in enumerate(df.index):
            if ts in spy_ts_map and c[idx] is not None and spy_closes[spy_ts_map[ts]] is not None:
                common.append((idx, spy_ts_map[ts]))

        if len(common) < LOOKBACK:
            return {
                "rv": round(rvol * 100) if rvol else None,
                "am": round(atr_mult * 100), "ax": round(atr_ext * 100), "ch": round(change, 2),
                "c5": round(c5, 2) if c5 is not None else None,
                "c20": round(c20, 2) if c20 is not None else None,
                "rs": None, "rf": 0, "ra": 0, "p": round(price, 2),
                "fr": None, "vs": None,
            }

        etf_atr_series = compute_atr_series(h, l, c, ATR_PERIOD)
        extended = common[-(LOOKBACK + 1):]
        vars_series = []
        cumulative = 0.0
        for k in range(1, len(extended)):
            etf_i_prev, spy_i_prev = extended[k - 1]
            etf_i, spy_i = extended[k]
            etf_ret = (c[etf_i] - c[etf_i_prev]) / c[etf_i_prev] if c[etf_i_prev] != 0 else 0
            spy_ret = (spy_closes[spy_i] - spy_closes[spy_i_prev]) / spy_closes[spy_i_prev] if spy_closes[spy_i_prev] != 0 else 0
            etf_atr_pct = etf_atr_series[etf_i] / c[etf_i_prev] if c[etf_i_prev] != 0 and etf_atr_series[etf_i] > 0 else 1
            spy_atr_pct = spy_atr_series[spy_i] / spy_closes[spy_i_prev] if spy_closes[spy_i_prev] != 0 and spy_atr_series[spy_i] > 0 else 1
            etf_norm = etf_ret / etf_atr_pct if etf_atr_pct > 0 else 0
            spy_norm = spy_ret / spy_atr_pct if spy_atr_pct > 0 else 0
            cumulative += (etf_norm - spy_norm)
            vars_series.append(round(cumulative, 4))

        rs_series = vars_series
        vs_series = [0.0] + vars_series
        final_rs = rs_series[-1] if rs_series else 0
        rs_pctrank = percentrank_inc(rs_series, final_rs) if len(rs_series) > 1 else None

        adv_streak = 0
        for j in range(len(rs_series) - 1, 0, -1):
            if rs_series[j] > rs_series[j - 1]:
                adv_streak += 1
            else:
                break

        dec_streak = 0
        for j in range(len(rs_series) - 1, 0, -1):
            if rs_series[j] < rs_series[j - 1]:
                dec_streak += 1
            else:
                break

        return {
            "rv": round(rvol * 100) if rvol else None,
            "am": round(atr_mult * 100),
            "ax": round(atr_ext * 100),
            "ch": round(change, 2),
            "c5": round(c5, 2) if c5 is not None else None,
            "c20": round(c20, 2) if c20 is not None else None,
            "rs": round(rs_pctrank * 100) if rs_pctrank is not None else None,
            "rf": dec_streak,
            "ra": adv_streak,
            "p": round(price, 2),
            "fr": round(final_rs, 4),
            "vs": vs_series,
        }

    def process_ticker_weekly(ticker, df_w):
        """Process weekly data for a ticker — same VARS logic but on weekly bars."""
        if df_w is None or len(df_w) < 5:
            return {"w_rv": None, "w_am": None, "w_rs": None, "w_rf": 0, "w_ra": 0, "w_vs": None}

        c = df_w["Close"].values
        h = df_w["High"].values
        l = df_w["Low"].values
        v = df_w["Volume"].values
        length = len(c)

        # Weekly ATR mult: this week's change / weekly ATR%
        atr = compute_atr(h, l, c, ATR_PERIOD)
        week_change = (c[-1] / c[-2] - 1) * 100 if length >= 2 else 0
        atr_pct_w = (atr / c[-2] * 100) if length >= 2 and c[-2] != 0 else 1
        atr_mult = abs(week_change) / atr_pct_w if atr_pct_w > 0 else 0

        # Weekly R.Vol
        vols = [x for x in v if x is not None and x > 0]
        if len(vols) > 1:
            today_vol = vols[-1]
            avg_vol = np.mean(vols[:-1][-10:])
            rvol = today_vol / avg_vol if avg_vol > 0 else None
        else:
            rvol = None

        # Weekly VARS
        common = []
        for idx, ts in enumerate(df_w.index):
            if ts in spy_w_ts_map and c[idx] is not None and spy_w_closes[spy_w_ts_map[ts]] is not None:
                common.append((idx, spy_w_ts_map[ts]))

        if len(common) < LOOKBACK_W:
            return {"w_rv": round(rvol * 100) if rvol else None, "w_am": round(atr_mult * 100),
                    "w_rs": None, "w_rf": 0, "w_ra": 0, "w_vs": None}

        etf_atr_series = compute_atr_series(h, l, c, ATR_PERIOD)
        extended = common[-(LOOKBACK_W + 1):]
        vars_series = []
        cumulative = 0.0
        for k in range(1, len(extended)):
            etf_i_prev, spy_i_prev = extended[k - 1]
            etf_i, spy_i = extended[k]
            etf_ret = (c[etf_i] - c[etf_i_prev]) / c[etf_i_prev] if c[etf_i_prev] != 0 else 0
            spy_ret = (spy_w_closes[spy_i] - spy_w_closes[spy_i_prev]) / spy_w_closes[spy_i_prev] if spy_w_closes[spy_i_prev] != 0 else 0
            etf_atr_pct = etf_atr_series[etf_i] / c[etf_i_prev] if c[etf_i_prev] != 0 and etf_atr_series[etf_i] > 0 else 1
            spy_atr_pct = spy_w_atr_series[spy_i] / spy_w_closes[spy_i_prev] if spy_i < len(spy_w_atr_series) and spy_w_closes[spy_i_prev] != 0 and spy_w_atr_series[spy_i] > 0 else 1
            etf_norm = etf_ret / etf_atr_pct if etf_atr_pct > 0 else 0
            spy_norm = spy_ret / spy_atr_pct if spy_atr_pct > 0 else 0
            cumulative += (etf_norm - spy_norm)
            vars_series.append(round(cumulative, 4))

        rs_series = vars_series
        vs_series = [0.0] + vars_series
        final_rs = rs_series[-1] if rs_series else 0
        rs_pctrank = percentrank_inc(rs_series, final_rs) if len(rs_series) > 1 else None

        adv_streak = 0
        for j in range(len(rs_series) - 1, 0, -1):
            if rs_series[j] > rs_series[j - 1]:
                adv_streak += 1
            else:
                break
        dec_streak = 0
        for j in range(len(rs_series) - 1, 0, -1):
            if rs_series[j] < rs_series[j - 1]:
                dec_streak += 1
            else:
                break

        return {
            "w_rv": round(rvol * 100) if rvol else None,
            "w_am": round(atr_mult * 100),
            "w_rs": round(rs_pctrank * 100) if rs_pctrank is not None else None,
            "w_rf": dec_streak,
            "w_ra": adv_streak,
            "w_vs": vs_series,
        }

    # ─── Process regular ETFs ──────────────────────────────
    results = []
    for info in regular_etfs:
        ticker = info.get("yt", info["t"])
        df = get_df(ticker)
        metrics = process_ticker(ticker, df)
        w_metrics = process_ticker_weekly(ticker, get_df_w(ticker))
        if metrics is None:
            results.append({**info, "rv": None, "am": None, "ax": None, "ch": None, "c5": None,
                            "c20": None, "rs": None, "rf": 0, "ra": 0, "p": None,
                            "fr": None, "vs": None, **w_metrics})
        else:
            results.append({**info, **metrics, **w_metrics})

    # ─── Process custom baskets (averaged component metrics) ──
    # Pre-compute metrics for all basket components (daily + weekly)
    component_metrics = {}
    component_w_metrics = {}
    for tk in sorted(basket_components):
        df = get_df(tk)
        m = process_ticker(tk, df)
        if m:
            component_metrics[tk] = m
        wm = process_ticker_weekly(tk, get_df_w(tk))
        if wm:
            component_w_metrics[tk] = wm

    for info in basket_etfs:
        holdings = [h.strip() for h in info["h"].split(",") if h.strip()]
        valid = [h for h in holdings if h in component_metrics]

        if not valid:
            results.append({**info, "rv": None, "am": None, "ax": None, "ch": None, "c5": None,
                            "c20": None, "rs": None, "rf": 0, "ra": 0, "p": None,
                            "fr": None, "vs": None,
                            "w_rv": None, "w_am": None, "w_rs": None, "w_rf": 0, "w_ra": 0, "w_vs": None})
            continue

        # Average daily VARS series (equal-weighted)
        vs_lists = [component_metrics[h]["vs"] for h in valid if component_metrics[h].get("vs")]
        if vs_lists:
            min_len = min(len(vs) for vs in vs_lists)
            avg_vs = []
            for i in range(min_len):
                avg_vs.append(round(np.mean([vs[i] for vs in vs_lists]), 4))

            rs_series = avg_vs[1:] if len(avg_vs) > 1 else []
            final_rs = rs_series[-1] if rs_series else 0
            rs_pctrank = percentrank_inc(rs_series, final_rs) if len(rs_series) > 1 else None

            adv_streak = 0
            for j in range(len(rs_series) - 1, 0, -1):
                if rs_series[j] > rs_series[j - 1]:
                    adv_streak += 1
                else:
                    break
            dec_streak = 0
            for j in range(len(rs_series) - 1, 0, -1):
                if rs_series[j] < rs_series[j - 1]:
                    dec_streak += 1
                else:
                    break
        else:
            avg_vs = None
            rs_pctrank = None
            adv_streak = 0
            dec_streak = 0

        # Average weekly VARS series (equal-weighted)
        valid_w = [h for h in holdings if h in component_w_metrics]
        w_vs_lists = [component_w_metrics[h]["w_vs"] for h in valid_w if component_w_metrics[h].get("w_vs")]
        if w_vs_lists:
            w_min_len = min(len(vs) for vs in w_vs_lists)
            w_avg_vs = []
            for i in range(w_min_len):
                w_avg_vs.append(round(np.mean([vs[i] for vs in w_vs_lists]), 4))

            w_rs_series = w_avg_vs[1:] if len(w_avg_vs) > 1 else []
            w_final_rs = w_rs_series[-1] if w_rs_series else 0
            w_rs_pctrank = percentrank_inc(w_rs_series, w_final_rs) if len(w_rs_series) > 1 else None

            w_adv_streak = 0
            for j in range(len(w_rs_series) - 1, 0, -1):
                if w_rs_series[j] > w_rs_series[j - 1]:
                    w_adv_streak += 1
                else:
                    break
            w_dec_streak = 0
            for j in range(len(w_rs_series) - 1, 0, -1):
                if w_rs_series[j] < w_rs_series[j - 1]:
                    w_dec_streak += 1
                else:
                    break
        else:
            w_avg_vs = None
            w_rs_pctrank = None
            w_adv_streak = 0
            w_dec_streak = 0

        # Average scalar metrics
        def avg_metric(key):
            vals = [component_metrics[h][key] for h in valid if component_metrics[h].get(key) is not None]
            return round(np.mean(vals), 2) if vals else None

        def avg_w_metric(key):
            vals = [component_w_metrics[h][key] for h in valid_w if component_w_metrics[h].get(key) is not None]
            return round(np.mean(vals), 2) if vals else None

        results.append({
            **info,
            "rv": round(avg_metric("rv")) if avg_metric("rv") is not None else None,
            "am": round(avg_metric("am")) if avg_metric("am") is not None else None,
            "ax": round(avg_metric("ax")) if avg_metric("ax") is not None else None,
            "ch": avg_metric("ch"),
            "c5": avg_metric("c5"),
            "c20": avg_metric("c20"),
            "rs": round(rs_pctrank * 100) if rs_pctrank is not None else None,
            "rf": dec_streak,
            "ra": adv_streak,
            "p": None,
            "fr": round(final_rs, 4) if rs_series else None,
            "vs": avg_vs,
            "w_rv": round(avg_w_metric("w_rv")) if avg_w_metric("w_rv") is not None else None,
            "w_am": round(avg_w_metric("w_am")) if avg_w_metric("w_am") is not None else None,
            "w_rs": round(w_rs_pctrank * 100) if w_rs_pctrank is not None else None,
            "w_rf": w_dec_streak,
            "w_ra": w_adv_streak,
            "w_vs": w_avg_vs,
        })

    # Sort: RS desc, then Change desc (daily ranks)
    results.sort(key=lambda x: (x["rs"] if x["rs"] is not None else -1,
                                 x["ch"] if x["ch"] is not None else -999), reverse=True)
    for i, r in enumerate(results):
        r["rk"] = i + 1

    # Weekly ranks (sort by w_rs)
    w_sorted = sorted(results, key=lambda x: (x["w_rs"] if x["w_rs"] is not None else -1,
                                                x["c5"] if x["c5"] is not None else -999), reverse=True)
    w_rank_map = {}
    for i, r in enumerate(w_sorted):
        w_rank_map[r["t"]] = i + 1
    for r in results:
        r["w_rk"] = w_rank_map.get(r["t"], len(results))

    # ─── Rolling rank history (rank_history.json) ────────────
    # Stores up to 8 weekly snapshots of daily and weekly ranks, one per calendar week.
    # Structure: {"weeks": [{"date":"2026-03-14","d":{ticker:rank},"w":{ticker:rank}}, ...]}
    # Most recent week is last in the list.
    from datetime import date
    today_str = date.today().isoformat()
    # Determine ISO week key (year-week)
    today_week = date.today().isocalendar()
    week_key = f"{today_week[0]}-W{today_week[1]:02d}"

    rank_history = {"weeks": []}
    if os.path.exists("rank_history.json"):
        try:
            with open("rank_history.json") as rhf:
                rank_history = json.load(rhf)
        except Exception:
            rank_history = {"weeks": []}

    # Current rank snapshot
    current_d_ranks = {r["t"]: r["rk"] for r in results}
    current_w_ranks = {r["t"]: r["w_rk"] for r in results}

    # Update or append this week's snapshot (overwrite if same week)
    weeks = rank_history.get("weeks", [])
    if weeks and weeks[-1].get("wk") == week_key:
        # Update existing week entry
        weeks[-1] = {"wk": week_key, "date": today_str, "d": current_d_ranks, "w": current_w_ranks}
    else:
        # New week — append
        weeks.append({"wk": week_key, "date": today_str, "d": current_d_ranks, "w": current_w_ranks})

    # Keep only last 9 weeks (current + 8 prior)
    if len(weeks) > 9:
        weeks = weeks[-9:]
    rank_history["weeks"] = weeks

    with open("rank_history.json", "w") as rhf:
        json.dump(rank_history, rhf, separators=(",", ":"))
    print(f"Rank history: {len(weeks)} weekly snapshots stored")

    # ─── Attach rank history to data.json for frontend ────────
    # Store the full rank history so the frontend can compute deltas for any N
    # Format: rh_d = [[ticker, rank_N_weeks_ago], ...] for each week going back
    # We send arrays of {ticker: rank} dicts, oldest first
    rh_d = []  # daily rank snapshots, oldest to newest (excluding current)
    rh_w = []  # weekly rank snapshots
    for snap in weeks[:-1]:  # exclude current week
        rh_d.append(snap.get("d", {}))
        rh_w.append(snap.get("w", {}))

    # Remove per-ETF rd/w_rd since frontend will compute them

    # ─── Load cached holding grades from grades.json ────────
    holding_grades = {}
    if os.path.exists("grades.json"):
        with open("grades.json") as gf:
            holding_grades = json.load(gf)
        print(f"Loaded {len(holding_grades)} cached holding grades from grades.json")
    else:
        print("WARNING: grades.json not found — all holdings will show bronze")

    for e in results:
        if e.get("h"):
            hg = {}
            for hh in e["h"].split(","):
                hh = hh.strip()
                if hh:
                    hg[hh] = holding_grades.get(hh, "b")
            e["hg"] = hg

    # ─── Advancing / Declining stats ────────────────────────
    total_with_rs = len([r for r in results if r.get("vs") and len(r["vs"]) >= 2])
    adv_today = sum(1 for r in results if r.get("vs") and len(r["vs"]) >= 2
                     and r["vs"][-1] > r["vs"][-2])
    dec_today = sum(1 for r in results if r.get("vs") and len(r["vs"]) >= 2
                     and r["vs"][-1] < r["vs"][-2])

    adv_pct = round(adv_today / total_with_rs * 100, 1) if total_with_rs else 0
    dec_pct = round(dec_today / total_with_rs * 100, 1) if total_with_rs else 0

    adv_streak_list = [[r["t"], r["n"]] for r in results if r.get("ra", 0) >= 4]
    dec_streak_list = [[r["t"], r["n"]] for r in results if r.get("rf", 0) >= 4]

    # Weekly streak lists (3+ weeks)
    w_adv_streak_list = [[r["t"], r["n"]] for r in results if r.get("w_ra", 0) >= 3]
    w_dec_streak_list = [[r["t"], r["n"]] for r in results if r.get("w_rf", 0) >= 3]

    # ─── Breadth MA Status for IWM, QQQ, SPY ────────────────
    breadth_tickers = ["IWM", "QQQ", "SPY"]
    breadth_status = {}
    for bt in breadth_tickers:
        df = get_df(bt) if bt != "SPY" else spy_df
        if df is not None and len(df) >= 50:
            c = df["Close"].values.tolist()
            status = compute_ma_status(c)
            if status:
                regime = compute_market_regime(c)
                if regime:
                    status["regime"] = regime["regime"]
                    status["regime_score"] = regime["score"]
                breadth_status[bt] = status
    print(f"Breadth MA status: {list(breadth_status.keys())}")
    for bt in breadth_tickers:
        if bt in breadth_status and "regime" in breadth_status[bt]:
            print(f"  {bt} regime: {breadth_status[bt]['regime']} (score: {breadth_status[bt].get('regime_score')})")

    data = {
        "e": results,
        "s": {
            "ap": adv_pct,
            "dp": dec_pct,
            "a": adv_streak_list[:30],
            "d": dec_streak_list[:30],
            "wa": w_adv_streak_list[:30],
            "wd": w_dec_streak_list[:30],
        },
        "meta": {
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "spy_price": round(spy_price, 2),
            "spy_change": round(float(spy_chg), 2),
        },
        "breadth": breadth_status,
        "rh": {"d": rh_d, "w": rh_w},
    }

    with open("data.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))

    g_count = sum(1 for v in holding_grades.values() if v == "g")
    s_count = sum(1 for v in holding_grades.values() if v == "s")
    b_count = sum(1 for v in holding_grades.values() if v == "b")

    print(f"\nWritten data.json - {len(results)} entries ({len(regular_etfs)} ETFs + {len(basket_etfs)} baskets)")
    print(f"Top 5: {[r['t'] for r in results[:5]]}")
    print(f"Advancing: {adv_pct}% | Declining: {dec_pct}%")
    if holding_grades:
        print(f"Cached grades: {g_count} gold, {s_count} silver, {b_count} bronze")


if __name__ == "__main__":
    main()
