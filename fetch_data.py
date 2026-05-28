#!/usr/bin/env python3
"""
Fetch ETF data from Yahoo Finance, compute standard VARS
(Volatility-Adjusted Relative Strength) vs SPY, and write data.json.

Supports both real ETFs (price from Yahoo) and custom baskets
(equal-weighted synthetic metrics from component stocks).

Reads pre-computed holding grades from grades.json and dynamic
holdings for BUZZ from dynamic_holdings.json.

Usage:
    pip install yfinance numpy
    python fetch_data.py
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    {"t":"MAGS","n":"Magnificent Seven","fn":"Roundhill:Mag Seven ETF","h":"GOOGL,AMZN,AAPL,META,MSFT,NVDA,TSLA"},
    {"t":"XOP","n":"Oil Refining/Exploration","fn":"SS SPDR S&P Oil&Gas Exp","h":"APA,AR,CLMT,CNX,COP,CTRA,CVX,DINO,DK,DVN,EOG,EQT,EXE,FANG,GPOR,MGY,MPC,MUR,OVV,OXY,PARR,PBF,PR,PSX,RRC,SHEL,VLO,VNOM,XOM"},
    {"t":"REMX","n":"Rare Earth Metals","fn":"VanEck:RE & Str Metals","h":"ALB,SQM,LAC,MP,UUUU,USAR,SGML,DNN,UEC,CCJ,LAR,IDR,IPX,TROX"},
    {"t":"ITA","n":"Aerospace & Defense","fn":"iShares:US Aer&Def ETF","h":"ACHR,ATI,AVAV,AXON,BA,BWXT,CRS,CW,GD,GE,HEI,HII,HWM,HXL,KRMN,KTOS,LHX,LMT,NOC,RKLB,RTX,SARO,TDG,TXT,WWD"},
    {"t":"FCG","n":"Natural Gas","fn":"FT:Natural Gas","h":"AR,EXE,DVN,EQT,FANG,CTRA,APA,EOG,OXY,SM,COP,HESM"},
    {"t":"BOAT","n":"Maritime & Shipping","fn":"Tidal:SS Glb Ship","h":"FRO,MATX,STNG,INSW,ZIM,SBLK,CMDB,DHT,CMRE,DAC,TNK,NAT"},
    {"t":"XLE","n":"Energy","fn":"Sel Sector:Enrgy SS SPDR","h":"APA,BKR,COP,CTRA,CVX,DVN,EOG,EQT,EXE,FANG,HAL,HES,KMI,MPC,OKE,OXY,PSX,SHEL,SLB,TPL,TRGP,VLO,WMB,XOM"},
    {"t":"AMLP","n":"Energy Infrastructure","fn":"Alerian MLP","h":"WES,PAA,SUN,ET,EPD,MPLX,HESM,CQP,USAC,GEL,SPH,GLP,DKL"},
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
    {"t":"OIH","n":"Oil Services","fn":"VanEck:Oil Services","h":"RIG,WFRD,FTI,NE,BKR,NOV,VAL,TS,SLB,PUMP,HAL"},
    {"t":"KIE","n":"Insurance","fn":"SS SPDR S&P Insurance","h":"LMND,BHF,MCY,WTM,RYAN,ORI,L,RNR,AGO,CINF,MKL,AIZ,TRV,CB,PFG,AXS,ALL,THG,CNO,HIG,PLMR,ACGL,PRU,AFL,UNM"},
    {"t":"SLX","n":"Steel","fn":"VanEck:Steel","h":"VALE,STLD,NUE,MT,CLF,ATI,RIO,HCC,CRS,PKX"},
    {"t":"XHE","n":"Healthcare Equipment","fn":"SS SPDR S&P Hlth Care Eq","h":"TNDM,AXGN,INSP,GMED,HAE,ATEC,TMDX,ISRG,GKOS,SOLV,COO,IDXX,HOLX,OMCL,LNTH,MDT,LIVN,ALGN,UFPT,ICUI,EW,STE,PEN,NEOG,GEHC"},
    {"t":"XLV","n":"Healthcare","fn":"Sel Sector:HC SS SPDR","h":"LLY,JNJ,ABBV,UNH,MRK,ABT,TMO,ISRG,AMGN,GILD,BSX,PFE,DHR,MDT,SYK,VRTX,MCK,CVS,BMY,HCA,REGN,ELV,CI,COR,IDXX,CNC,WST,BDX,EW,CAH,LH,ZBH,DXCM,DVA,STE,IQV,DGX"},
    {"t":"COPX","n":"Copper Miners","fn":"Glbl X Copper Miners ETF","h":"ERO,TGB,FCX,HBM,IE,TECK,SCCO"},
    {"t":"JEDI","n":"Drones & Modern Warfare","fn":"Defiance Drone and Modern Warfare ETF","h":"RDW,LUNR,UMAC,ONDS,RKLB,ASTS,ACHR,BKSY,AI,RCAT,PL,AVAV,JOBY,FEIM,SPIR,IRDM,KTOS,EVTL,PLTR,MRCY,PSN,AIRO,SAIC,TLS,CACI,SWMR,LDOS,GD,LHX,ESLT,RTX,KRKNF"},
    {"t":"URA","n":"Uranium / Nuclear","fn":"Glbl X Uranium ETF","h":"ASPI,BWXT,CCJ,CEG,DNN,EU,LEU,LTBR,MIR,NNE,NXE,OKLO,SBSW,SMR,SRUUF,TLN,UEC,URNJ,URNM,UROY,UUUU"},
    {"t":"UFO","n":"Space","fn":"Procure Space","h":"RKLB,PL,VSAT,IRDM,FLY,SIRI,GSAT,LUNR,ASTS,SATS,GRMN,TRMB,VOYG,BA,HON,RTX,LHX,BKSY,LMT,RDW,NOC,SATL,GILT,CMCSA,SPIR,KRMN,SIDU"},
    {"t":"PAVE","n":"Infrastructure Dev","fn":"Glbl X US Infra Dev","h":"HWM,PWR,PH,CRH,SRE,NSC,FAST,TT,CSX,ROK,URI,EMR,DE,ETN,UNP,VMC,MLM,NUE,EME,STLD,HUBB,TRMB,FTV,WWD,PNR"},
    {"t":"LIT","n":"Lithium & Battery","fn":"Glbl X Lith & Bat Tech","h":"RIO,ALB,LAR,SEI,MVST,ENS,EOSE,SQM,AMPX,TSLA,BLDP,SLDP,ABAT,SGML,SLI,LAC"},
    {"t":"XLB","n":"Basic Materials","fn":"Sel Sector:Mat SS SPDR","h":"LIN,NEM,SHW,ECL,NUE,FCX,MLM,VMC,APD,CTVA,STLD,PPG,IP,AMCR,PKG,IFF,DOW,DD,ALB,AVY,BALL,CF,LYB,MOS,CE,EMN,FMC"},
    {"t":"XPH","n":"Pharmaceuticals","fn":"SS SPDR S&P Pharm","h":"LQDA,MBX,OGN,AXSM,EWTX,ELAN,VTRS,AMLX,AMRX,MRK,CRNX,XERS,LGND,JNJ,LLY,PBH,BMY,PRGO,SUPN,RPRX,ZTS,JAZZ,AVDL,PFE"},
    {"t":"PBJ","n":"Food & Beverage","fn":"Invesco Food & Beverage","h":"MNST,HSY,KR,SYY,MDLZ,KHC,CTVA,DASH,HLF,UNFI,SEB,IMKTA,TSN,USFD,FDP,CART,CHEF,ADM,AGRO,TR,ACI,DPZ,POST,JBS,WMK"},
    {"t":"XLP","n":"Consumer Staples","fn":"Sel Sector:C SSS SPDR I","h":"ADM,CAG,CHD,CL,CLX,COST,CPB,DG,DLTR,EL,GIS,HRL,HSY,K,KDP,KHC,KMB,KO,KR,KVUE,MDLZ,MNST,MO,PEP,PG,PM,SJM,SYY,TGT,WMT"},
    {"t":"XLK","n":"Technology","fn":"Sel Sector:T SSS SPDR I","h":"NVDA,MSFT,AAPL,AVGO,ORCL,PLTR,CRM,CSCO,IBM,AMD,INTU,QCOM,TXN,ADBE,NOW,ACN,AMAT,MU,ANET,LRCX,KLAC,ADI,INTC,PANW,CRWD,APH,CDNS,SNPS,MSI,ROP"},
    {"t":"ROBO","n":"Robotics & Automation","fn":"Robo Glbl Robots & Auto","h":"SYM,TER,ISRG,SERV,IRBT,COHR,ROK,ILMN,RR,ARBE,AUR,GMED,PRCT,NOVT,PDYN,IPGP,NDSN,EMR,TSLA,NVDA,QCOM,XPEV,AMZN,AMD,GOOGL,BABA,OUST,MBLY,HSAI,ARM,TKR,RBC,ABB,AEVA"},
    {"t":"ARKK","n":"Innovation / Growth","fn":"ARK Innovation","h":"TSLA,ROKU,COIN,TEM,CRSP,SHOP,HOOD,RBLX,AMD,PLTR,BEAM,TER,CRCL,BMNR,ACHR,TXG,TWST,ILMN,AMZN,VCYT,BLSH,NVDA,NTRA,META,DKNG"},
    {"t":"ARKW","n":"Next Gen Internet","fn":"ARK Next Generation Internet ETF","h":"AMD,TSLA,HOOD,CRCL,ROKU,SHOP,AMZN,GOOG,CRWV,COIN,XYZ,PLTR,CRWD,RBLX,DDOG,AVGO,TSM,META,NET,BLSH,P,BMNR,RBRK,FIG,NFLX,SPOT,NVDA,GTLB,BABA,TOST,GENI,CBRS,BIDU,DKNG,DASH,MELI,ABNB,SLMT"},
    {"t":"HYDR","n":"Hydrogen","fn":"Glbl X Hydrogen ETF","h":"BE,PLUG,BLDP,SLDP,FCEL,CMI,APD"},
    {"t":"PEJ","n":"Leisure & Ent","fn":"Invesco Leisure and Ent","h":"WBD,LVS,SYY,CCL,DASH,LYV,RCL,FLUT,LYFT,LION,EXPE,FOXA,CNK,TKO,CPA,USFD,PSKY,CART,BYD,EAT,BH,RRR,DPZ,MGM,ACEL"},
    {"t":"EEM","n":"Emerging Markets","fn":"iShares:MSCI Em Mkts","h":"PDD,BABA,NU,MELI,DLO,JMIA"},
    {"t":"IWO","n":"Small Cap Growth","fn":"iShares:Russ 2000 Gr","h":"CRDO,BE,FN,IONQ,GH,KTOS,BBIO,MDGL,ENSG,NXT,RMBS,SPXC,DY,STRL,GTLS,IDCC,HQY,MOD,AVAV,AEIS,WTS,RGTI,ZWS,HIMS,LUMN"},
    {"t":"MOO","n":"Agribusiness","fn":"VanEck:Agribusiness","h":"DE,ZTS,CTVA,ADM,TSN,CF,BG,ELAN,MOS,CNH,DAR,TTC,AGCO,CAT"},
    {"t":"XME","n":"Metals & Mining","fn":"SS SPDR S&P Metals&Mng","h":"HL,AA,HCC,STLD,NEM,LEU,CDE,NUE,CLF,CMC,RGLD,BTU,CNR,RS,FCX,UEC,MP,AMR,MTRN,CENX,IE,KALU,USAR,WS,MUX"},
    {"t":"ARKF","n":"Fintech Innovation","fn":"ARK BC & Fintech Innov","h":"SHOP,COIN,HOOD,PLTR,TOST,SOFI,XYZ,RBLX,ROKU,CRCL,MELI,AMD,DKNG,META,AMZN,BMNR,PINS,NU,KLAR,SE,BLSH,FUTU,Z"},
    {"t":"EWZ","n":"Brazil","fn":"iShares:MSCI Brazil","h":"NU,MELI,DLO"},
    {"t":"XBI","n":"Biotechnology","fn":"SS SPDR S&P Biotech","h":"EXAS,RVMD,RNA,INSM,NTRA,BBIO,REGN,MDGL,IONS,BIIB,AMGN,UTHR,INCY,ROIV,EXEL,VRTX,GILD,NBIX,ABBV,BMRN,CRSP,MRNA,PTCT,ALNY,KRYS"},
    {"t":"XSD","n":"Semiconductors","fn":"SPDR:S&P Semiconductor ETF","h":"MU,INTC,RGTI,AMD,MRVL,MTSI,FSLR,RMBS,SITM,SMTC,MPWR,ADI,QCOM,CRUS,ON,AVGO,CRDO,LSCC,NVDA,QRVO,SLAB,TXN,NXPI,SWKS,OLED,TSEM,TSM,AMAT,ASML,LRCX,KLAC,CDNS,SNPS,TER,MCHP,STM"},
    {"t":"ARKG","n":"Genomics","fn":"ARK Genomic Revolution","h":"TEM,CRSP,PSNL,GH,TWST,TXG,NTRA,BEAM,ILMN,VCYT,RXRX,ADPT,IONS,ABSI,CDNA,SDGR,NTLA,NRIX,PACB,WGS,BFLY,PRME,ARCT,AMGN"},
    {"t":"GBTC","n":"Bitcoin","fn":"GRAYSCALE BITCOIN TRUST","h":"IBIT,ETHA,MSTR,BMNR,SBET,COIN"},
    {"t":"IGV","n":"Software","fn":"iShares:Expand Tch-Sftwr","h":"ADBE,ADSK,AGYS,APP,APPN,BBAI,CDNS,CIFR,CLSK,CRM,CRNC,CRWD,CTSH,DDOG,EA,EPAM,FICO,FTNT,GDYN,HUT,IBM,IDCC,INTU,JAMF,MSFT,MSTR,NOW,ORCL,PANW,PATH,PLTR,PRO,PTC,QBTS,ROP,SNPS,TDC,TEAM,TTWO,WDAY,WK,WULF,ZM,ZS"},
    {"t":"GRID","n":"Grid Infrastructure","fn":"FT NASDAQ Clean Edge Smart Grid Infrastructure","h":"ETN,JCI,PWR,NVT,HUBB,NVDA,TSLA,CSCO,ORCL,APTV,TXN,GEV,QCOM,IBM,ADI,ENPH,MYRG,HON"},
    {"t":"PHO","n":"Water Infrastructure","fn":"Invesco Water Res","h":"WAT,FERG,ECL,ROP,AWK,MLI,IEX,WMS,XYL,PNR,VLTO,AOS,ACM,CNM,VMI,WTRG,BMI,TTEK,ITRI,WTS,ZWS,MWA,SBS,FELE,HWKN"},
    {"t":"BLOK","n":"Blockchain","fn":"Amplify Blockchain Tech","h":"BBBY,BITB,BKKT,BLK,CAN,CIFR,CLSK,CME,CMPO,COIN,CORZ,CRCL,FBTC,FIGR,GLXY,HOOD,HUT,IBIT,IBM,MELI,NU,OPRA,PYPL,RBLX,WULF,XYZ"},
    {"t":"WCLD","n":"Cloud Computing","fn":"WisdomTree:Cloud Cmptng","h":"FSLY,MDB,DOCN,FROG,PATH,SNOW,BILL,DDOG,CFLT,WK,TWLO,CRWD,PCOR,AGYS,IOT,BRZE,QLYS,CWAN,BL,CLBT,PANW,SHOP,INTA,NET"},
    {"t":"WGMI","n":"Crypto Miners / Data Centers","fn":"CoinShares Btc Mining","h":"APLD,BTBT,BTDR,CAN,CANG,CIFR,CLSK,CORZ,CRWV,GLXY,HIVE,HUT,IREN,MARA,NBIS,NVDA,RIOT,TSM,WULF,XYZ"},
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
    {"t":"SOCL","n":"Social Media","fn":"Global X Social Media ETF","h":"META,RDDT,GOOGL,BIDU,NTES,SPOT,PINS,MTCH,SNAP,BILI,TME,IAC,JOYY,YELP,DJT,RUM,WB,MOMO,CXM,NXDR,GRPN,GRND,SPT,FVRR,BMBL"},
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
    {"t":"EUV","n":"Photonics","fn":"Corgi Lithography & Semiconductor Photonics ETF","h":"TSM,ASML,LRCX,GLW,AMAT,CIEN,LITE,KLAC,COHR,MTSI,AXTI,CRDO,MRVL,FN,NVMI,ENTG,AAOI,VIAV,NOVT,ONTO,LASR,FORM,OUST,VECO,AEVA,UCTT,HSAI,PLAB,IPGP,CAMT,AEHR,PDFS,POET,ADTN,TDY,TSEM,GFS,LWLG,MTRN"},
    {"t":"LIDAR","n":"LiDAR","fn":"Custom Basket","h":"OUST,AEVA,HSAI,TDY","basket":True},
    {"t":"PSEMI","n":"Power Semis","fn":"Custom Basket","h":"STM,ON,MPWR,ADI,TXN,NVTS,WOLF,FLEX,POWI,VRT,ETN,AOSL,SEDG,ENPH,VICR,ALGM,IPWR,AMSC","basket":True},
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


def compute_ema_value(closes, period):
    """Compute a single endpoint EMA value (matches fetch_leaders.py implementation)."""
    if len(closes) < period:
        return None
    k = 2.0 / (period + 1)
    e = float(closes[0])
    for v in closes[1:]:
        e = float(v) * k + e * (1 - k)
    return e


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


def compute_setup_adjustment(c, h, l, n):
    """Evaluate technical setup criteria. Returns (adj, flags) tuple where adj is the
    score adjustment in percentile points and flags is a bitmask of which criteria
    are met. Ported from fetch_leaders.py — must match its bitmask exactly so the
    Overview tab's vcpMedal() logic works on ETFs the same way it does on stocks.
    """
    if n < 50:
        return 0.0, 0
    price = float(c[-1])
    ema9 = compute_ema_value(c, 9)
    ema21 = compute_ema_value(c, 21)
    if ema9 is None or ema21 is None:
        return 0.0, 0
    sma50 = float(np.mean(c[-50:])) if n >= 50 else float(np.mean(c))
    sma100 = float(np.mean(c[-100:])) if n >= 100 else None
    sma200 = float(np.mean(c[-200:])) if n >= 200 else None
    sma50_prev = float(np.mean(c[-55:-5])) if n >= 55 else None
    sma50_rising = (sma50 > sma50_prev) if sma50_prev is not None else None
    sma200_prev = float(np.mean(c[-205:-5])) if n >= 205 else None
    sma200_rising = (sma200 > sma200_prev) if sma200_prev is not None else None
    adr = float(np.mean([h[i] - l[i] for i in range(max(0, n - 14), n)])) if n > 0 else 0

    adj = 0.0
    flags = 0

    # Gold
    GB, GP = 1.5, -2.0
    g1 = ema21 > sma50; adj += GB if g1 else GP
    if g1: flags |= 1
    g2 = price > ema21; adj += GB if g2 else GP
    if g2: flags |= 2
    g3 = price > sma50; adj += GB if g3 else GP
    if g3: flags |= 4
    if sma50_rising is not None:
        adj += GB if sma50_rising else GP
        if sma50_rising: flags |= 8

    # Silver
    SB, SP = 0.75, -1.0
    if sma100 is not None:
        s1 = price > sma100; adj += SB if s1 else SP
        if s1: flags |= 16
        s2 = sma50 > sma100; adj += SB if s2 else SP
        if s2: flags |= 32
    if sma200_rising is not None:
        adj += SB if sma200_rising else SP
        if sma200_rising: flags |= 64

    # Bronze (bonus only)
    BB = 0.5
    if adr > 0 and abs(ema9 - ema21) < (0.5 * adr):
        adj += BB
        flags |= 128
    if sma200 is not None and price > sma200:
        adj += BB
        flags |= 256

    # Overview-tab medal-tier flags (no scoring impact)
    if price > ema9: flags |= 512
    if ema9 > ema21: flags |= 1024
    if sma200 is not None and sma50 > sma200: flags |= 2048

    return round(adj, 2), flags


def compute_trend_zone(c, h, l, spy_closes, spy_ts_map, df_index):
    """Compute the smooth_trend zone label. Ported from fetch_leaders.py.
    Returns one of: bull_strong / bull_light / neutral / bear_light / bear_strong.
    """
    c_arr = np.asarray([x for x in c if x is not None], dtype=float)
    n = len(c_arr)
    if n < 55:
        return "neutral"

    def _ema_series(arr, span):
        if len(arr) < span:
            return None
        k = 2.0 / (span + 1)
        out = [float(arr[0])]
        for v in arr[1:]:
            out.append(float(v) * k + out[-1] * (1 - k))
        return np.array(out)

    def _rsi_series(arr, length=14):
        if len(arr) < length + 1:
            return None
        deltas = np.diff(arr)
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)
        avg_gain = np.mean(gains[:length])
        avg_loss = np.mean(losses[:length])
        rsi_vals = [50.0] * length
        rs = avg_gain / avg_loss if avg_loss > 0 else float('inf')
        rsi_vals.append(100.0 - 100.0 / (1.0 + rs) if avg_loss > 0 else 100.0)
        for i in range(length, len(deltas)):
            avg_gain = (avg_gain * (length - 1) + gains[i]) / length
            avg_loss = (avg_loss * (length - 1) + losses[i]) / length
            if avg_loss == 0:
                rsi_vals.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_vals.append(100.0 - 100.0 / (1.0 + rs))
        return np.array(rsi_vals)

    WINDOW = 10
    rsi_s = _rsi_series(c_arr, 14)
    if rsi_s is None or len(rsi_s) < WINDOW:
        return "neutral"

    ema_f_intra_s = _ema_series(c_arr, 10)
    ema_s_intra_s = _ema_series(c_arr, 21)
    ema_f_day_s   = _ema_series(c_arr, 5)
    ema_s_day_s   = _ema_series(c_arr, 20)
    ema_f_wk_s    = _ema_series(c_arr, 15)
    ema_s_wk_s    = _ema_series(c_arr, 50)
    if ema_s_wk_s is None or ema_s_day_s is None or ema_s_intra_s is None:
        return "neutral"

    bb_len = 20
    bb_mult = 2.0
    bb_score_s = np.zeros(n)
    for i in range(bb_len - 1, n):
        window = c_arr[i - bb_len + 1:i + 1]
        basis = np.mean(window)
        std = np.std(window, ddof=0)
        upper = basis + bb_mult * std
        lower = basis - bb_mult * std
        if upper - lower > 0:
            pos = (c_arr[i] - lower) / (upper - lower) * 100
            bb_score_s[i] = (pos - 50) * 2

    rs_score_s = np.zeros(n)
    for i, ts in enumerate(df_index):
        if i < 1:
            continue
        stock_chg = (c[i] / c[i - 1] - 1) * 100 if c[i] is not None and c[i - 1] else 0
        spy_chg = 0
        if ts in spy_ts_map:
            si = spy_ts_map[ts]
            if si > 0 and spy_closes[si] and spy_closes[si - 1]:
                spy_chg = (spy_closes[si] / spy_closes[si - 1] - 1) * 100
        rs_score_s[i] = max(-100, min(100, (stock_chg - spy_chg) * 10))

    trend_scores = []
    for offset in range(WINDOW, 0, -1):
        idx = n - offset
        if idx < 50:
            continue
        rsi_val = rsi_s[idx] if idx < len(rsi_s) else 50
        rsi_score = (rsi_val - 50) * 2
        if ema_s_intra_s[idx] > 0:
            md_intra = (ema_f_intra_s[idx] - ema_s_intra_s[idx]) / ema_s_intra_s[idx] * 100
            ma_intra = max(-100, min(100, md_intra * 5))
        else:
            ma_intra = 0
        if ema_s_day_s[idx] > 0:
            md_day = (ema_f_day_s[idx] - ema_s_day_s[idx]) / ema_s_day_s[idx] * 100
            ma_day = max(-100, min(100, md_day * 5))
        else:
            ma_day = 0
        if ema_s_wk_s[idx] > 0:
            md_wk = (ema_f_wk_s[idx] - ema_s_wk_s[idx]) / ema_s_wk_s[idx] * 100
            ma_wk = max(-100, min(100, md_wk * 5))
        else:
            ma_wk = 0
        ma_score = (ma_intra + ma_day + ma_wk) / 3
        bb_score = bb_score_s[idx] if idx < len(bb_score_s) else 0
        rs_score = rs_score_s[idx] if idx < len(rs_score_s) else 0
        ts_score = (rsi_score * 0.25) + (ma_score * 0.35) + (bb_score * 0.20) + (rs_score * 0.20)
        trend_scores.append(ts_score)

    if not trend_scores:
        return "neutral"

    k = 2.0 / (5 + 1)
    smooth = trend_scores[0]
    for v in trend_scores[1:]:
        smooth = v * k + smooth * (1 - k)

    if smooth > 30:    return "bull_strong"
    elif smooth > 10:  return "bull_light"
    elif smooth < -30: return "bear_strong"
    elif smooth < -10: return "bear_light"
    else:              return "neutral"


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
    """Load dynamic holdings for BUZZ from dynamic_holdings.json."""
    if os.path.exists("dynamic_holdings.json"):
        with open("dynamic_holdings.json") as f:
            return json.load(f)
    return {}


def close_on_or_before(df_index, closes, target_date):
    """Find the close of the most recent trading day at or before target_date.
    Returns a float or None. Used for calendar-based 1W/1M baselines that match
    MarketWatch / Yahoo / TradingView conventions.
    """
    for i in range(len(df_index) - 1, -1, -1):
        ts = df_index[i]
        d = ts.date() if hasattr(ts, 'date') else None
        if d is None:
            continue
        if d <= target_date:
            v = closes[i]
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    return None
    return None


# ─── Intraday overlay helpers ────────────────────────────────────────────────

def fetch_live_quote(ticker):
    """Fetch live (intraday) quote for a single ticker via fast_info.
    Returns (last_price, prev_close) or (None, None) on failure.
    """
    try:
        ti = yf.Ticker(ticker, session=_session) if _session else yf.Ticker(ticker)
        fi = ti.fast_info
        last = fi.get("lastPrice", fi.get("last_price"))
        prev = fi.get("previousClose", fi.get("previous_close"))
        last = float(last) if last not in (None, 0) else None
        prev = float(prev) if prev not in (None, 0) else None
        return last, prev
    except Exception:
        return None, None


def fetch_live_quotes_bulk(tickers, max_workers=20):
    """Parallel-fetch live quotes for many tickers. Returns {ticker: (last, prev)}."""
    out = {}
    if not tickers:
        return out
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_live_quote, t): t for t in tickers}
        for fut in as_completed(futs):
            tk = futs[fut]
            try:
                last, prev = fut.result(timeout=15)
                if last is not None and prev is not None:
                    out[tk] = (last, prev)
            except Exception:
                pass
    return out


def is_us_market_open():
    """Return True only during US regular session (9:30am–3:55pm ET, Mon–Fri).

    IMPORTANT: Post-close, return False. fast_info.lastPrice and fast_info.previousClose
    are unreliable outside regular hours — they can reflect after-hours activity or
    the wrong reference close, corrupting 1D calculations. Post-close, the daily df from
    yf.download is the canonical source of truth.
    """
    try:
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
        et_hour = (now_utc.hour - 4) % 24
        et_min  = now_utc.minute
        et_min_of_day = et_hour * 60 + et_min
        if now_utc.weekday() >= 5:
            return False
        # 9:30am (570) to 3:55pm (955) ET — stop 5 min early so the 4:00pm scheduled
        # run uses the EOD df values instead.
        return 570 <= et_min_of_day <= 955
    except Exception:
        return False


def _overlay_one(entry, live, prev_close):
    """Apply live price to a single result entry, recomputing ch/c5/c20/ytd/p.
    Returns False and leaves entry untouched if values are missing or suspicious.
    """
    if live is None or live <= 0 or prev_close is None or prev_close <= 0:
        return False
    eod = entry.get("p")
    if eod and (live > eod * 1.5 or live < eod * 0.5):
        # Sanity check — reject suspicious live values vs EOD snapshot
        return False
    candidate_ch = (live / prev_close - 1) * 100
    if candidate_ch > 75 or candidate_ch < -75:
        # Reject implausible 1D — usually indicates fast_info returned a stale reference close
        return False
    entry["p"] = round(live, 2)
    entry["ch"] = round(candidate_ch, 2)
    if entry.get("_5b") and entry["_5b"] > 0:
        entry["c5"] = round((live / entry["_5b"] - 1) * 100, 2)
    if entry.get("_20b") and entry["_20b"] > 0:
        entry["c20"] = round((live / entry["_20b"] - 1) * 100, 2)
    if entry.get("_yb") and entry["_yb"] > 0:
        entry["ytd"] = round((live / entry["_yb"] - 1) * 100, 2)
    return True


def apply_intraday_overlay_to_etfs(results, component_metrics):
    """Refresh ch/c5/c20/ytd/p in `results` so they reflect live intraday prices.

    Two paths:
      - Regular ETFs (yt or t ticker): fetch live quote for the ETF ticker itself
      - Custom baskets (no real ticker): re-average from live-refreshed component_metrics

    Only runs during US market hours. RS / VARS are NOT recomputed (anchored to EOD).
    """
    if not is_us_market_open():
        print("  [Intraday overlay] Skipping — US market not open.")
        return

    # ---- 1. Refresh component (holding) metrics with live quotes ----
    component_tickers = list(component_metrics.keys())
    print(f"  [Intraday overlay] Fetching live quotes for {len(component_tickers)} basket components...")
    t0 = time.time()
    comp_quotes = fetch_live_quotes_bulk(component_tickers)
    print(f"  [Intraday overlay] Got {len(comp_quotes)} component quotes in {time.time()-t0:.1f}s")
    comp_refreshed = 0
    for tk, m in component_metrics.items():
        q = comp_quotes.get(tk)
        if not q:
            continue
        if _overlay_one(m, q[0], q[1]):
            comp_refreshed += 1
    print(f"  [Intraday overlay] Refreshed {comp_refreshed}/{len(component_metrics)} components")

    # ---- 2. Identify ETF tickers to refresh (the ones in results that have prices) ----
    etf_tickers = []
    etf_lookup = {}
    for r in results:
        if r.get("p") is None:
            # Custom basket — no own price to fetch; will be recomputed by averaging components
            continue
        # Use yt override if present (matches what was downloaded for the daily df)
        tk = r.get("yt") or r.get("t")
        etf_tickers.append(tk)
        etf_lookup.setdefault(tk, []).append(r)

    print(f"  [Intraday overlay] Fetching live quotes for {len(etf_tickers)} ETFs...")
    t0 = time.time()
    etf_quotes = fetch_live_quotes_bulk(etf_tickers)
    print(f"  [Intraday overlay] Got {len(etf_quotes)} ETF quotes in {time.time()-t0:.1f}s")
    etf_refreshed = 0
    for tk, entries in etf_lookup.items():
        q = etf_quotes.get(tk)
        if not q:
            continue
        for r in entries:
            if _overlay_one(r, q[0], q[1]):
                etf_refreshed += 1
    print(f"  [Intraday overlay] Refreshed {etf_refreshed}/{len(etf_tickers)} ETF entries")

    # ---- 3. Re-average basket entries from refreshed components ----
    basket_refreshed = 0
    for r in results:
        if r.get("p") is not None:
            continue  # not a basket
        holdings_str = r.get("h", "")
        if not holdings_str:
            continue
        holdings = [h.strip() for h in holdings_str.split(",") if h.strip()]
        valid = [h for h in holdings if h in component_metrics]
        if not valid:
            continue
        for key in ("ch", "c5", "c20", "ytd"):
            vals = [component_metrics[h][key] for h in valid if component_metrics[h].get(key) is not None]
            if vals:
                r[key] = round(float(np.mean(vals)), 2)
        basket_refreshed += 1
    print(f"  [Intraday overlay] Re-averaged {basket_refreshed} custom baskets from live components")


def strip_internal_fields_etfs(results, *extra_dicts):
    """Remove _-prefixed internal fields, `fr`/`w_fr` raw scoring values, and the `h`
    (holdings string) before serializing.

    `fr`/`w_fr` are raw VARS endpoints used internally for cross-sectional ranking;
    the frontend only displays the percentile-ranked `rs`/`w_rs`.

    `extra_dicts` is a tuple of dicts (e.g. component_metrics) whose values are also
    cleaned, even though they aren't written directly — defensive in case anything
    serializes them later.
    """
    INTERNAL_NON_UNDERSCORE = {"fr", "w_fr"}
    for r in results:
        for k in list(r.keys()):
            if k.startswith("_") or k in INTERNAL_NON_UNDERSCORE:
                del r[k]
    for d in extra_dicts:
        for v in d.values():
            if isinstance(v, dict):
                for k in list(v.keys()):
                    if k.startswith("_") or k in INTERNAL_NON_UNDERSCORE:
                        del v[k]


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    # ─── Holiday check ────────────────────────────────────────────
    # If today (ET) is a US market holiday, exit cleanly without doing any work.
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    # Hardcoded list — must match fetch_leaders.py's US_MARKET_HOLIDAYS exactly.
    # Last updated: 2026-04-29 (covers through 2028).
    _US_MARKET_HOLIDAYS = {
        "2026-01-01","2026-01-19","2026-02-16","2026-04-03","2026-05-25",
        "2026-06-19","2026-07-03","2026-09-07","2026-11-26","2026-12-25",
        "2027-01-01","2027-01-18","2027-02-15","2027-03-26","2027-05-31",
        "2027-06-18","2027-07-05","2027-09-06","2027-11-25","2027-12-24",
        "2028-01-17","2028-02-21","2028-04-14","2028-05-29","2028-06-19",
        "2028-07-04","2028-09-04","2028-11-23","2028-12-25",
    }
    _US_HOLIDAYS_VALID_THROUGH = 2028
    _et_today = _dt.now(_tz.utc).astimezone(_tz(_td(hours=-4))).date().isoformat()
    _year = int(_et_today[:4])
    if _year > _US_HOLIDAYS_VALID_THROUGH:
        print(f"  [WARNING] Holiday list expires after {_US_HOLIDAYS_VALID_THROUGH}; "
              f"date {_et_today} is unchecked. Please update _US_MARKET_HOLIDAYS in fetch_data.py.")
    if _et_today in _US_MARKET_HOLIDAYS:
        print(f"\n[holiday] {_et_today} is a US market holiday — exiting without changes.")
        return

    # Override dynamic ETF holdings (e.g. BUZZ) from dynamic file. Any keys in
    # the dynamic file for ETFs no longer in ETF_INFO (e.g. FFTY, removed) are
    # silently ignored.
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
    # Need enough bars to (1) compute 21-day change/ATR/etc. and (2) include the last bar of the
    # previous calendar year so YTD can be computed. 140 days covers Jan 1 with a comfortable buffer.
    start = end - timedelta(days=140)

    raw = yf.download(
        all_download,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        # auto_adjust=False so historical closes are UNADJUSTED (raw). This matches
        # Yahoo Finance's quote-page % changes (1D/1W/1M/YTD) exactly. With auto_adjust=True
        # yfinance silently lowers historical closes by any subsequent dividends, inflating
        # the computed YTD relative to Yahoo's displayed YTD.
        auto_adjust=False,
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
        auto_adjust=False,
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

        # ─── 1W / 1M via CALENDAR-date lookback ─────────────────────────
        # Matches MarketWatch / Yahoo / TradingView convention: "close closest to N calendar
        # days ago", with reference = next business day after the last completed session so
        # the week-ago baseline advances when a session closes.
        try:
            _last_ts = df.index[-1]
            _last_date = _last_ts.date() if hasattr(_last_ts, 'date') else None
        except Exception:
            _last_date = None
        if _last_date is not None:
            try:
                import pandas as _pd
                _today_ref = (_last_ts + _pd.offsets.BDay(1)).date()
            except Exception:
                _today_ref = _last_date + timedelta(days=1)
        else:
            from datetime import date as _d
            _today_ref = _d.today()
        _w_target = _today_ref - timedelta(days=7)
        _m_target = _today_ref - timedelta(days=30)
        w_base = close_on_or_before(df.index, c, _w_target)
        m_base = close_on_or_before(df.index, c, _m_target)
        c5  = ((c[-1] / w_base) - 1) * 100 if (w_base and w_base > 0) else None
        c20 = ((c[-1] / m_base) - 1) * 100 if (m_base and m_base > 0) else None

        # YTD: price change vs last bar of previous calendar year.
        # Falls back to the first available bar of the current year if the data series
        # doesn't extend into the prior year (e.g. very recent IPOs).
        from datetime import date as _date
        current_year = _date.today().year
        ytd = None
        ytd_base = None
        try:
            for i, ts in enumerate(df.index):
                bar_year = ts.year if hasattr(ts, 'year') else int(str(ts)[:4])
                if bar_year >= current_year:
                    if i > 0 and c[i - 1] is not None and c[-1] is not None and c[i - 1] != 0:
                        # Standard case: use the last bar of the previous year as baseline
                        ytd_base = float(c[i - 1])
                        ytd = (c[-1] / ytd_base - 1) * 100
                    elif c[i] is not None and c[-1] is not None and c[i] != 0:
                        # Fallback: data series begins in current year (IPO or insufficient lookback).
                        # Use the first available bar of this year as baseline.
                        ytd_base = float(c[i])
                        ytd = (c[-1] / ytd_base - 1) * 100
                    break
        except Exception:
            ytd = None
            ytd_base = None

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
                "ytd": round(ytd, 2) if ytd is not None else None,
                "rs": None, "rf": 0, "ra": 0, "p": round(price, 2),
                "fr": None, "vs": None,
                # Internal baselines for intraday overlay (stripped before output)
                "_pc": float(c[-2]) if length >= 2 else None,
                "_5b": w_base,
                "_20b": m_base,
                "_yb": ytd_base,
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

        # ─── Setup flags + trend zone (for Overview tab VCP detection) ──
        # Mirrors the stock screener's sf/tz pipeline so vcpMedal() can identify
        # coiled ETFs the same way it identifies coiled stocks.
        try:
            _sa, _sf = compute_setup_adjustment(c, h, l, length)
        except Exception:
            _sa, _sf = 0.0, 0
        try:
            _tz = compute_trend_zone(c, h, l, spy_closes, spy_ts_map, df.index)
        except Exception:
            _tz = "neutral"

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
            "ytd": round(ytd, 2) if ytd is not None else None,
            # rs (final 0-100 percentile) is assigned cross-sectionally in main()
            # so the score reflects this ETF's strength relative to the OTHER ETFs,
            # not just relative to its own past values.
            "rs": None,
            "rf": dec_streak,
            "ra": adv_streak,
            "p": round(price, 2),
            "fr": round(final_rs, 4),
            "vs": vs_series,
            # Setup flags + trend zone for Overview-tab VCP detection
            "sf": _sf,
            "tz": _tz,
            # Internal baselines for intraday overlay (stripped before output)
            "_pc": float(c[-2]) if length >= 2 else None,
            "_5b": w_base,
            "_20b": m_base,
            "_yb": ytd_base,
        }

    def process_ticker_weekly(ticker, df_w):
        """Process weekly data for a ticker — same VARS logic but on weekly bars."""
        if df_w is None or len(df_w) < 5:
            return {"w_rv": None, "w_am": None, "w_rs": None, "w_rf": 0, "w_ra": 0, "w_vs": None, "w_fr": None}

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
                    "w_rs": None, "w_rf": 0, "w_ra": 0, "w_vs": None, "w_fr": None}

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
            # w_rs is assigned cross-sectionally in main() based on w_fr
            "w_rs": None,
            "w_rf": dec_streak,
            "w_ra": adv_streak,
            "w_vs": vs_series,
            "w_fr": round(final_rs, 4),
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
                            "c20": None, "ytd": None, "rs": None, "rf": 0, "ra": 0, "p": None,
                            "fr": None, "vs": None, "sf": 0, "tz": "neutral", **w_metrics})
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
                            "c20": None, "ytd": None, "rs": None, "rf": 0, "ra": 0, "p": None,
                            "fr": None, "vs": None, "sf": 0, "tz": "neutral",
                            "w_rv": None, "w_am": None, "w_rs": None, "w_rf": 0, "w_ra": 0, "w_vs": None,
                            "w_fr": None})
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
            rs_series = []
            final_rs = 0

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
            w_rs_series = []
            w_final_rs = 0

        # Average scalar metrics
        def avg_metric(key):
            vals = [component_metrics[h][key] for h in valid if component_metrics[h].get(key) is not None]
            return round(np.mean(vals), 2) if vals else None

        def avg_w_metric(key):
            vals = [component_w_metrics[h][key] for h in valid_w if component_w_metrics[h].get(key) is not None]
            return round(np.mean(vals), 2) if vals else None

        # ─── Build synthetic equal-weighted basket OHLC series for sf/tz computation ──
        # Each component's daily Close/High/Low series is normalised to start at 100,
        # then averaged across components to form a synthetic basket price series.
        # We then run the standard setup/trend functions on this series as if it were
        # a single ticker. Skipped for baskets with insufficient component history.
        # Also persist a trimmed close series + dates as `bp`/`bd` so the frontend can
        # draw a custom price chart for the basket (real ETFs use TradingView, baskets
        # use this since they have no listed ticker for TV to look up).
        _basket_sf = 0
        _basket_tz = "neutral"
        _basket_price_series = None
        _basket_dates = None
        try:
            comp_dfs = []
            for h in valid:
                _df = get_df(h)
                if _df is None or len(_df) < 50:
                    continue
                comp_dfs.append((h, _df))
            if comp_dfs:
                # Use the SHORTEST component series to avoid alignment issues.
                # Synthetic series gets the last min_len bars of each component.
                min_len = min(len(d) for _, d in comp_dfs)
                if min_len >= 50:
                    syn_c = np.zeros(min_len)
                    syn_h = np.zeros(min_len)
                    syn_l = np.zeros(min_len)
                    n_components = 0
                    for _, _df in comp_dfs:
                        cv = _df["Close"].values[-min_len:]
                        hv = _df["High"].values[-min_len:]
                        lv = _df["Low"].values[-min_len:]
                        if cv[0] is None or cv[0] <= 0:
                            continue
                        scale = 100.0 / float(cv[0])
                        syn_c += np.array([float(x) * scale for x in cv])
                        syn_h += np.array([float(x) * scale for x in hv])
                        syn_l += np.array([float(x) * scale for x in lv])
                        n_components += 1
                    if n_components > 0:
                        syn_c /= n_components
                        syn_h /= n_components
                        syn_l /= n_components
                        # Use the FIRST valid component's DataFrame index for SPY alignment
                        ref_df = comp_dfs[0][1]
                        ref_index = ref_df.index[-min_len:]
                        try:
                            _, _basket_sf = compute_setup_adjustment(syn_c, syn_h, syn_l, min_len)
                        except Exception:
                            _basket_sf = 0
                        try:
                            _basket_tz = compute_trend_zone(syn_c, syn_h, syn_l, spy_closes, spy_ts_map, ref_index)
                        except Exception:
                            _basket_tz = "neutral"
                        # Persist the synthetic close series + matching dates for the frontend.
                        # Trim to the last ~150 trading days (~7 months) — enough to give the
                        # chart visual context without bloating data.json. Round to 2dp since
                        # the series is normalised to start at 100 (small/scale-invariant).
                        _BP_TRIM = 150
                        _bp_full = [round(float(x), 2) for x in syn_c.tolist()]
                        _bp_dates_full = [str(d.date()) if hasattr(d, "date") else str(d)[:10]
                                          for d in ref_index]
                        if len(_bp_full) > _BP_TRIM:
                            _basket_price_series = _bp_full[-_BP_TRIM:]
                            _basket_dates = _bp_dates_full[-_BP_TRIM:]
                        else:
                            _basket_price_series = _bp_full
                            _basket_dates = _bp_dates_full
        except Exception as _e:
            print(f"  [warn] Could not compute synthetic sf/tz for {info.get('t','?')}: {_e}")
            _basket_sf = 0
            _basket_tz = "neutral"
            _basket_price_series = None
            _basket_dates = None

        results.append({
            **info,
            "rv": round(avg_metric("rv")) if avg_metric("rv") is not None else None,
            "am": round(avg_metric("am")) if avg_metric("am") is not None else None,
            "ax": round(avg_metric("ax")) if avg_metric("ax") is not None else None,
            "ch": avg_metric("ch"),
            "c5": avg_metric("c5"),
            "c20": avg_metric("c20"),
            "ytd": avg_metric("ytd"),
            # rs/w_rs assigned cross-sectionally in main() based on fr/w_fr
            "rs": None,
            "rf": dec_streak,
            "ra": adv_streak,
            "p": None,
            "fr": round(final_rs, 4) if rs_series else None,
            "vs": avg_vs,
            # Setup flags + trend zone for Overview-tab VCP detection.
            # Built from a synthetic equal-weighted basket price series: each component's
            # daily OHLC series is normalised to start at 100, then averaged across
            # components to form a single synthetic price series. MA/EMA/setup logic is
            # then run on this series as if it were a regular ticker. This is heavier
            # than aggregating per-component flags but more accurately reflects the
            # collective trend behaviour of the basket.
            "sf": _basket_sf,
            "tz": _basket_tz,
            "w_rv": round(avg_w_metric("w_rv")) if avg_w_metric("w_rv") is not None else None,
            "w_am": round(avg_w_metric("w_am")) if avg_w_metric("w_am") is not None else None,
            "w_rs": None,
            "w_rf": w_dec_streak,
            "w_ra": w_adv_streak,
            "w_vs": w_avg_vs,
            "w_fr": round(w_final_rs, 4) if w_rs_series else None,
            # Synthetic basket close series + matching ISO dates, captured above.
            # Frontend uses these to draw a custom price chart for baskets (since
            # baskets have no listed ticker for TradingView). Both fields are None
            # when component history was insufficient.
            "bp": _basket_price_series,
            "bd": _basket_dates,
        })

    # ─── Intraday overlay: replace EOD price with live quote when market is open ──
    # Updates p / ch / c5 / c20 / ytd to reflect the current intraday move. RS scores
    # remain anchored to the last completed daily bar (recomputing intraday would require
    # re-running the full VARS pipeline, which is expensive).
    apply_intraday_overlay_to_etfs(results, component_metrics)

    # ─── Cross-sectional RS percentrank ─────────────────────────
    # Each ETF's `fr` is its raw VARS endpoint — a directional value that's only meaningful
    # when ranked against other ETFs. Compute percentile rank across the whole ETF universe
    # so `rs` (0-100) reflects this ETF's strength RELATIVE TO OTHER ETFs, not relative to
    # its own historical values. Same approach as the Stock Screener uses.
    all_d_fr = [r["fr"] for r in results if r.get("fr") is not None]
    all_w_fr = [r["w_fr"] for r in results if r.get("w_fr") is not None]
    for r in results:
        if r.get("fr") is not None and len(all_d_fr) > 1:
            r["rs"] = round(percentrank_inc(all_d_fr, r["fr"]) * 100)
        if r.get("w_fr") is not None and len(all_w_fr) > 1:
            r["w_rs"] = round(percentrank_inc(all_w_fr, r["w_fr"]) * 100)

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

    # ─── Daily rank history (data_score_history.json) ────────────
    # Stores per-trading-day snapshots of daily and weekly ranks for each ETF, in the
    # same shape as the Stock Screener's leaders_score_history.json. The frontend uses
    # this to compute Δ ranks at lookbacks of 1-10 trading days.
    # Structure: {"dates": ["2026-04-21", ...], "d": {"XSD": {"r": [...], "wr": [...]}, ...}}
    from datetime import date, datetime as _dt, timezone as _tz, timedelta as _td

    # Use US Eastern date (UTC-4 EDT approximation), not UTC date — matches screener.
    _et_now = _dt.now(_tz.utc).astimezone(_tz(_td(hours=-4)))
    today_str = _et_now.date().isoformat()
    _weekday = _et_now.weekday()  # Mon=0 ... Sun=6
    skip_history_write = _weekday >= 5
    if skip_history_write:
        print(f"[history] Skipping ETF history write — {today_str} is a weekend in ET (weekday={_weekday})")

    score_history = {"dates": [], "d": {}}
    if os.path.exists("data_score_history.json"):
        try:
            with open("data_score_history.json") as f:
                score_history = json.load(f)
        except Exception:
            score_history = {"dates": [], "d": {}}

    dates_list = score_history.get("dates", [])
    scores = score_history.get("d", {})

    if skip_history_write:
        # Don't touch the history at all on weekends — preserve existing file.
        pass
    elif not dates_list or dates_list[-1] != today_str:
        # New trading day → append a new entry to every ticker's r/wr arrays
        dates_list.append(today_str)
        for r in results:
            tk = r["t"]
            if tk not in scores:
                # New ticker: pad with Nones for prior dates so array lengths align
                scores[tk] = {"r": [None] * (len(dates_list) - 1), "wr": [None] * (len(dates_list) - 1)}
            scores[tk]["r"].append(r.get("rk"))
            scores[tk]["wr"].append(r.get("w_rk"))
        # Also pad any tickers that were in history but not in today's results
        for tk, rec in scores.items():
            if len(rec.get("r", [])) < len(dates_list):
                rec["r"].append(None)
            if len(rec.get("wr", [])) < len(dates_list):
                rec["wr"].append(None)
    else:
        # Same trading day → update last entry in place (intraday refresh)
        for r in results:
            tk = r["t"]
            if tk not in scores:
                scores[tk] = {"r": [None] * len(dates_list), "wr": [None] * len(dates_list)}
            scores[tk]["r"][-1] = r.get("rk")
            scores[tk]["wr"][-1] = r.get("w_rk")

    # Trim to last 30 trading days to keep the file lean (frontend lookback maxes at 10)
    MAX_HISTORY = 30
    if len(dates_list) > MAX_HISTORY:
        excess = len(dates_list) - MAX_HISTORY
        dates_list = dates_list[excess:]
        for tk, rec in scores.items():
            if "r" in rec: rec["r"] = rec["r"][excess:]
            if "wr" in rec: rec["wr"] = rec["wr"][excess:]

    score_history["dates"] = dates_list
    score_history["d"] = scores
    with open("data_score_history.json", "w") as shf:
        json.dump(score_history, shf, separators=(",", ":"))
    print(f"ETF score history: {len(dates_list)} trading days, {len(scores)} tickers")

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
    }

    # Strip internal _-prefixed baseline fields used by the intraday overlay so they
    # don't bloat the output JSON.
    strip_internal_fields_etfs(results, component_metrics)

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
