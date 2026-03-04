#!/usr/bin/env python3
"""
Fetch ETF data from Yahoo Finance, compute ATR-adjusted RS vs SPY,
grade individual holdings by moving average structure, and write data.json.

Usage:
    pip install yfinance numpy
    python fetch_data.py
"""

import json
import sys
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf

BENCHMARK = "SPY"
LOOKBACK = 25
ATR_PERIOD = 14

ETF_INFO = [
    {"t":"XTL","n":"Telecom","fn":"SS SPDR S&P Telecom","h":"LITE,CIEN,ASTS,ONDS,LUMN,VIAV,COMM,GSAT,VSAT,CSCO,TDS,FYBR,VZ,UI,IRDM,AAOI,T,CALX,TMUS,ANET,MSI,FFIV,EXTR,NTCT,CCOI"},
    {"t":"XOP","n":"Oil & Gas Exploration","fn":"SS SPDR S&P Oil&Gas Exp","h":"CNX,EXE,MUR,GPOR,EQT,VLO,RRC,FANG,CTRA,AR,APA,MPC,DVN,PSX,XOM,DINO,PR,OVV,VNOM,MGY,CVX,COP,PBF,OXY,EOG,SHEL"},
    {"t":"REMX","n":"Rare Earth Metals","fn":"VanEck:RE & Str Metals","h":"ALB,SQM,LAC,MP,UUUU,USAR,SGML,DNN,UEC,CCJ,LAR,IDR,IPX,TROX"},
    {"t":"CRAK","n":"Oil Refining","fn":"VanEck:Oil Refiners","h":"MPC,PSX,VLO,DINO,PBF,DK,PARR,CLMT"},
    {"t":"ITA","n":"Aerospace & Defense","fn":"iShares:US Aer&Def ETF","h":"GE,RTX,BA,HWM,TDG,GD,LHX,LMT,NOC,AXON,CW,WWD,RKLB,BWXT,CRS,TXT,ATI,HEI,KTOS,HII,AVAV,HXL,KRMN"},
    {"t":"FCG","n":"Natural Gas","fn":"FT:Natural Gas","h":"AR,EXE,DVN,EQT,FANG,CTRA,APA,EOG,OXY,SM,COP,HESM"},
    {"t":"XAR","n":"Aerospace & Defense","fn":"SS SPDR S&P Aero&Def","h":"ATI,CRS,WWD,HII,KTOS,CW,RTX,HWM,HXL,BWXT,TDG,AVAV,GD,GE,TXT,LHX,HEI,LMT,NOC,ACHR,RKLB,BA,KRMN,AXON,SARO"},
    {"t":"BOAT","n":"Maritime & Shipping","fn":"Tidal:SS Glb Ship","h":"FRO,MATX,STNG,INSW,ZIM,SBLK,CMDB,DHT,CMRE,DAC,TNK,NAT"},
    {"t":"RSPG","n":"Energy (EW)","fn":"Invesco S&P500 EW En","h":"KMI,OXY,APA,WMB,EQT,CTRA,OKE,FANG,BKR,TRGP,XOM,COP,TPL,EOG,CVX,HAL,HES,SLB,PSX,DVN,VLO,MPC,SHEL"},
    {"t":"XLE","n":"Energy","fn":"Sel Sector:Enrgy SS SPDR","h":"XOM,CVX,COP,WMB,MPC,EOG,PSX,VLO,SLB,KMI,BKR,OKE,TRGP,EQT,OXY,FANG,EXE,DVN,HAL,CTRA,TPL,APA"},
    {"t":"FFTY","n":"IBD 50","fn":"Innovator IBD 50","h":"ANAB,CLS,ARQT,STOK,FIX,MU,AU,TARS,APH,TVTX,AGI,KGC,GH,VRT,AEM,LLY,HWM,ARGX,ONC,WGS,ZYME,FN,TMDX"},
    {"t":"AMLP","n":"Energy Infrastructure","fn":"Alerian MLP","h":"MPLX,WES,EPD,SUN,PAA,ET,HESM,CQP,USAC,GEL,SPH,GLP,DKL"},
    {"t":"IYZ","n":"Telecom","fn":"iShares:US Telecom ETF","h":"CSCO,T,VZ,LITE,ASTS,CIEN,ANET,TMUS,TIGO,CMCSA,FYBR,ROKU,IRDM,MSI,UI,CHTR,GLIBK"},
    {"t":"XLI","n":"Industrials","fn":"Sel Sector:Ind SS SPDR","h":"GE,CAT,RTX,UBER,GEV,BA,UNP,ETN,HON,DE,PH,ADP,TT,MMM,LMT,GD,HWM,WM,TDG,JCI,EMR,NOC,UPS,CMI,PWR"},
    {"t":"RSPN","n":"Industrials (EW)","fn":"Invesco S&P500 EW In","h":"BA,TDG,LUV,GE,UAL,MMM,GD,HON,PAYX,J,UBER,RTX,FDX,HII,NOC,ADP,UNP,VLTO,ROK,UPS,EFX,VRSK,GEV,LMT,AME"},
    {"t":"GNR","n":"Natural Resources","fn":"SS SPDR S&P Glbl Nat Res","h":"XOM,NEM,CVX,CTVA,FCX,ADM,VALE,AEM"},
    {"t":"FAN","n":"Wind Energy","fn":"FT II:Global Wind Energy","h":"CWEN,NEE,ACA,LNT,TKR"},
    {"t":"RSPC","n":"Comm Services (EW)","fn":"Invesco S&P 500 EW CS","h":"MTCH,TKO,FYBR,WBD,LYV,TTWO,T,DIS,NFLX,NYT,CMCSA,TMUS,VZ,META,OMC,IPG,PARA,CHTR,EA,NWSA,FOXA,GOOGL,FOX,NWS"},
    {"t":"IYT","n":"Transportation","fn":"iShares:US Transportatn","h":"UBER,UNP,UPS,FDX,CSX,NSC,DAL,UAL,ODFL,EXPD,CHRW,LUV,XPO,JBHT,AAL,LYFT,SAIA,KNX,JOBY,R,KEX,GXO,ALK,LSTR"},
    {"t":"XES","n":"Oil & Gas E&S","fn":"SS SPDR S&P Oil&Gas E&S","h":"LBRT,HP,RIG,WFRD,SEI,NOV,HAL,FTI,VAL,PTEN,BKR,KGS,NE,SLB,WHD,AROC,TDW,SDRL,OII,XPRO,PUMP,AESI,NBR,HLX,WTTR"},
    {"t":"RSP","n":"S&P 500 Equal Weight","fn":"Invesco S&P500 EWght","h":"WBD,ALB,WDC,MU,INTC,TER,AMAT,LRCX,STX,LLY,CAH,REGN,WAT,AMD,CAT,FSLR,UHS,HCA,GM,BIIB,ISRG,JBHT,TECH,RVTY,STLD"},
    {"t":"GDX","n":"Gold Miners","fn":"VanEck:Gold Miners","h":"PAAS,WPM,KGC,AGI,FNV,AEM,B,GFI,NEM"},
    {"t":"GDXJ","n":"Jr Gold Miners","fn":"VanEck:Jr Gold Miners","h":"PAAS,KGC,EQX,AGI,HMY,IAG,RGLD,USAU,GORO,CDE,GLDG"},
    {"t":"SIL","n":"Silver Miners","fn":"Glbl X Silver Miners ETF","h":"EXK,AG,PAAS,HL,FSM,WPM,HCHDF,OR,SVM,SSRM,CDE"},
    {"t":"SILJ","n":"Jr Silver Miners","fn":"Amplify Jr Slvr Miners","h":"AG,HYMC,HL,PAAS,OR,FSM,EXK,GORO,SSRM,CDE"},
    {"t":"RSPU","n":"Utilities (EW)","fn":"Invesco S&P500 EW Ut","h":"CNP,ETR,EXC,PCG,SRE,CMS,AEE,DTE,NI,SO,XEL,ATO,D,PPL,PEG,WEC,EVRG,FE,DUK,AEP,LNT,EIX,NEE,ED,NRG"},
    {"t":"AIPO","n":"AI & Power Infra","fn":"Defiance AI & Pow Infra","h":"PWR,VRT,GEV,ETN,CCJ,AVGO,NVDA,CEG,BE,HUBB,FLNC,AMD,OKLO,MTZ,VST,NVT,EOSE,BWXT"},
    {"t":"XHS","n":"Healthcare Services","fn":"SS SPDR S&P Hlth Cr Svc","h":"GH,PACS,BKD,NEO,CAH,WGS,UHS,BTSG,HCA,SEM,COR,MCK,CNC,ALHC,RDNT,THC,HQY,PGNY,ENSG,ADUS,HSIC,OPCH,ELV,CVS,PRVA"},
    {"t":"OIH","n":"Oil Services","fn":"VanEck:Oil Services","h":"RIG,WFRD,FTI,NE,BKR,NOV,VAL,TS,SLB,PUMP,HAL"},
    {"t":"RSPH","n":"Healthcare (EW)","fn":"Invesco S&P500 EW HC","h":"CNC,PFE,WST,BDX,EW,ABBV,GILD,BMY,CAH,ABT,LH,BSX,MCK,MRK,ZBH,DXCM,DVA,STE,IQV,JNJ,TMO,LLY,ISRG,DHR,DGX"},
    {"t":"KIE","n":"Insurance","fn":"SS SPDR S&P Insurance","h":"LMND,BHF,MCY,WTM,RYAN,ORI,L,RNR,AGO,CINF,MKL,AIZ,TRV,CB,PFG,AXS,ALL,THG,CNO,HIG,PLMR,ACGL,PRU,AFL,UNM"},
    {"t":"SLX","n":"Steel","fn":"VanEck:Steel","h":"VALE,STLD,NUE,MT,CLF,ATI,RIO,HCC,CRS,PKX"},
    {"t":"XHE","n":"Healthcare Equipment","fn":"SS SPDR S&P Hlth Care Eq","h":"TNDM,AXGN,INSP,GMED,HAE,ATEC,TMDX,ISRG,GKOS,SOLV,COO,IDXX,HOLX,OMCL,LNTH,MDT,LIVN,ALGN,UFPT,ICUI,EW,STE,PEN,NEOG,GEHC"},
    {"t":"XLV","n":"Healthcare","fn":"Sel Sector:HC SS SPDR","h":"LLY,JNJ,ABBV,UNH,MRK,ABT,TMO,ISRG,AMGN,GILD,BSX,PFE,DHR,MDT,SYK,VRTX,MCK,CVS,BMY,HCA,REGN,ELV,CI,COR,IDXX"},
    {"t":"COPX","n":"Copper Miners","fn":"Glbl X Copper Miners ETF","h":"ERO,TGB,FCX,HBM,IE,TECK,SCCO"},
    {"t":"DRNZ","n":"Drones","fn":"REX Drone ETF","h":"ONDS,AVAV,ELS,DRSHF,EH,RCAT,UMAC,PLTR,DPRO,KTOS"},
    {"t":"URA","n":"Uranium / Nuclear","fn":"Glbl X Uranium ETF","h":"CCJ,OKLO,UEC,URNM,NXE,LEU,LTBR,UUUU,SMR,SBSW,DNN,URNJ,SRUUF,NNE,EU,ASPI,UROY,BWXT,MIR,CEG"},
    {"t":"UFO","n":"Space","fn":"Procure Space","h":"ASTS,GSAT,PL,VSAT,TRMB,SATS,SIRI,RKLB,GRMN,IRDM,RTX,GE,LHX,VOYG,NOC,LMT,HON,LUNR,BA,FLY,CMCSA"},
    {"t":"PAVE","n":"Infrastructure Dev","fn":"Glbl X US Infra Dev","h":"HWM,PWR,PH,CRH,SRE,NSC,FAST,TT,CSX,ROK,URI,EMR,DE,ETN,UNP,VMC,MLM,NUE,EME,STLD,HUBB,TRMB,FTV,WWD,PNR"},
    {"t":"LIT","n":"Lithium & Battery","fn":"Glbl X Lith & Bat Tech","h":"RIO,ALB,LAR,SEI,MVST,ENS,EOSE,SQM,AMPX,TSLA,BLDP,SLDP,ABAT,SGML,SLI,LAC"},
    {"t":"XLB","n":"Basic Materials","fn":"Sel Sector:Mat SS SPDR","h":"LIN,NEM,SHW,ECL,NUE,FCX,MLM,VMC,APD,CTVA,STLD,PPG,IP,AMCR,PKG,IFF,DOW,DD,ALB,AVY,BALL,CF,LYB,MOS"},
    {"t":"XPH","n":"Pharmaceuticals","fn":"SS SPDR S&P Pharm","h":"LQDA,MBX,OGN,AXSM,EWTX,ELAN,VTRS,AMLX,AMRX,MRK,CRNX,XERS,LGND,JNJ,LLY,PBH,BMY,PRGO,SUPN,RPRX,ZTS,JAZZ,AVDL,PFE"},
    {"t":"PBJ","n":"Food & Beverage","fn":"Invesco Food & Beverage","h":"MNST,HSY,KR,SYY,MDLZ,KHC,CTVA,DASH,HLF,UNFI,SEB,IMKTA,TSN,USFD,FDP,CART,CHEF,ADM,AGRO,TR,ACI,DPZ,POST,JBS,WMK"},
    {"t":"RSPS","n":"Consumer Staples (EW)","fn":"Invesco S&P500 EW CS","h":"DLTR,KR,KMB,MNST,K,CHD,CAG,KO,TGT,PG,CLX,CL,KHC,SJM,GIS,PEP,CPB,HSY,KDP,KVUE,WMT,HRL,MO,SYY"},
    {"t":"XLP","n":"Consumer Staples","fn":"Sel Sector:C SSS SPDR I","h":"WMT,COST,PG,KO,PM,PEP,CL,MDLZ,MO,MNST,TGT,KR,KDP,SYY,KMB,KVUE,ADM,HSY,GIS,DG,EL,K,KHC,DLTR,CHD"},
    {"t":"QQQE","n":"Nasdaq-100 (EW)","fn":"Direxion:NASDAQ-100 EWI","h":"MU,AMD,REGN,AMAT,ISRG,BIIB,INTC,WBD,AZN,LRCX,ROST,AMGN,MRVL,MNST,EA,AVGO,IDXX,CTSH,AEP,DDOG,MAR,AAPL,VRTX,XEL,GILD"},
    {"t":"ROBO","n":"Robotics & Automation","fn":"Robo Glbl Robots & Auto","h":"SYM,TER,ISRG,SERV,IRBT,COHR,ROK,ILMN,RR,ARBE,AUR,GMED,ROK,PRCT,NOVT,PDYN,IPGP,NDSN,EMR"},
    {"t":"ARKK","n":"Innovation / Growth","fn":"ARK Innovation","h":"TSLA,ROKU,COIN,TEM,CRSP,SHOP,HOOD,RBLX,AMD,PLTR,BEAM,TER,CRCL,BMNR,ACHR,TXG,TWST,ILMN,AMZN,VCYT,BLSH,NVDA,NTRA,META,DKNG"},
    {"t":"IWM","n":"Russell 2000","fn":"iShares:Russ 2000 ETF","h":"CRDO,BE,FN,IONQ,NXT,GH,TSLA,KTOS,BBIO,CDE,MDGL,ENSG,HL,RMBS,SPXC,SATS,DY,STRL,OKLO,GTLS,IDCC,HQY,MOD,UMBF,AVAV"},
    {"t":"HYDR","n":"Hydrogen","fn":"Glbl X Hydrogen ETF","h":"BE,PLUG,BLDP,SLDP,FCEL,CMI,APD"},
    {"t":"PEJ","n":"Leisure & Ent","fn":"Invesco Leisure and Ent","h":"WBD,LVS,SYY,CCL,DASH,LYV,RCL,FLUT,LYFT,LION,EXPE,FOXA,CNK,TKO,CPA,USFD,PSKY,CART,BYD,EAT,BH,RRR,DPZ,MGM,ACEL"},
    {"t":"EEM","n":"Emerging Markets","fn":"iShares:MSCI Em Mkts","h":"PDD,BABA,NU,MELI,DLO,JMIA"},
    {"t":"IWO","n":"Small Cap Growth","fn":"iShares:Russ 2000 Gr","h":"CRDO,BE,FN,IONQ,GH,KTOS,BBIO,MDGL,ENSG,NXT,RMBS,SPXC,DY,STRL,GTLS,IDCC,HQY,MOD,AVAV,AEIS,WTS,RGTI,ZWS,HIMS,LUMN"},
    {"t":"MOO","n":"Agribusiness","fn":"VanEck:Agribusiness","h":"DE,ZTS,CTVA,ADM,TSN,CF,BG,ELAN,MOS,CNH,DAR,TTC,AGCO,CAT"},
    {"t":"XME","n":"Metals & Mining","fn":"SS SPDR S&P Metals&Mng","h":"HL,AA,HCC,STLD,NEM,LEU,CDE,NUE,CLF,CMC,RGLD,BTU,CNR,RS,FCX,UEC,MP,AMR,MTRN,CENX,IE,KALU,USAR,WS,MUX"},
    {"t":"ARKF","n":"Fintech Innovation","fn":"ARK BC & Fintech Innov","h":"SHOP,COIN,HOOD,PLTR,TOST,SOFI,XYZ,RBLX,ROKU,CRCL,MELI,AMD,DKNG,META,AMZN,BMNR,PINS,NU,KLAR,SE,BLSH,FUTU,Z"},
    {"t":"EWZ","n":"Brazil","fn":"iShares:MSCI Brazil","h":"NU,MELI,DLO"},
    {"t":"RSPM","n":"Materials (EW)","fn":"Invesco S&P500 EW Mt","h":"CE,IP,PPG,BALL,LYB,ECL,DOW,IFF,LIN,CTVA,AVY,PKG,MLM,CF,AMCR,VMC,APD,DD,EMN,SHW,FCX,NEM,MOS,FMC"},
    {"t":"RSPT","n":"Technology (EW)","fn":"Invesco S&P500 EW Tc","h":"AVGO,JBL,PLTR,TER,ANET,AAPL,VRSN,CSCO,INTC,SWKS,JNPR,GLW,ADI,TXN,KLAC,TDY,DELL,HPE,CDNS,ANSS,CDW,QCOM,NVDA,FFIV,MPWR"},
    {"t":"XBI","n":"Biotechnology","fn":"SS SPDR S&P Biotech","h":"EXAS,RVMD,RNA,INSM,NTRA,BBIO,REGN,MDGL,IONS,BIIB,AMGN,UTHR,INCY,ROIV,EXEL,VRTX,GILD,NBIX,ABBV,BMRN,CRSP,MRNA,PTCT,ALNY,KRYS"},
    {"t":"VERS","n":"Metaverse ETF","fn":"ProShares:Metaverse ETF","h":"GOOGL,AAPL,QCOM,EXPI,U,AMZN,NVDA,MU,MSFT,META,KOPN,RBLX,HIMX,SNAP,STGW,VUZI,EA,ARM,AMBA,ADSK,PTC,AMKR,TTWO,WSM,NOK"},
    {"t":"XSD","n":"Semiconductors (EW)","fn":"SS SPDR S&P Semiconductr","h":"MU,INTC,RGTI,AMD,MRVL,MTSI,FSLR,RMBS,SITM,SMTC,MPWR,ADI,QCOM,CRUS,ON,AVGO,CRDO,LSCC,NVDA,QRVO,SLAB,TXN,NXPI,SWKS,OLED,TSEM"},
    {"t":"ARKG","n":"Genomics","fn":"ARK Genomic Revolution","h":"TEM,CRSP,PSNL,GH,TWST,TXG,NTRA,BEAM,ILMN,VCYT,RXRX,ADPT,IONS,ABSI,CDNA,SDGR,NTLA,NRIX,PACB,WGS,BFLY,PRME,ARCT,AMGN"},
    {"t":"GBTC","n":"Bitcoin","fn":"GRAYSCALE BITCOIN TRUST","h":"IBIT,ETHA,MSTR,BMNR,SBET,COIN"},
    {"t":"IGV","n":"Software","fn":"iShares:Expand Tch-Sftwr","h":"PLTR,MSFT,CRM,INTU,NOW,ORCL,APP,ADBE,CRWD,PANW,CDNS,SNPS,ADSK,FTNT,DDOG,ROP,WDAY,EA,MSTR,TTWO,FICO,TEAM,ZS,ZM,PTC"},
    {"t":"XTN","n":"Transport & Logistics","fn":"SPDR S&P Trans","h":"JBHT,KEX,CHRW,FDX,EXPD,KNX,UPS,LYFT,LUV,XPO,AAL,CSX,MATX,UNP,NSC,DAL,HUBG,LSTR,JOBY,GXO,SNDR,ODFL,SAIA,UAL,WERN"},
    {"t":"PHO","n":"Water Infrastructure","fn":"Invesco Water Res","h":"WAT,FERG,ECL,ROP,AWK,MLI,IEX,WMS,XYL,PNR,VLTO,AOS,ACM,CNM,VMI,WTRG,BMI,TTEK,ITRI,WTS,ZWS,MWA,SBS,FELE,HWKN"},
    {"t":"LABU","n":"Biotech (3x)","fn":"Direxion:S&P Btech Bl 3X","h":"EXAS,RVMD,RNA,INSM,REGN,NTRA,MDGL,BBIO,BIIB,IONS,UTHR,AMGN,INCY,ROIV,EXEL"},
    {"t":"BLOK","n":"Blockchain","fn":"Amplify Blockchain Tech","h":"HOOD,CIFR,HUT,GLXY,CLSK,WULF,COIN,IBM,NU,BBBY,PYPL,CMPO,CORZ,FBTC,OPRA,RBLX,BLK,FIGR,XYZ,CME,IBIT,BITB,MELI,CAN,BKKT"},
    {"t":"BUZZ","n":"Social Sentiment","fn":"VanEck:Social Sentiment","h":"APLD,INTC,GME,NBIS,TSLA,META,SOFI,RGTI,IREN,AMZN,PLTR,NVDA,HOOD,ASTS,OPEN,AMD,MSTR,HIMS,GOOGL,AAPL,UNH,SOUN,SMCI,DKNG,PYPL"},
    {"t":"XSW","n":"Software & Services","fn":"SS SPDR S&P Sftwre & Svc","h":"CIFR,SEMR,PRO,WULF,HUT,CLSK,TDC,QBTS,JAMF,APPN,BBAI,EPAM,PATH,WK,EA,IBM,CRWD,IDCC,GDYN,FICO,CRNC,DDOG,SNPS,CTSH,AGYS"},
    {"t":"SMH","n":"Semiconductors","fn":"VanEck:Semiconductor","h":"NVDA,TSM,AVGO,MU,INTC,AMAT,AMD,ASML,LRCX,KLAC,ADI,QCOM,TXN,CDNS,SNPS,MRVL,NXPI,MPWR,TER,MCHP,STM,ON,SWKS,QRVO,OLED,TSEM"},
    {"t":"WCLD","n":"Cloud Computing","fn":"WisdomTree:Cloud Cmptng","h":"FSLY,SEMR,MDB,DOCN,FROG,PATH,SNOW,BILL,DDOG,CFLT,WK,TWLO,CRWD,PCOR,AGYS,IOT,BRZE,QLYS,CWAN,BL,CLBT,PANW,SHOP,INTA,NET"},
    {"t":"DRIV","n":"EV & Mobility","fn":"Glbl X Auto & Elct Vhcls","h":"GOOGL,BE,TSLA,INTC,NVDA,MSFT,RIO,QCOM,GM,ALB,NBIS,COHR,HON,SQM,ENS,F,BIDU,AMPX,SITM"},
    {"t":"WGMI","n":"Crypto Miners","fn":"CoinShares Btc Mining","h":"CIFR,IREN,BITF,WULF,RIOT,HUT,CORZ,APLD,CLSK,HIVE,BTDR,MARA,NVDA,CANG,GLXY,XYZ,BTBT,TSM,CAN"},
    {"t":"IBUY","n":"Online Retail","fn":"Amplify Online Retail","h":"FIGS,LQDT,CVNA,UPWK,EXPE,CART,W,RVLV,CHWY,LYFT,MSM,EBAY,BKNG,AFRM,SPOT,TRIP,ABNB,PTON,UBER,AMZN,CPRT,PYPL,HIMS,SSTK,ETSY"},
    {"t":"XHB","n":"Homebuilders","fn":"SS SPDR S&P Homebuilders","h":"SKY,SGI,WMS,CVCO,BLD,JCI,IBP,TT,KBH,TOL,ALLE,LEN,PHM,MTH,LOW,TMHC,NVR,WSM,DHI,MAS,LII,CARR,HD,CSL,BLDR"},
    {"t":"TAN","n":"Solar Energy","fn":"Invesco Solar","h":"NXT,FSLR,RUN,ENPH,SEDG,HASI,CSIQ,CWEN,DQ,SHLS,ARRY,JKS"},
    {"t":"KCE","n":"Capital Markets","fn":"SS SPDR S&P Cap Mkts","h":"AMG,IVZ,MS,CBOE,BK,SF,CME,STT,HOOD,LPLA,GS,NTRS,IBKR,STEP,SCHW,MSCI,PIPR,JHG,TPG,MCO,EVR,TROW,GLXY,FHI,NDAQ"},
    {"t":"IPAY","n":"Digital Payments","fn":"Amplify Digital Payments","h":"AXP,V,MA,PYPL,CPAY,FIS,AFRM,XYZ,GPN,COIN,TOST,FISV,FOUR,WEX,QTWO,STNE,ACIW,EEFT,WU,RELY"},
    {"t":"ITB","n":"Home Construction","fn":"iShares:US Home Cons ETF","h":"DHI,LEN,PHM,NVR,TOL,SHW,LOW,BLD,HD,LII,MAS,BLDR,TMHC,IBP,MTH,OC,SKY,CVCO,KBH,EXP,SSD,FND,MHO,FBIN,MHK"},
    {"t":"RSPF","n":"Financials (EW)","fn":"Invesco S&P500 EW Fn","h":"GL,ERIE,MET,WTW,FI,V,AJG,CB,ALL,L,CME,MMC,MA,AXP,AON,MS,EG,WFC,STT,FDS,AFL,PRU,AIZ,AIG,JPM,FICO"},
    {"t":"WOOD","n":"Timber & Forestry","fn":"iShares:Gl Timber","h":"PCH,RYN,SLVM,WY,SW,IP,CLW"},
    {"t":"COAL","n":"Coal","fn":"Range Glbl Coal Index","h":"HCC,AMR,BTU,NRP,ARLP,METC,BHP,AREC,NC"},
    {"t":"ICLN","n":"Clean Energy","fn":"iShares:Gl Cl Energy","h":"BE,FSLR,NXT,ORA,ENPH,RUN,CWEN,PLUG"},
    {"t":"KRE","n":"Regional Banks","fn":"SS SPDR S&P Reg Banking","h":"CADE,VLY,COLB,CFG,TFC,PB,FNB,EWBC,FHN,FLG,WBS,ONB,MTB,PNFP,RF,HBAN,ZION,BPOP,WAL,UMBF,WTFC,SSB,CFR,HWC"},
    {"t":"CIBR","n":"Cybersecurity","fn":"FT II:Nsdq Cybersecurity","h":"AVGO,CRWD,CSCO,INFY,PANW,LDOS,FTNT,CYBR,CHKP,NET,ZS,GEN,AKAM,FFIV,OKTA,BAH,RBRK,S,CVLT,QLYS,SAIC,VRNS"},
    {"t":"PBW","n":"Clean Energy","fn":"Invesco WldHill CE","h":"CSIQ,TE,FLNC,SGML,LAC,LAR,ALB,SQM,EOSE,NXT,FSLR,ORA,SLI,BE,NVTS,BEPC,AEIS,MYRG,JKS,PWR,SLDP,RUN,SHLS,RIVN,AMRC"},
    {"t":"XRT","n":"Retail","fn":"SS SPDR S&P Retail","h":"VSCO,REAL,KSS,M,ODP,EYE,DDS,ROST,GAP,DLTR,WMT,URBN,SBH,FIVE,PSMT,AEO,TJX,RVLV,ULTA,BOOT,ANF,CASY,SIG,CVNA,DG"},
    {"t":"RSPD","n":"Consumer Disc (EW)","fn":"Invesco S&P500 EW CD","h":"DRI,TPR,GM,ULTA,APTV,TSLA,BBY,DECK,RL,EBAY,ROST,MCD,EXPE,TJX,YUM,HLT,MAR,AMZN,NKE,LULU,AZO,F,KMX,ABNB,LKQ"},
    {"t":"XLF","n":"Financials","fn":"Sel Sector:Fin SS SPDR I","h":"BRK.B,JPM,V,MA,BAC,WFC,GS,MS,AXP,C,SCHW,SPGI,BLK,COF,PGR,CB,BX,CME,HOOD,MMC,ICE,KKR,BK,USB,PNC"},
    {"t":"ESPO","n":"Esports & Gaming","fn":"VanEck:VG and eSports","h":"EA,NTES,TTWO,RBLX,SE,U,LOGI,GME,PLTK,CRSR,SONY"},
    {"t":"JETS","n":"Airlines & Travel","fn":"US Global Jets","h":"LUV,AAL,DAL,UAL,ALGT,SNCY,SKYW,ULCC,JBLU,EXPE,GD,ALK,TXT,SABR,BKNG,TRIP,BA,RYAAY"},
    {"t":"KBE","n":"Banks","fn":"SS SPDR S&P Bank ETF","h":"CMA,BKU,BANC,EBC,PFSI,CADE,BK,VLY,COLB,WFC,BAC,INDB,C,FBK,CFG,TCBI,BOKF,SBCF,TFC,PB,NTRS,ABCB,FIBK,JPM,WSBC"},
    {"t":"FXI","n":"China Large-Cap","fn":"iShares:China Large Cp","h":"BABA,TCEHY,NTES,BYDDF,TCOM,JD,BIDU,PTR,SNP,FUTU,KWEB"},
    {"t":"XLY","n":"Consumer Disc","fn":"Sel Sctr:C D SS SPDR In","h":"AMZN,TSLA,HD,MCD,TJX,BKNG,LOW,SBUX,ORLY,NKE,DASH,GM,MAR,RCL,HLT,AZO,ROST,F,ABNB,CMG,DHI,YUM,EBAY,GRMN,EXPE"},
    {"t":"MSOS","n":"Cannabis","fn":"AdvsrShs Pure USCannabis","h":"VFF,MNMD,TLRY,MO,CRON,GTBIF,TCNNF"},
]

TICKERS = [e["t"] for e in ETF_INFO]


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


def percentrank_inc(values, x):
    n = len(values)
    if n <= 1:
        return None
    return sum(1 for v in values if v < x) / (n - 1)


def compute_ema(closes, period):
    """Compute EMA for the given period. Returns final EMA value."""
    if len(closes) < period:
        return None
    mult = 2 / (period + 1)
    ema = np.mean(closes[:period])
    for c in closes[period:]:
        ema = (c - ema) * mult + ema
    return ema


def compute_sma(closes, period):
    """Compute SMA for the given period. Returns final SMA value."""
    if len(closes) < period:
        return None
    return np.mean(closes[-period:])


def grade_holding(closes):
    """
    Grade a holding based on moving average structure.
    Gold:   price > EMA21 AND EMA9 > EMA21 > SMA50 > SMA200
    Silver: (i) price > EMA21 AND EMA9 > EMA21, but SMA50 < SMA200
            OR (ii) SMA50 > SMA200, but EMA9 < EMA21 and/or EMA21 < SMA50
    Bronze: everything else
    """
    if closes is None or len(closes) < 200:
        return "b"

    price = closes[-1]
    sma200 = compute_sma(closes, 200)
    sma50 = compute_sma(closes, 50)
    ema21 = compute_ema(closes, 21)
    ema9 = compute_ema(closes, 9)

    if sma200 is None or sma50 is None or ema21 is None or ema9 is None:
        return "b"

    # Gold: price > EMA21 AND perfectly stacked EMA9 > EMA21 > SMA50 > SMA200
    if price > ema21 and ema9 > ema21 > sma50 > sma200:
        return "g"

    # Silver case (i): price > EMA21 AND EMA9 > EMA21, but SMA50 < SMA200
    if price > ema21 and ema9 > ema21 and sma50 < sma200:
        return "s"

    # Silver case (ii): SMA50 > SMA200, but EMA9 < EMA21 and/or EMA21 < SMA50
    if sma50 > sma200 and (ema9 < ema21 or ema21 < sma50):
        return "s"

    return "b"


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Fetching data for {len(TICKERS) + 1} ETF tickers...")
    all_tickers = ["SPY"] + TICKERS

    end = datetime.now()
    start = end - timedelta(days=90)

    raw = yf.download(
        all_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        threads=True,
    )

    if raw.empty:
        print("ERROR: No data returned from Yahoo Finance")
        sys.exit(1)

    def get_df(ticker):
        try:
            df = raw[ticker].dropna(subset=["Close"])
            return df
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
    spy_price = float(spy_closes[-1])
    spy_chg = (spy_closes[-1] / spy_closes[-2] - 1) * 100 if len(spy_closes) >= 2 else 0
    spy_ts_map = {ts: i for i, ts in enumerate(spy_df.index)}

    # ─── Process each ETF ───────────────────────────────────
    results = []
    for info in ETF_INFO:
        ticker = info["t"]
        df = get_df(ticker)

        if df is None or len(df) < 10:
            results.append({**info, "rv": None, "am": None, "ch": None, "c5": None,
                            "c20": None, "rs": None, "rf": 0, "ra": 0, "p": None,
                            "fr": None, "vs": None})
            continue

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
        atr_mult = abs(c[-1] - sma50) / atr if atr > 0 else 0

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
            results.append({
                **info, "rv": round(rvol * 100) if rvol else None,
                "am": round(atr_mult * 100), "ch": round(change, 2),
                "c5": round(c5, 2) if c5 is not None else None,
                "c20": round(c20, 2) if c20 is not None else None,
                "rs": None, "rf": 0, "ra": 0, "p": round(price, 2),
                "fr": None, "vs": None,
            })
            continue

        recent = common[-LOOKBACK:]
        atr_ratio = atr / spy_atr if spy_atr > 0 else 1
        rs_series = []
        for etf_i, spy_i in recent:
            ratio = c[etf_i] / spy_closes[spy_i]
            rs_val = ratio / atr_ratio if atr_ratio > 0 else ratio
            rs_series.append(float(rs_val))

        # VARS sparkline: include one extra day for context (26 values)
        extended = common[-(LOOKBACK + 1):]
        vs_series = []
        for etf_i, spy_i in extended:
            ratio = c[etf_i] / spy_closes[spy_i]
            rs_val = ratio / atr_ratio if atr_ratio > 0 else ratio
            vs_series.append(round(float(rs_val), 4))

        final_rs = rs_series[-1]
        rs_pctrank = percentrank_inc(rs_series, final_rs)

        # Consecutive streaks from RS series
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

        results.append({
            **info,
            "rv": round(rvol * 100) if rvol else None,
            "am": round(atr_mult * 100),
            "ch": round(change, 2),
            "c5": round(c5, 2) if c5 is not None else None,
            "c20": round(c20, 2) if c20 is not None else None,
            "rs": round(rs_pctrank * 100) if rs_pctrank is not None else None,
            "rf": dec_streak,
            "ra": adv_streak,
            "p": round(price, 2),
            "fr": round(final_rs, 4),
            "vs": vs_series,
        })

    # Sort: RS desc, then Change desc
    results.sort(key=lambda x: (x["rs"] if x["rs"] is not None else -1,
                                 x["ch"] if x["ch"] is not None else -999), reverse=True)
    for i, r in enumerate(results):
        r["rk"] = i + 1

    # ─── Grade individual holdings ──────────────────────────
    print("Grading individual holdings...")
    all_holdings = set()
    for e in results:
        if e.get("h"):
            for hh in e["h"].split(","):
                hh = hh.strip()
                if hh:
                    all_holdings.add(hh)

    print(f"  Fetching {len(all_holdings)} unique holdings for MA grading...")
    holding_tickers = list(all_holdings)
    holding_grades = {}

    CHUNK = 200
    for i in range(0, len(holding_tickers), CHUNK):
        chunk = holding_tickers[i:i + CHUNK]
        print(f"  Batch {i // CHUNK + 1}: fetching {len(chunk)} tickers...")
        try:
            h_start = end - timedelta(days=400)  # ~276 trading days for reliable SMA200
            h_raw = yf.download(
                chunk,
                start=h_start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                group_by="ticker",
                auto_adjust=True,
                threads=True,
            )
            for tk in chunk:
                try:
                    if len(chunk) == 1:
                        hdf = h_raw
                    else:
                        hdf = h_raw[tk]
                    hdf = hdf.dropna(subset=["Close"])
                    closes = hdf["Close"].values.tolist()
                    grade = grade_holding(closes)
                    holding_grades[tk] = grade
                    # Debug: log MA values for verification
                    if tk in ("T", "CMCSA", "BAC", "AAPL", "XOM"):
                        _p = closes[-1] if closes else 0
                        _s200 = compute_sma(closes, 200) if len(closes) >= 200 else None
                        _s50 = compute_sma(closes, 50) if len(closes) >= 50 else None
                        _e21 = compute_ema(closes, 21) if len(closes) >= 21 else None
                        _e9 = compute_ema(closes, 9) if len(closes) >= 9 else None
                        print(f"    DEBUG {tk}: price={_p:.2f} EMA9={_e9:.2f if _e9 else 'N/A'} "
                              f"EMA21={_e21:.2f if _e21 else 'N/A'} SMA50={_s50:.2f if _s50 else 'N/A'} "
                              f"SMA200={_s200:.2f if _s200 else 'N/A'} → {grade} "
                              f"(data_points={len(closes)})")
                except Exception:
                    holding_grades[tk] = "b"
        except Exception as ex:
            print(f"  Warning: batch failed: {ex}")
            for tk in chunk:
                holding_grades[tk] = "b"

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

    # Panels: 4+ consecutive days of RS increase/decrease
    adv_streak_list = [[r["t"], r["n"]] for r in results if r.get("ra", 0) >= 4]
    dec_streak_list = [[r["t"], r["n"]] for r in results if r.get("rf", 0) >= 4]

    data = {
        "e": results,
        "s": {
            "ap": adv_pct,
            "dp": dec_pct,
            "a": adv_streak_list[:30],
            "d": dec_streak_list[:30],
        },
        "meta": {
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "spy_price": round(spy_price, 2),
            "spy_change": round(float(spy_chg), 2),
        },
    }

    with open("data.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))

    g_count = sum(1 for v in holding_grades.values() if v == "g")
    s_count = sum(1 for v in holding_grades.values() if v == "s")
    b_count = sum(1 for v in holding_grades.values() if v == "b")

    print(f"\nWritten data.json — {len(results)} ETFs")
    print(f"Top 5: {[r['t'] for r in results[:5]]}")
    print(f"Advancing: {adv_pct}% | Declining: {dec_pct}%")
    print(f"4+ day advance streaks: {len(adv_streak_list)}")
    print(f"4+ day decline streaks: {len(dec_streak_list)}")
    print(f"Holdings graded: {g_count} gold, {s_count} silver, {b_count} bronze")


if __name__ == "__main__":
    main()

