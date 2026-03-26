import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import hashlib
from datetime import datetime

st.set_page_config(
    page_title="Ace-Trade · Professional Terminal",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# ── Users (username, password, email, role) ───────────────────────────────────
USERS = {
    "admin":   {"password":"acetrade123",  "email":"admin@acetrade.in",   "name":"Admin",   "role":"Owner"},
    "trader1": {"password":"trade@2024",   "email":"trader1@acetrade.in", "name":"Trader",  "role":"Analyst"},
    "ruchi":   {"password":"ruchi123",     "email":"ruchi@acetrade.in",   "name":"Ruchi",   "role":"Analyst"},
}
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()
USERS_H = {u: hash_pw(d["password"]) for u,d in USERS.items()}
EMAIL_MAP = {d["email"]: u for u,d in USERS.items()}

# ── Curated NSE symbols (100+) ────────────────────────────────────────────────
SYMBOL_DB = [
    # Large Cap NSE
    ("Reliance Industries","RELIANCE.NS","NSE"),("Tata Consultancy Services","TCS.NS","NSE"),
    ("Infosys","INFY.NS","NSE"),("HDFC Bank","HDFCBANK.NS","NSE"),("ICICI Bank","ICICIBANK.NS","NSE"),
    ("Wipro","WIPRO.NS","NSE"),("State Bank of India","SBIN.NS","NSE"),("Axis Bank","AXISBANK.NS","NSE"),
    ("Bajaj Finance","BAJFINANCE.NS","NSE"),("Kotak Mahindra Bank","KOTAKBANK.NS","NSE"),
    ("Larsen & Toubro","LT.NS","NSE"),("Maruti Suzuki","MARUTI.NS","NSE"),
    ("Tata Motors","TATAMOTORS.NS","NSE"),("Sun Pharmaceutical","SUNPHARMA.NS","NSE"),
    ("HCL Technologies","HCLTECH.NS","NSE"),("Tech Mahindra","TECHM.NS","NSE"),
    ("Adani Enterprises","ADANIENT.NS","NSE"),("Adani Ports","ADANIPORTS.NS","NSE"),
    ("Adani Green Energy","ADANIGREEN.NS","NSE"),("Adani Power","ADANIPOWER.NS","NSE"),
    ("Max Healthcare","MAXHEALTH.NS","NSE"),("Hindustan Unilever","HINDUNILVR.NS","NSE"),
    ("Asian Paints","ASIANPAINT.NS","NSE"),("Nestle India","NESTLEIND.NS","NSE"),
    ("Titan Company","TITAN.NS","NSE"),("UltraTech Cement","ULTRACEMCO.NS","NSE"),
    ("NTPC","NTPC.NS","NSE"),("Power Grid Corp","POWERGRID.NS","NSE"),("ONGC","ONGC.NS","NSE"),
    ("Tata Steel","TATASTEEL.NS","NSE"),("JSW Steel","JSWSTEEL.NS","NSE"),
    ("Hindalco Industries","HINDALCO.NS","NSE"),("Coal India","COALINDIA.NS","NSE"),
    ("Dr Reddy's Labs","DRREDDY.NS","NSE"),("Cipla","CIPLA.NS","NSE"),
    ("Divi's Laboratories","DIVISLAB.NS","NSE"),("Bajaj Auto","BAJAJ-AUTO.NS","NSE"),
    ("Hero MotoCorp","HEROMOTOCO.NS","NSE"),("IndusInd Bank","INDUSINDBK.NS","NSE"),
    ("Zomato","ZOMATO.NS","NSE"),("Paytm","PAYTM.NS","NSE"),("Nykaa","NYKAA.NS","NSE"),
    ("DMart","DMART.NS","NSE"),("IRCTC","IRCTC.NS","NSE"),("Yes Bank","YESBANK.NS","NSE"),
    ("Punjab National Bank","PNB.NS","NSE"),("Bank of Baroda","BANKBARODA.NS","NSE"),
    ("Vedanta","VEDL.NS","NSE"),("SBI Life Insurance","SBILIFE.NS","NSE"),
    ("HDFC Life Insurance","HDFCLIFE.NS","NSE"),("Pidilite Industries","PIDILITIND.NS","NSE"),
    ("Havells India","HAVELLS.NS","NSE"),("Dabur India","DABUR.NS","NSE"),
    ("Marico","MARICO.NS","NSE"),("Colgate-Palmolive India","COLPAL.NS","NSE"),
    ("Trent","TRENT.NS","NSE"),("MRF","MRF.NS","NSE"),("Eicher Motors","EICHERMOT.NS","NSE"),
    ("Bosch India","BOSCHLTD.NS","NSE"),("Siemens India","SIEMENS.NS","NSE"),
    ("BHEL","BHEL.NS","NSE"),("GAIL India","GAIL.NS","NSE"),("BPCL","BPCL.NS","NSE"),
    ("Indian Oil Corp","IOC.NS","NSE"),("HPCL","HINDPETRO.NS","NSE"),
    ("IndiGo Airlines","INDIGO.NS","NSE"),("SpiceJet","SPICEJET.NS","NSE"),
    ("Lupin","LUPIN.NS","NSE"),("Torrent Pharma","TORNTPHARM.NS","NSE"),
    ("Muthoot Finance","MUTHOOTFIN.NS","NSE"),("Tata Power","TATAPOWER.NS","NSE"),
    ("SAIL","SAIL.NS","NSE"),("Info Edge (Naukri)","NAUKRI.NS","NSE"),
    ("Jubilant FoodWorks","JUBLFOOD.NS","NSE"),("ABB India","ABB.NS","NSE"),
    ("Godrej Consumer","GODREJCP.NS","NSE"),("Page Industries","PAGEIND.NS","NSE"),
    ("Berger Paints","BERGEPAINT.NS","NSE"),("Voltas","VOLTAS.NS","NSE"),
    ("Emami","EMAMILTD.NS","NSE"),("Zydus Lifesciences","ZYDUSLIFE.NS","NSE"),
    ("Alkem Laboratories","ALKEM.NS","NSE"),("Shriram Finance","SHRIRAMFIN.NS","NSE"),
    ("Cholamandalam Finance","CHOLAFIN.NS","NSE"),("NMDC","NMDC.NS","NSE"),
    ("Jindal Steel","JINDALSTEL.NS","NSE"),("3M India","3MINDIA.NS","NSE"),
    # Mid & Small Cap
    ("Godawari Power","GPIL.NS","NSE"),("Godawari Power & Ispat","GPIL.NS","NSE"),
    ("GPIL","GPIL.NS","NSE"),
    ("Suzlon Energy","SUZLON.NS","NSE"),("IRFC","IRFC.NS","NSE"),
    ("REC Limited","RECLTD.NS","NSE"),("PFC","PFC.NS","NSE"),
    ("Bharat Dynamics","BDL.NS","NSE"),("HAL","HAL.NS","NSE"),
    ("Mazagon Dock","MAZDOCK.NS","NSE"),("Garden Reach Shipbuilders","GRSE.NS","NSE"),
    ("Cochin Shipyard","COCHINSHIP.NS","NSE"),("RVNL","RVNL.NS","NSE"),
    ("NBCC","NBCC.NS","NSE"),("HUDCO","HUDCO.NS","NSE"),
    ("Kalyan Jewellers","KALYANKJIL.NS","NSE"),("Senco Gold","SENCO.NS","NSE"),
    ("PC Jeweller","PCJEWELLER.NS","NSE"),("Titan Company","TITAN.NS","NSE"),
    ("Waaree Energies","WAAREEENER.NS","NSE"),("Premier Energies","PREMIENERG.NS","NSE"),
    ("Kaynes Technology","KAYNES.NS","NSE"),("Dixon Technologies","DIXON.NS","NSE"),
    ("Amber Enterprises","AMBER.NS","NSE"),("Syrma SGS Technology","SYRMA.NS","NSE"),
    ("Avalon Technologies","AVALON.NS","NSE"),("Tata Elxsi","TATAELXSI.NS","NSE"),
    ("Persistent Systems","PERSISTENT.NS","NSE"),("Mphasis","MPHASIS.NS","NSE"),
    ("Coforge","COFORGE.NS","NSE"),("LTIMindtree","LTIM.NS","NSE"),
    ("Polycab India","POLYCAB.NS","NSE"),("KEI Industries","KEI.NS","NSE"),
    ("RR Kabel","RRKABEL.NS","NSE"),("Finolex Cables","FNXCABLE.NS","NSE"),
    ("Astral Pipes","ASTRAL.NS","NSE"),("Supreme Industries","SUPREMEIND.NS","NSE"),
    ("Deepak Nitrite","DEEPAKNTR.NS","NSE"),("SRF","SRF.NS","NSE"),
    ("Alkyl Amines","ALKYLAMINE.NS","NSE"),("Fine Organic","FINEORG.NS","NSE"),
    ("Galaxy Surfactants","GALAXYSURF.NS","NSE"),("Tata Chemicals","TATACHEM.NS","NSE"),
    ("UPL","UPL.NS","NSE"),("PI Industries","PIIND.NS","NSE"),
    ("Balrampur Chini","BALRAMCHIN.NS","NSE"),("EID Parry","EIDPARRY.NS","NSE"),
    ("Triveni Engineering","TRIVENI.NS","NSE"),
    ("InterGlobe Aviation","INDIGO.NS","NSE"),
    ("Blue Dart Express","BLUEDART.NS","NSE"),("Delhivery","DELHIVERY.NS","NSE"),
    ("VRL Logistics","VRLLOG.NS","NSE"),
    ("Laurus Labs","LAURUSLABS.NS","NSE"),("Granules India","GRANULES.NS","NSE"),
    ("Aarti Drugs","AARTIDRUGS.NS","NSE"),("Solara Active Pharma","SOLARA.NS","NSE"),
    ("Narayana Hrudayalaya","NH.NS","NSE"),("Apollo Hospitals","APOLLOHOSP.NS","NSE"),
    ("Fortis Healthcare","FORTIS.NS","NSE"),("Global Health (Medanta)","MEDANTA.NS","NSE"),
    ("Aster DM Healthcare","ASTERDM.NS","NSE"),("Rainbow Children's","RAINBOW.NS","NSE"),
    ("Syngene International","SYNGENE.NS","NSE"),("Divi's Labs","DIVISLAB.NS","NSE"),
    ("Ajanta Pharma","AJANTPHARM.NS","NSE"),("Ipca Laboratories","IPCALAB.NS","NSE"),
    ("Natco Pharma","NATCOPHARM.NS","NSE"),
    ("Varun Beverages","VBL.NS","NSE"),("Radico Khaitan","RADICO.NS","NSE"),
    ("United Spirits","MCDOWELL-N.NS","NSE"),("United Breweries","UBL.NS","NSE"),
    ("Westlife Foodworld","WESTLIFE.NS","NSE"),("Devyani International","DEVYANI.NS","NSE"),
    ("Sapphire Foods","SAPPHIRE.NS","NSE"),
    ("AU Small Finance Bank","AUBANK.NS","NSE"),("Ujjivan Small Finance","UJJIVANSFB.NS","NSE"),
    ("Equitas SFB","EQUITASBNK.NS","NSE"),("Jana Small Finance","JANASFB.NS","NSE"),
    ("IDFC First Bank","IDFCFIRSTB.NS","NSE"),("Bandhan Bank","BANDHANBNK.NS","NSE"),
    ("City Union Bank","CUB.NS","NSE"),("Karnataka Bank","KTKBANK.NS","NSE"),
    ("Federal Bank","FEDERALBNK.NS","NSE"),("South Indian Bank","SOUTHBANK.NS","NSE"),
    ("RBL Bank","RBLBANK.NS","NSE"),("DCB Bank","DCBBANK.NS","NSE"),
    ("Birla Corporation","BIRLACORPN.NS","NSE"),("ACC","ACC.NS","NSE"),
    ("Ambuja Cements","AMBUJACEM.NS","NSE"),("Shree Cement","SHREECEM.NS","NSE"),
    ("JK Cement","JKCEMENT.NS","NSE"),("Heidelberg Cement","HEIDELBERG.NS","NSE"),
    ("Prism Cement","PRISMCEM.NS","NSE"),("Star Cement","STARCEMENT.NS","NSE"),
    ("Greenpanel Industries","GREENPANEL.NS","NSE"),("Century Plyboards","CENTURYPLY.NS","NSE"),
    ("Greenply Industries","GREENPLY.NS","NSE"),("Sheela Foam","SFL.NS","NSE"),
    ("Relaxo Footwear","RELAXO.NS","NSE"),("Bata India","BATAINDIA.NS","NSE"),
    ("Metro Brands","METROBRAND.NS","NSE"),("Campus Activewear","CAMPUS.NS","NSE"),
    ("V-Mart Retail","VMART.NS","NSE"),("Aditya Birla Fashion","ABFRL.NS","NSE"),
    ("Vedant Fashions (Manyavar)","MANYAVAR.NS","NSE"),
    ("Raymond","RAYMOND.NS","NSE"),("KPR Mill","KPRMILL.NS","NSE"),
    ("Vardhman Textiles","VTL.NS","NSE"),("Welspun India","WELSPUNIND.NS","NSE"),
    ("Trident","TRIDENT.NS","NSE"),
    ("Cera Sanitaryware","CERA.NS","NSE"),("Kajaria Ceramics","KAJARIACER.NS","NSE"),
    ("Somany Ceramics","SOMANYCERA.NS","NSE"),
    ("Finolex Industries","FINPIPE.NS","NSE"),
    ("Maharashtra Seamless","MAHSEAMLES.NS","NSE"),
    ("Goa Carbon","GOACARBON.NS","NSE"),
    ("MOIL","MOIL.NS","NSE"),("Hindustan Zinc","HINDZINC.NS","NSE"),
    ("National Aluminium","NATIONALUM.NS","NSE"),
    ("Ratnamani Metals","RATNAMANI.NS","NSE"),
    ("Lloyds Metals","LLOYDMETAL.NS","NSE"),
    ("Anupam Rasayan","ANUPAMRAS.NS","NSE"),
    ("Navin Fluorine","NAVINFLUOR.NS","NSE"),
    ("Gujarat Fluorochemicals","GUJFLUORO.NS","NSE"),
    ("Aether Industries","AETHER.NS","NSE"),
    # Indices
    ("Nifty 50","^NSEI","INDEX"),("Sensex (BSE 30)","^BSESN","INDEX"),
    ("Nifty Bank","^NSEBANK","INDEX"),("S&P 500","^GSPC","INDEX"),
    ("Dow Jones","^DJI","INDEX"),("NASDAQ","^IXIC","INDEX"),
    # Commodities
    ("Gold","GC=F","COMMODITY"),("Silver","SI=F","COMMODITY"),
    ("Crude Oil (WTI)","CL=F","COMMODITY"),("Natural Gas","NG=F","COMMODITY"),
    ("Copper","HG=F","COMMODITY"),("Platinum","PL=F","COMMODITY"),
    ("Brent Crude","BZ=F","COMMODITY"),("Aluminium","ALI=F","COMMODITY"),
    # US Stocks
    ("Apple","AAPL","US"),("Microsoft","MSFT","US"),("Alphabet (Google)","GOOGL","US"),
    ("Amazon","AMZN","US"),("Tesla","TSLA","US"),("Meta Platforms","META","US"),
    ("NVIDIA","NVDA","US"),("Netflix","NFLX","US"),("AMD","AMD","US"),
]

# Deduplicate
seen = set()
SYMBOL_DB_CLEAN = []
for item in SYMBOL_DB:
    if item[1] not in seen:
        seen.add(item[1]); SYMBOL_DB_CLEAN.append(item)
SYMBOL_DB = SYMBOL_DB_CLEAN
ALL_LABELS = [f"{n} ({t})" for n,t,_ in SYMBOL_DB]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],[data-testid="stMain"]{
  background:#000 !important;font-family:'Inter',-apple-system,sans-serif !important;
  color:#FFF;-webkit-font-smoothing:antialiased}
[data-testid="stSidebar"]{background:#0A0A0A !important;border-right:1px solid #1C1C1C !important}
[data-testid="stSidebarNav"]{display:none !important}
.block-container{padding:1.2rem 1.8rem 4rem !important;max-width:1440px}

div[data-testid="stButton"]>button{background:#FFF !important;color:#000 !important;border:none !important;
  border-radius:10px !important;font-family:'Inter',sans-serif !important;font-weight:700 !important;
  font-size:0.88rem !important;transition:all .15s !important}
div[data-testid="stButton"]>button:hover{background:#E0E0E0 !important;transform:translateY(-1px) !important}

input[type="text"],input[type="password"]{background:#111 !important;border:1px solid #2A2A2A !important;
  border-radius:10px !important;color:#FFF !important;font-family:'Inter',sans-serif !important;font-size:0.95rem !important}
input:focus{border-color:#FFF !important;box-shadow:0 0 0 2px rgba(255,255,255,.08) !important}
input::placeholder{color:#3A3A3A !important}
.stTextInput label,.stSelectbox label,.stNumberInput label,.stSlider label{
  color:#888 !important;font-size:0.75rem !important;font-weight:600 !important;
  letter-spacing:0.6px !important;text-transform:uppercase !important}
.stSelectbox>div>div{background:#111 !important;border:1px solid #2A2A2A !important;
  border-radius:10px !important;color:#FFF !important;font-size:0.92rem !important}
.stNumberInput>div>div>input{background:#111 !important;border:1px solid #2A2A2A !important;
  border-radius:10px !important;color:#FFF !important}
.stRadio>div{gap:6px !important}
.stRadio>div>label{background:#111 !important;border:1px solid #2A2A2A !important;
  border-radius:8px !important;padding:0.5rem 1rem !important;color:#AAA !important;
  font-size:0.85rem !important;font-weight:500 !important;cursor:pointer !important}
.stRadio>div>label:has(input:checked){background:#FFF !important;color:#000 !important;border-color:#FFF !important}
[data-testid="stMetric"]{background:#111;border:1px solid #1E1E1E;border-radius:12px;padding:0.9rem 1rem}
[data-testid="stMetricLabel"]{color:#555 !important;font-size:0.65rem !important;font-weight:600 !important;text-transform:uppercase !important;letter-spacing:1px !important}
[data-testid="stMetricValue"]{color:#FFF !important;font-size:1.25rem !important;font-weight:700 !important}
[data-testid="stMetricDelta"]{font-size:0.75rem !important}
hr{border:none !important;border-top:1px solid #1C1C1C !important;margin:0.9rem 0 !important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-thumb{background:#2A2A2A;border-radius:2px}
.stSpinner>div{border-top-color:#FFF !important}
.stAlert{border-radius:10px !important;font-size:0.88rem !important}

/* Logo */
.logo-wrap{display:inline-flex;align-items:center;gap:10px}
.logo-icon{background:#FFF;border-radius:9px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.logo-name{font-weight:800;color:#FFF;letter-spacing:-0.5px;line-height:1.1}
.logo-sub{font-size:0.52rem;color:#3A3A3A;letter-spacing:2px;text-transform:uppercase;margin-top:1px}

/* Topbar */
.page-title{font-size:1.4rem;font-weight:700;color:#FFF;letter-spacing:-0.4px}
.user-chip{display:inline-flex;align-items:center;gap:7px;background:#111;border:1px solid #222;
  border-radius:20px;padding:5px 12px 5px 9px;font-size:0.78rem;color:#888}
.live-dot{width:7px;height:7px;border-radius:50%;background:#4ADE80;box-shadow:0 0 5px #4ADE80}

.sec-label{font-size:0.68rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  color:#666;padding-bottom:0.5rem;margin-bottom:0.6rem;border-bottom:1px solid #1C1C1C;margin-top:1.2rem}

/* Feature cards */
.feat-card{background:#141414;border:1px solid #2A2A2A;border-radius:14px;padding:1.4rem 1.2rem;transition:border-color .2s,transform .15s}
.feat-card:hover{border-color:#3A3A3A;transform:translateY(-2px)}
.feat-num{font-size:0.6rem;color:#555;letter-spacing:1px;margin-bottom:0.5rem;font-family:monospace}
.feat-title{font-size:1rem;font-weight:700;color:#FFF;margin-bottom:0.4rem}
.feat-desc{font-size:0.82rem;color:#AAA;line-height:1.6}

/* Ticker header */
.tk-wrap{background:#111;border:1px solid #1E1E1E;border-radius:14px;padding:0.9rem 1.2rem;
  display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:0.9rem}
.tk-name{font-size:1.1rem;font-weight:700;color:#FFF;letter-spacing:-0.3px}
.tk-sym{font-size:0.72rem;color:#555;font-family:monospace;margin-top:2px}
.tk-pos{color:#4ADE80;font-weight:700;font-size:0.88rem}
.tk-neg{color:#F87171;font-weight:700;font-size:0.88rem}
.tk-price{font-size:1rem;font-weight:700;color:#FFF;margin-left:auto}
.tk-time{font-size:0.65rem;color:#444}

/* Cards */
.card{background:#111;border:1px solid #1E1E1E;border-radius:14px;overflow:hidden}
.card-hdr{padding:0.65rem 1rem;background:#0A0A0A;border-bottom:1px solid #1C1C1C;
  font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#666}
.card-row{display:flex;align-items:flex-start;gap:10px;padding:0.6rem 1rem;border-bottom:1px solid #141414}
.card-row:last-child{border-bottom:none}
.card-row:hover{background:#161616}

/* Pills */
.pill{font-size:0.62rem;font-weight:800;padding:3px 8px;border-radius:5px;
  letter-spacing:0.5px;font-family:monospace;flex-shrink:0;margin-top:2px}
.p-bull{background:#0A2010;color:#4ADE80;border:1px solid #1A4020}
.p-bear{background:#200A0A;color:#F87171;border:1px solid #401A1A}
.p-neu{background:#201A0A;color:#FBBF24;border:1px solid #402A1A}
.row-name{font-size:0.86rem;font-weight:600;color:#DDD}
.row-sub{font-size:0.75rem;color:#666;margin-top:2px}
.row-wt{color:#333;font-size:0.62rem}
.pat-bar{width:3px;border-radius:2px;min-height:34px;flex-shrink:0;margin-top:2px}
.bar-bull{background:#4ADE80}.bar-bear{background:#F87171}.bar-neu{background:#FBBF24}

/* Score */
.score-card{background:#111;border:1px solid #1E1E1E;border-radius:14px;padding:1rem}
.sc-hdr{font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#666;margin-bottom:0.9rem}
.sc-row{margin:0.6rem 0}
.sc-top{display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:4px}
.sc-lbl{color:#666}.sc-bull{color:#4ADE80;font-weight:700}.sc-bear{color:#F87171;font-weight:700}
.sc-track{height:5px;border-radius:3px;background:#1E1E1E;overflow:hidden}
.sc-fb{height:100%;border-radius:3px;background:#4ADE80}
.sc-rb{height:100%;border-radius:3px;background:#F87171}

/* Levels */
.lvl-row{display:flex;justify-content:space-between;align-items:center;padding:0.5rem 1rem;border-bottom:1px solid #141414}
.lvl-row:last-child{border-bottom:none}.lvl-row:hover{background:#161616}
.lvl-k{font-size:0.74rem;color:#666;text-transform:uppercase;letter-spacing:0.5px}
.lvl-v{font-size:0.86rem;font-weight:600;color:#DDD;font-family:monospace}
.lvl-hl{color:#FFF !important}

/* Quick stats */
.qs-row{display:flex;justify-content:space-between;align-items:center;padding:0.52rem 1rem;border-bottom:1px solid #141414}
.qs-row:last-child{border-bottom:none}
.qs-k{font-size:0.78rem;color:#666}.qs-v{font-size:0.82rem;font-weight:600;color:#DDD;font-family:monospace}
.qs-bull{color:#4ADE80 !important}.qs-bear{color:#F87171 !important}.qs-neu{color:#FBBF24 !important}

/* Verdict */
.verd{border-radius:14px;padding:1.4rem 1.6rem;margin-top:0.5rem}
.vb{background:#050E05;border:1px solid #1A4020}
.vs{background:#0E0505;border:1px solid #401A1A}
.vw{background:#0E0A05;border:1px solid #403010}
.vt{font-size:1.8rem;font-weight:800;letter-spacing:-0.5px;margin-bottom:0.5rem}
.vt-b{color:#4ADE80}.vt-s{color:#F87171}.vt-w{color:#FBBF24}
.vbody{font-size:0.88rem;color:#888;line-height:1.7}
.vbody b{color:#CCC}
.v-chips{display:flex;gap:10px;margin-top:1rem;flex-wrap:wrap}
.vchip{background:#141414;border:1px solid #222;border-radius:9px;padding:0.38rem 0.85rem}
.vc-l{font-size:0.58rem;text-transform:uppercase;letter-spacing:0.8px;color:#555;display:block;margin-bottom:2px}
.vc-v{font-size:0.85rem;font-weight:700;color:#FFF}

/* Reasoning box */
.reason-box{background:#0A0A0A;border:1px solid #1E1E1E;border-radius:14px;padding:1.2rem 1.4rem;margin-top:0.8rem}
.reason-title{font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#666;margin-bottom:0.8rem}
.reason-item{display:flex;gap:10px;padding:0.45rem 0;font-size:0.84rem;color:#AAA;line-height:1.5;border-bottom:1px solid #141414}
.reason-item:last-child{border-bottom:none}
.reason-icon{flex-shrink:0;margin-top:1px;font-size:0.82rem}

/* Fundamental card */
.fund-section{background:#0A0A0A;border:1px solid #1E1E1E;border-radius:16px;padding:1.3rem 1.5rem;margin-bottom:1rem}
.fund-title{font-size:0.9rem;font-weight:700;color:#FFF;margin-bottom:0.3rem}
.fund-sub{font-size:0.78rem;color:#666;margin-bottom:1rem;line-height:1.5}
.fund-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:1rem}
.fund-cell{background:#141414;border:1px solid #1E1E1E;border-radius:10px;padding:0.75rem 1rem}
.fund-cell-label{font-size:0.6rem;text-transform:uppercase;letter-spacing:1px;color:#555;margin-bottom:3px}
.fund-cell-value{font-size:0.92rem;font-weight:700;color:#FFF}
.fund-cell-sub{font-size:0.7rem;color:#555;margin-top:2px}
.fund-verdict{border-radius:12px;padding:1.1rem 1.4rem;margin-top:0.8rem}
.fv-bull{background:#050E05;border:1px solid #1A4020}
.fv-bear{background:#0E0505;border:1px solid #401A1A}
.fv-neu{background:#0E0A05;border:1px solid #403010}
.fv-title{font-size:1.1rem;font-weight:700;margin-bottom:0.4rem}
.fv-tb{color:#4ADE80}.fv-ts{color:#F87171}.fv-tw{color:#FBBF24}
.fv-body{font-size:0.84rem;color:#888;line-height:1.65}
.fv-body b{color:#CCC}
.fv-point{display:flex;gap:10px;padding:0.4rem 0;font-size:0.83rem;color:#AAA;border-bottom:1px solid #141414}
.fv-point:last-child{border-bottom:none}
.sector-badge{display:inline-flex;align-items:center;gap:6px;background:#141414;border:1px solid #222;
  border-radius:20px;padding:3px 10px;font-size:0.72rem;color:#888;margin-right:6px;margin-bottom:4px}

/* Login */
.login-outer{min-height:80vh;display:flex;align-items:center;justify-content:center;padding:2rem}
.login-card{background:#0A0A0A;border:1px solid #1E1E1E;border-radius:20px;padding:2.8rem 3rem;
  width:100%;max-width:400px;text-align:center}
.login-title{font-size:1.5rem;font-weight:700;color:#FFF;margin:1.2rem 0 0.4rem;letter-spacing:-0.4px}
.login-sub{font-size:0.84rem;color:#555;margin-bottom:1.8rem;line-height:1.6}
.login-tab{display:inline-flex;background:#111;border:1px solid #1E1E1E;border-radius:10px;padding:3px;gap:3px;margin-bottom:1.5rem}
.ltab{padding:6px 20px;border-radius:8px;font-size:0.8rem;font-weight:600;cursor:pointer;color:#555;transition:all .15s}
.ltab.active{background:#FFF;color:#000}

/* History */
.hist-item{background:#111;border:1px solid #1E1E1E;border-radius:12px;padding:0.9rem 1.1rem;
  display:flex;align-items:center;gap:14px;margin-bottom:8px;flex-wrap:wrap}

/* Trade planner */
.tp-cell{background:#141414;border:1px solid #1E1E1E;border-radius:10px;padding:0.8rem 1rem}
.tp-cell-label{font-size:0.6rem;text-transform:uppercase;letter-spacing:1px;color:#555;margin-bottom:3px}
.tp-cell-value{font-size:0.95rem;font-weight:700;color:#FFF;font-family:monospace}
.tp-cell-sub{font-size:0.7rem;color:#555;margin-top:2px}
.tgt-row{display:flex;align-items:center;gap:12px;padding:0.6rem 0;border-bottom:1px solid #141414}
.tgt-row:last-child{border-bottom:none}
.tgt-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.rule-item{display:flex;gap:10px;padding:0.4rem 0;font-size:0.82rem;color:#AAA;line-height:1.5}
.rule-dot{color:#FFF;font-weight:700;flex-shrink:0;margin-top:1px}
.risk-badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:20px;font-size:0.72rem;font-weight:700}
.rb-low{background:#0A2010;color:#4ADE80;border:1px solid #1A4020}
.rb-med{background:#201A0A;color:#FBBF24;border:1px solid #402A1A}
.rb-high{background:#200A0A;color:#F87171;border:1px solid #401A1A}
.tp-badge{font-size:0.7rem;font-weight:800;padding:4px 12px;border-radius:6px;letter-spacing:0.5px;font-family:monospace}
.tb-buy{background:#0A2010;color:#4ADE80;border:1px solid #1A4020}
.tb-sell{background:#200A0A;color:#F87171;border:1px solid #401A1A}
.tb-wait{background:#201A0A;color:#FBBF24;border:1px solid #402A1A}
.wl-icon{width:40px;height:40px;border-radius:10px;background:#141414;border:1px solid #2A2A2A;
  display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:800;color:#888;flex-shrink:0}

/* Disclaimer */
.disclaimer{background:#0A0A0A;border:1px solid #1C1C1C;border-radius:12px;padding:1rem 1.2rem;
  margin-top:1.2rem;font-size:0.8rem;color:#666;line-height:1.7}
.disclaimer strong{color:#999}

/* Team sharing */
.team-card{background:#111;border:1px solid #1E1E1E;border-radius:12px;padding:1rem 1.2rem;
  display:flex;align-items:center;gap:12px;margin-bottom:8px}
.team-avatar{width:40px;height:40px;border-radius:50%;background:#1A1A1A;border:1px solid #2A2A2A;
  display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;color:#888;flex-shrink:0}
.role-badge{font-size:0.65rem;font-weight:700;padding:2px 8px;border-radius:4px;
  background:#141414;color:#888;border:1px solid #222;letter-spacing:0.5px}
.role-owner{background:#0A1A2A;color:#60A5FA;border-color:#1A3A5A}
.role-analyst{background:#1A1A0A;color:#FBBF24;border-color:#3A3A1A}
</style>
""", unsafe_allow_html=True)

def logo(sz="md"):
    dim={"lg":"42px","md":"36px","sm":"30px"}.get(sz,"36px")
    ts={"lg":"1.3rem","md":"1.15rem","sm":"0.95rem"}.get(sz,"1.15rem")
    ico=int(dim[:-2])-12
    return f"""<div class="logo-wrap">
      <div class="logo-icon" style="width:{dim};height:{dim};border-radius:{int(dim[:-2])//4}px">
        <svg viewBox="0 0 24 24" fill="none" width="{ico}" height="{ico}">
          <polyline points="2,16 8,9 13,13 22,4" stroke="#000" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
          <polyline points="17,4 22,4 22,9" stroke="#000" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
          <line x1="2" y1="20" x2="22" y2="20" stroke="#000" stroke-width="1" opacity=".3"/>
        </svg>
      </div>
      <div>
        <div class="logo-name" style="font-size:{ts}">Ace-Trade</div>
        <div class="logo-sub">Professional Terminal</div>
      </div>
    </div>"""

# ── Session ───────────────────────────────────────────────────────────────────
for k,v in [("logged_in",False),("history",[]),("page","Dashboard"),("watchlist",[]),("login_tab","username")]:
    if k not in st.session_state: st.session_state[k]=v

# ── Login ─────────────────────────────────────────────────────────────────────
if not st.session_state["logged_in"]:
    st.markdown(f'<div class="login-outer"><div class="login-card">{logo("md")}<div class="login-title">Welcome back</div><p class="login-sub">Sign in to your professional<br>trading analysis terminal</p></div></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        login_mode = st.radio("Login with", ["Username", "Email"], horizontal=True, label_visibility="collapsed")
        if login_mode == "Username":
            u = st.text_input("Username", placeholder="Enter your username")
            p = st.text_input("Password", type="password", placeholder="Enter your password")
            if st.button("Sign In →", use_container_width=True):
                if u in USERS_H and hash_pw(p)==USERS_H[u]:
                    st.session_state.update({"logged_in":True,"username":u,"user_info":USERS[u]}); st.rerun()
                else: st.error("Invalid username or password.")
        else:
            em = st.text_input("Email", placeholder="Enter your registered email")
            p  = st.text_input("Password", type="password", placeholder="Enter your password")
            if st.button("Sign In →", use_container_width=True):
                u_from_email = EMAIL_MAP.get(em.lower().strip())
                if u_from_email and hash_pw(p)==USERS_H[u_from_email]:
                    st.session_state.update({"logged_in":True,"username":u_from_email,"user_info":USERS[u_from_email]}); st.rerun()
                else: st.error("Invalid email or password.")
        st.markdown('<p style="color:#222;font-size:0.68rem;text-align:center;margin-top:8px">admin / acetrade123 &nbsp;·&nbsp; admin@acetrade.in</p>', unsafe_allow_html=True)
    st.stop()

uinfo = st.session_state.get("user_info", USERS.get(st.session_state.get("username","admin"), {}))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(logo("sm"), unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    for icon,pg in [("◈  ","Dashboard"),("◎  ","Trade Planner"),("◉  ","Fundamental Analysis"),("◷  ","Search History"),("⊞  ","Team & Sharing"),("▣  ","About")]:
        if st.button(f"{icon}{pg}", key=f"nav_{pg}", use_container_width=True):
            st.session_state["page"]=pg; st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.session_state["history"]:
        st.markdown('<div style="font-size:0.6rem;color:#333;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:6px">Recent</div>', unsafe_allow_html=True)
        for h in reversed(st.session_state["history"][-5:]):
            vc={"BUY":"#4ADE80","SELL":"#F87171"}.get(h["verdict"],"#FBBF24")
            st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:0.35rem 0.2rem;margin-bottom:2px"><div><div style="font-size:0.76rem;font-weight:600;color:#CCC;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{h["name"]}</div><div style="font-size:0.6rem;color:#333">{h["time"]}</div></div><div style="color:{vc};font-size:0.65rem;font-weight:800;flex-shrink:0">{h["verdict"]}</div></div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.7rem;color:#333;margin-bottom:6px">{uinfo.get("name","User")} <span style="color:#555">({uinfo.get("role","Analyst")})</span></div>', unsafe_allow_html=True)
    if st.button("Sign Out", key="so", use_container_width=True):
        st.session_state["logged_in"]=False; st.rerun()

page = st.session_state["page"]
t1,t2 = st.columns([4,1])
with t1: st.markdown(f'<div class="page-title">{page}</div>', unsafe_allow_html=True)
with t2: st.markdown(f'<div style="text-align:right;padding-top:0.15rem"><span class="user-chip"><span class="live-dot"></span>{uinfo.get("name","User")}</span></div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Core analysis ─────────────────────────────────────────────────────────────
def compute(df):
    c,h,l,v=df["Close"],df["High"],df["Low"],df["Volume"]
    df["EMA20"]=ta.trend.ema_indicator(c,window=20)
    df["EMA50"]=ta.trend.ema_indicator(c,window=50)
    df["EMA200"]=ta.trend.ema_indicator(c,window=200)
    df["ADX"]=ta.trend.adx(h,l,c,window=14)
    df["RSI"]=ta.momentum.rsi(c,window=14)
    df["MACD"]=ta.trend.macd(c)
    df["MACD_signal"]=ta.trend.macd_signal(c)
    df["MACD_hist"]=ta.trend.macd_diff(c)
    df["Stoch_k"]=ta.momentum.stoch(h,l,c)
    bb=ta.volatility.BollingerBands(c,window=20)
    df["BB_upper"]=bb.bollinger_hband();df["BB_lower"]=bb.bollinger_lband();df["BB_mid"]=bb.bollinger_mavg()
    df["ATR"]=ta.volatility.average_true_range(h,l,c)
    df["OBV"]=ta.volume.on_balance_volume(c,v)
    df["Vol_avg"]=v.rolling(20).mean()
    return df

def get_patterns(df):
    P,c,h,l=[],df["Close"],df["High"],df["Low"];last=c.iloc[-1]
    if len(h)>=20:
        fh=h.iloc[-20:-10].max()
        if fh>0 and abs(fh-h.iloc[-10:].max())/fh<0.025:P.append(("Double Top","Bearish","Two peaks at same level — bearish reversal"))
    if len(l)>=20:
        fl=l.iloc[-20:-10].min()
        if fl>0 and abs(fl-l.iloc[-10:].min())/fl<0.025:P.append(("Double Bottom","Bullish","Two troughs at same level — bullish reversal"))
    if len(h)>=15:
        if (h.iloc[-15:].max()-h.iloc[-15:].min())/h.iloc[-15:].max()<0.018 and l.iloc[-10:].is_monotonic_increasing:
            P.append(("Ascending Triangle","Bullish","Flat resistance + rising lows — breakout likely"))
    if len(l)>=15:
        if (l.iloc[-15:].max()-l.iloc[-15:].min())/l.iloc[-15:].max()<0.018 and h.iloc[-10:].is_monotonic_decreasing:
            P.append(("Descending Triangle","Bearish","Flat support + falling highs — breakdown likely"))
    ph,pl=h.iloc[-20:].max(),l.iloc[-20:].min()
    if last>=ph*0.985:P.append(("20D Breakout","Bullish","Price at 20-day high — strong momentum"))
    if last<=pl*1.015:P.append(("20D Breakdown","Bearish","Price at 20-day low — selling pressure"))
    if len(c)>=10:
        if c.iloc[-5:].is_monotonic_increasing:P.append(("Rally Streak","Bullish","5 consecutive up closes"))
        elif c.iloc[-5:].is_monotonic_decreasing:P.append(("Drop Streak","Bearish","5 consecutive down closes"))
    if not P:P.append(("Consolidation","Neutral","Sideways — await breakout"))
    return P

def get_signals(df):
    la,pr=df.iloc[-1],df.iloc[-2];cl=la["Close"];S=[]
    S.append(("Price vs EMA 50","Bullish" if cl>la["EMA50"] else "Bearish",f"Price {'above' if cl>la['EMA50'] else 'below'} EMA 50",2))
    S.append(("EMA 50 vs EMA 200","Bullish" if la["EMA50"]>la["EMA200"] else "Bearish","Long-term structure",2))
    if pr["EMA50"]<pr["EMA200"] and la["EMA50"]>=la["EMA200"]:S.append(("Golden Cross","Bullish","EMA50 crossed above EMA200 — major buy signal",3))
    elif pr["EMA50"]>pr["EMA200"] and la["EMA50"]<=la["EMA200"]:S.append(("Death Cross","Bearish","EMA50 crossed below EMA200 — major sell signal",3))
    r=la["RSI"]
    if r<=35:S.append(("RSI","Bullish",f"RSI {r:.1f} — oversold, bounce likely",2))
    elif r>=65:S.append(("RSI","Bearish",f"RSI {r:.1f} — overbought, pullback likely",2))
    elif r>50:S.append(("RSI","Bullish",f"RSI {r:.1f} — above 50, bullish bias",1))
    else:S.append(("RSI","Bearish",f"RSI {r:.1f} — below 50, bearish bias",1))
    S.append(("MACD","Bullish" if la["MACD"]>la["MACD_signal"] else "Bearish","MACD vs Signal line",2))
    if la["MACD_hist"]>0 and la["MACD_hist"]>pr["MACD_hist"]:S.append(("MACD Histogram","Bullish","Histogram rising — momentum building",1))
    elif la["MACD_hist"]<0 and la["MACD_hist"]<pr["MACD_hist"]:S.append(("MACD Histogram","Bearish","Histogram falling — momentum fading",1))
    else:S.append(("MACD Histogram","Neutral","Histogram mixed",1))
    if cl<la["BB_lower"]:S.append(("Bollinger Bands","Bullish","Below lower band — oversold bounce zone",1))
    elif cl>la["BB_upper"]:S.append(("Bollinger Bands","Bearish","Above upper band — overbought zone",1))
    elif cl>la["BB_mid"]:S.append(("Bollinger Bands","Bullish","Above BB midline — bullish bias",1))
    else:S.append(("Bollinger Bands","Bearish","Below BB midline — bearish bias",1))
    sk=la["Stoch_k"]
    if sk<25:S.append(("Stochastic","Bullish",f"Stoch {sk:.1f} — oversold",1))
    elif sk>75:S.append(("Stochastic","Bearish",f"Stoch {sk:.1f} — overbought",1))
    elif sk>50:S.append(("Stochastic","Bullish",f"Stoch {sk:.1f} — above midline",1))
    else:S.append(("Stochastic","Bearish",f"Stoch {sk:.1f} — below midline",1))
    if la["Vol_avg"]>0:
        vr=la["Volume"]/la["Vol_avg"]
        if vr>1.3 and cl>pr["Close"]:S.append(("Volume","Bullish",f"Vol {vr:.1f}× avg on up-day",1))
        elif vr>1.3 and cl<pr["Close"]:S.append(("Volume","Bearish",f"Vol {vr:.1f}× avg on down-day",1))
        else:S.append(("Volume","Neutral",f"Vol {vr:.1f}× average — normal",1))
    adx=la["ADX"]
    if adx>25:S.append(("ADX Strength","Bullish" if cl>la["EMA50"] else "Bearish",f"ADX {adx:.1f} — strong trend",1))
    else:S.append(("ADX Strength","Neutral",f"ADX {adx:.1f} — weak/no trend",1))
    return S

def score(sigs):
    bs=sum(w for _,d,_,w in sigs if d=="Bullish")
    rs=sum(w for _,d,_,w in sigs if d=="Bearish")
    tt=bs+rs+sum(w for _,d,_,w in sigs if d=="Neutral")
    return bs,rs,tt,sum(1 for _,d,_,_ in sigs if d=="Bullish"),sum(1 for _,d,_,_ in sigs if d=="Bearish"),len(sigs)

def verdict(bs,rs,tt):
    if tt==0:return "WAIT","wait"
    if bs/tt*100>=55:return "BUY","buy"
    if rs/tt*100>=55:return "SELL","sell"
    return "WAIT","wait"

def get_sr(df):
    c,h,l=df["Close"],df["High"],df["Low"];last=c.iloc[-1]
    res=h.iloc[-50:].max();sup=l.iloc[-50:].min()
    piv=(h.iloc[-1]+l.iloc[-1]+c.iloc[-1])/3
    return {"Resistance (50D)":res,"Support (50D)":sup,"Pivot":piv,
            "R1":2*piv-l.iloc[-1],"S1":2*piv-h.iloc[-1],
            "52W High":h.iloc[-252:].max() if len(h)>=252 else h.max(),
            "52W Low":l.iloc[-252:].min() if len(l)>=252 else l.min(),
            "→ Resistance":((res-last)/last)*100,"← Support":((last-sup)/last)*100}

def build_reasoning(sigs, pat_list, verd_type, la, bp, rp, sc_):
    reasons = []
    bullish_sigs = [(n,r) for n,d,r,w in sigs if d=="Bullish" and w>=2]
    bearish_sigs = [(n,r) for n,d,r,w in sigs if d=="Bearish" and w>=2]
    bull_pats = [(n,r) for n,d,r in pat_list if d=="Bullish"]
    bear_pats = [(n,r) for n,d,r in pat_list if d=="Bearish"]

    if verd_type=="buy":
        reasons.append(("✅","Primary","Weighted score is BULLISH at {}% — majority of {} indicators agree".format(bp,sc_)))
        for n,r in bullish_sigs[:3]: reasons.append(("▲","Key Bullish Signal",f"{n}: {r}"))
        for n,r in bull_pats[:2]:    reasons.append(("◆","Chart Pattern",f"{n}: {r}"))
        if bearish_sigs:             reasons.append(("⚠️","Risk Factor",f"{bearish_sigs[0][0]} is bearish — watch this closely"))
        reasons.append(("◎","Trend","EMA structure and price action support the bullish case"))
    elif verd_type=="sell":
        reasons.append(("🔴","Primary","Weighted score is BEARISH at {}% — majority of {} indicators agree".format(rp,sc_)))
        for n,r in bearish_sigs[:3]: reasons.append(("▼","Key Bearish Signal",f"{n}: {r}"))
        for n,r in bear_pats[:2]:    reasons.append(("◆","Chart Pattern",f"{n}: {r}"))
        if bullish_sigs:             reasons.append(("⚠️","Contrarian Signal",f"{bullish_sigs[0][0]} — monitor for reversal"))
        reasons.append(("◎","Trend","Price action and momentum confirm bearish setup"))
    else:
        reasons.append(("⏳","Primary","Mixed signals — {} bullish vs {} bearish. No dominant trend.".format(bp,rp)))
        for n,r in bullish_sigs[:2]: reasons.append(("▲","Bullish Factor",f"{n}: {r}"))
        for n,r in bearish_sigs[:2]: reasons.append(("▼","Bearish Factor",f"{n}: {r}"))
        reasons.append(("◎","Advice","Wait for a breakout above resistance or breakdown below support for confirmation"))
    return reasons

def fundamental_analysis(ticker_sym, info):
    points_bull, points_bear, points_neu = [], [], []
    fund_data = {}
    na="N/A"

    pe     = info.get("trailingPE") or info.get("forwardPE")
    pb     = info.get("priceToBook")
    roe    = info.get("returnOnEquity")
    de     = info.get("debtToEquity")
    rev_g  = info.get("revenueGrowth")
    earn_g = info.get("earningsGrowth")
    profit = info.get("profitMargins")
    cur_r  = info.get("currentRatio")
    div_y  = info.get("dividendYield")
    beta   = info.get("beta")
    mktcap = info.get("marketCap")
    sector = info.get("sector","")
    industry= info.get("industry","")
    rec    = info.get("recommendationKey","")
    tgt    = info.get("targetMeanPrice")
    price  = info.get("currentPrice") or info.get("regularMarketPrice")
    eps    = info.get("trailingEps")
    fcf    = info.get("freeCashflow")
    gross_m= info.get("grossMargins")
    op_m   = info.get("operatingMargins")

    def fmt_cr(v):
        if v is None: return na
        if v>=1e12: return f"₹{v/1e12:.2f}T"
        if v>=1e9:  return f"₹{v/1e9:.2f}B"
        if v>=1e7:  return f"₹{v/1e7:.2f}Cr"
        return f"₹{v:,.0f}"

    fund_data = {
        "P/E Ratio":       (f"{pe:.1f}x" if pe else na, "Price to Earnings"),
        "P/B Ratio":       (f"{pb:.2f}x" if pb else na, "Price to Book Value"),
        "ROE":             (f"{roe*100:.1f}%" if roe else na, "Return on Equity"),
        "Debt/Equity":     (f"{de:.2f}" if de else na, "Financial Leverage"),
        "Revenue Growth":  (f"{rev_g*100:.1f}%" if rev_g else na, "YoY Revenue Growth"),
        "Earnings Growth": (f"{earn_g*100:.1f}%" if earn_g else na, "YoY Earnings Growth"),
        "Profit Margin":   (f"{profit*100:.1f}%" if profit else na, "Net Profit Margin"),
        "Current Ratio":   (f"{cur_r:.2f}" if cur_r else na, "Liquidity Ratio"),
        "Dividend Yield":  (f"{div_y*100:.2f}%" if div_y else na, "Annual Dividend %"),
        "Beta":            (f"{beta:.2f}" if beta else na, "Market Sensitivity"),
        "Market Cap":      (fmt_cr(mktcap), "Total Market Cap"),
        "EPS (TTM)":       (f"₹{eps:.2f}" if eps else na, "Earnings Per Share"),
    }

    # Build fundamental verdict
    if pe:
        if pe < 15:   points_bull.append("P/E ratio is low (<15×) — stock may be undervalued relative to earnings")
        elif pe > 40: points_bear.append(f"P/E ratio is high ({pe:.0f}×) — expensive valuation, high expectations priced in")
        else:         points_neu.append(f"P/E ratio is moderate ({pe:.0f}×) — fair valuation")
    if pb:
        if pb < 1.5:  points_bull.append(f"P/B ratio is low ({pb:.1f}×) — trading near or below book value")
        elif pb > 6:  points_bear.append(f"P/B ratio is high ({pb:.1f}×) — significant premium to book value")
    if roe:
        if roe > 0.18: points_bull.append(f"Strong ROE of {roe*100:.1f}% — efficient capital utilisation")
        elif roe < 0.08: points_bear.append(f"Weak ROE of {roe*100:.1f}% — poor return on equity")
    if de:
        if de < 30:   points_bull.append(f"Low debt/equity ratio ({de:.0f}) — conservative balance sheet")
        elif de > 100: points_bear.append(f"High debt/equity ({de:.0f}) — elevated leverage, watch interest coverage")
    if rev_g:
        if rev_g > 0.15:  points_bull.append(f"Strong revenue growth of {rev_g*100:.1f}% YoY — business is expanding")
        elif rev_g < 0:   points_bear.append(f"Revenue declining ({rev_g*100:.1f}% YoY) — top-line under pressure")
    if earn_g:
        if earn_g > 0.20: points_bull.append(f"Earnings growing at {earn_g*100:.1f}% YoY — strong profit momentum")
        elif earn_g < 0:  points_bear.append(f"Earnings declining ({earn_g*100:.1f}%) — bottom-line deteriorating")
    if profit:
        if profit > 0.15: points_bull.append(f"Net profit margin of {profit*100:.1f}% — highly profitable operations")
        elif profit < 0.05: points_bear.append(f"Thin net margin of {profit*100:.1f}% — low profitability")
    if cur_r:
        if cur_r > 1.5:   points_bull.append(f"Current ratio {cur_r:.1f} — strong short-term liquidity position")
        elif cur_r < 1.0: points_bear.append(f"Current ratio {cur_r:.1f} — liquidity concern, assets < liabilities")
    if tgt and price and price>0:
        upside = (tgt-price)/price*100
        if upside > 15:   points_bull.append(f"Analyst consensus target ₹{tgt:.0f} — implies {upside:.0f}% upside from current price")
        elif upside < -5: points_bear.append(f"Analyst consensus target ₹{tgt:.0f} — implies limited/negative upside")
        else:             points_neu.append(f"Analyst consensus target ₹{tgt:.0f} — limited upside at current levels")
    if fcf and fcf>0:     points_bull.append("Positive free cash flow — company generates cash after capex")
    elif fcf and fcf<0:   points_bear.append("Negative free cash flow — company is consuming cash")

    bull_score = len(points_bull); bear_score = len(points_bear); total = bull_score+bear_score+len(points_neu)
    if total==0: fverd,ftype="SEEK MORE INSIGHT","neu"
    elif bull_score/max(total,1)>=0.55: fverd,ftype="FUNDAMENTALLY BULLISH","bull"
    elif bear_score/max(total,1)>=0.55: fverd,ftype="FUNDAMENTALLY BEARISH","bear"
    else: fverd,ftype="SEEK MORE INSIGHT","neu"

    return fund_data, points_bull, points_bear, points_neu, fverd, ftype, sector, industry, rec, tgt, price

def build_chart(df):
    BG,GR="#000","#111"
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[.52,.16,.16,.16],
        vertical_spacing=.015,subplot_titles=("","RSI (14)","MACD","Volume"))
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],name="OHLC",
        increasing_line_color="#4ADE80",increasing_fillcolor="rgba(74,222,128,.75)",
        decreasing_line_color="#F87171",decreasing_fillcolor="rgba(248,113,113,.75)",line=dict(width=1)),row=1,col=1)
    for col,dash,w,nm,key in [("#FBBF24","dot",1,"EMA 20","EMA20"),("#FFF","dash",1.4,"EMA 50","EMA50"),("#888","solid",1.8,"EMA 200","EMA200")]:
        fig.add_trace(go.Scatter(x=df.index,y=df[key],name=nm,line=dict(color=col,width=w,dash=dash),opacity=.9),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BB_upper"],line=dict(color="rgba(255,255,255,.06)",width=1),showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BB_lower"],line=dict(color="rgba(255,255,255,.06)",width=1),fill="tonexty",fillcolor="rgba(255,255,255,.02)",showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["RSI"],name="RSI",line=dict(color="#AAA",width=1.5)),row=2,col=1)
    for y,c in [(70,"rgba(248,113,113,.3)"),(50,"rgba(100,100,100,.2)"),(30,"rgba(74,222,128,.3)")]:
        fig.add_hline(y=y,line_dash="dot",line_color=c,line_width=.8,row=2,col=1)
    hc=["#4ADE80" if v>=0 else "#F87171" for v in df["MACD_hist"]]
    fig.add_trace(go.Bar(x=df.index,y=df["MACD_hist"],name="Hist",marker_color=hc,opacity=.55,showlegend=False),row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["MACD"],name="MACD",line=dict(color="#FFF",width=1.4)),row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["MACD_signal"],name="Signal",line=dict(color="#AAA",width=1.4)),row=3,col=1)
    fig.add_hline(y=0,line_color="rgba(100,100,100,.2)",line_width=.8,row=3,col=1)
    vcols=["rgba(74,222,128,.45)" if df["Close"].iloc[i]>=df["Open"].iloc[i] else "rgba(248,113,113,.45)" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Vol",marker_color=vcols,showlegend=False),row=4,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["Vol_avg"],name="Vol MA",line=dict(color="rgba(255,255,255,.25)",width=1,dash="dot"),showlegend=False),row=4,col=1)
    fig.update_layout(height=860,template="plotly_dark",paper_bgcolor=BG,plot_bgcolor=BG,
        legend=dict(orientation="h",y=1.022,x=0,font=dict(size=10,color="#555"),bgcolor="rgba(0,0,0,0)",borderwidth=0),
        xaxis_rangeslider_visible=False,margin=dict(l=0,r=0,t=16,b=0),
        font=dict(family="Inter",size=11,color="#555"),hovermode="x unified",
        hoverlabel=dict(bgcolor="#111",bordercolor="#222",font_size=11,font_family="monospace"))
    for i in range(1,5):
        fig.update_xaxes(gridcolor=GR,zerolinecolor=GR,showspikes=True,spikecolor="#333",spikethickness=1,spikemode="across",row=i,col=1)
        fig.update_yaxes(gridcolor=GR,zerolinecolor=GR,row=i,col=1)
    return fig

def render_analysis(sel_name, sel_ticker, sel_exch, period, interval, save_hist=True):
    t_obj = yf.Ticker(sel_ticker)
    df    = t_obj.history(period=period, interval=interval)
    if df is None or df.empty: return None, None
    df = compute(df)
    sig_list = get_signals(df); pat_list = get_patterns(df)
    bs,rs,tt,bc,rc,sc_ = score(sig_list)
    verd,vtype = verdict(bs,rs,tt)
    sr = get_sr(df)
    la,pr = df.iloc[-1],df.iloc[-2]
    chgp = (la["Close"]-pr["Close"])/pr["Close"]*100
    info = t_obj.info
    name = info.get("longName", sel_name)
    curr = "₹" if (".NS" in sel_ticker or ".BO" in sel_ticker) else ""
    bp = int(bs/tt*100) if tt else 0
    rp = int(rs/tt*100) if tt else 0
    if save_hist:
        st.session_state["history"].append({"name":sel_name,"ticker":sel_ticker,"verdict":verd,
            "price":f"{curr}{la['Close']:.2f}","chg":f"{chgp:+.2f}%","bp":bp,"rp":rp,
            "time":datetime.now().strftime("%d %b %H:%M"),"period":period})

    arrow = "▲" if chgp>=0 else "▼"; chg_cls = "tk-pos" if chgp>=0 else "tk-neg"
    st.markdown(f'<div class="tk-wrap"><div><div class="tk-name">{name}</div><div class="tk-sym">{sel_ticker} · {sel_exch} · {interval}</div></div><span class="{chg_cls}">{arrow} {abs(chgp):.2f}%</span><div class="tk-price">{curr}{la["Close"]:.2f}</div><div class="tk-time">{datetime.now().strftime("%d %b %Y, %H:%M")}</div></div>', unsafe_allow_html=True)
    m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
    m1.metric("Close",f"{curr}{la['Close']:.2f}",f"{chgp:+.2f}%")
    m2.metric("RSI 14",f"{la['RSI']:.1f}")
    m3.metric("MACD",f"{la['MACD']:.2f}")
    m4.metric("ATR",f"{la['ATR']:.2f}")
    m5.metric("ADX",f"{la['ADX']:.1f}")
    m6.metric("EMA 50",f"{curr}{la['EMA50']:.2f}")
    m7.metric("EMA 200",f"{curr}{la['EMA200']:.2f}")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.plotly_chart(build_chart(df), use_container_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    col_l,col_m,col_r = st.columns([1.4,1,0.9])
    with col_l:
        rows="".join(f'<div class="card-row"><span class="pill {"p-bull" if d=="Bullish" else ("p-bear" if d=="Bearish" else "p-neu")}">{"BULL" if d=="Bullish" else ("BEAR" if d=="Bearish" else "NEUT")}</span><div><div class="row-name">{sn} <span class="row-wt">{"●●" if w>=2 else "●"}</span></div><div class="row-sub">{rs_}</div></div></div>' for sn,d,rs_,w in sig_list)
        st.markdown(f'<div class="card"><div class="card-hdr">Signal Breakdown — {sc_} indicators</div>{rows}</div>', unsafe_allow_html=True)
    with col_m:
        prows="".join(f'<div class="card-row"><div class="pat-bar {"bar-bull" if pd_=="Bullish" else ("bar-bear" if pd_=="Bearish" else "bar-neu")}"></div><div><div class="row-name">{pn}</div><div class="row-sub">{pr_}</div></div></div>' for pn,pd_,pr_ in pat_list)
        st.markdown(f'<div class="card"><div class="card-hdr">Chart Patterns</div>{prows}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        lvls="".join(f'<div class="lvl-row"><span class="lvl-k">{k}</span><span class="lvl-v {"lvl-hl" if k in ["Resistance (50D)","Support (50D)","52W High","52W Low"] else ""}">{"" if "→" in k or "←" in k else curr}{f"{v:+.2f}%" if "→" in k or "←" in k else f"{v:.2f}"}</span></div>' for k,v in sr.items())
        st.markdown(f'<div class="card"><div class="card-hdr">Key Levels</div>{lvls}</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown(f'<div class="score-card"><div class="sc-hdr">Weighted Score</div><div class="sc-row"><div class="sc-top"><span class="sc-lbl">Bullish — {bc} signals</span><span class="sc-bull">{bp}%</span></div><div class="sc-track"><div class="sc-fb" style="width:{bp}%"></div></div></div><div class="sc-row" style="margin-top:12px"><div class="sc-top"><span class="sc-lbl">Bearish — {rc} signals</span><span class="sc-bear">{rp}%</span></div><div class="sc-track"><div class="sc-rb" style="width:{rp}%"></div></div></div><div style="margin-top:14px;padding-top:12px;border-top:1px solid #1A1A1A"><div style="font-size:0.6rem;color:#333;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Total indicators</div><div style="font-size:1.3rem;font-weight:700;color:#FFF">{sc_}</div></div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        rsi_v=la["RSI"];adx_v=la["ADX"]
        rsi_l="Oversold" if rsi_v<35 else ("Overbought" if rsi_v>65 else "Neutral")
        rsi_c="qs-bull" if rsi_v<35 else ("qs-bear" if rsi_v>65 else "qs-neu")
        mac_l="Bullish" if la["MACD"]>la["MACD_signal"] else "Bearish"
        mac_c="qs-bull" if la["MACD"]>la["MACD_signal"] else "qs-bear"
        ema_l="Uptrend" if la["Close"]>la["EMA200"] else "Downtrend"
        ema_c="qs-bull" if la["Close"]>la["EMA200"] else "qs-bear"
        adx_l="Strong" if adx_v>25 else "Weak"
        adx_c="qs-bull" if adx_v>25 else "qs-neu"
        qs="".join(f'<div class="qs-row"><span class="qs-k">{k}</span><span class="qs-v {vc}">{v}</span></div>' for k,v,vc in [("RSI 14",f"{rsi_v:.1f}  {rsi_l}",rsi_c),("MACD",mac_l,mac_c),("EMA Trend",ema_l,ema_c),("ADX",f"{adx_v:.1f}  {adx_l}",adx_c),("Stochastic",f"{la['Stoch_k']:.1f}",""),("ATR",f"{la['ATR']:.2f}","")])
        st.markdown(f'<div class="card"><div class="card-hdr">Quick Stats</div>{qs}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    rr=abs((sr["Resistance (50D)"]-la["Close"])/(la["Close"]-sr["Support (50D)"])) if la["Close"]-sr["Support (50D)"]!=0 else 0
    if vtype=="buy":
        st.markdown(f'<div class="verd vb"><div class="vt vt-b">✅ BUY</div><div class="vbody"><b>{bc} of {sc_} indicators bullish</b> — weighted score <b>{bp}%</b>. Dominant trend is <b>upward</b>.</div><div class="v-chips"><div class="vchip"><span class="vc-l">Entry</span><span class="vc-v">{curr}{la["Close"]:.2f}</span></div><div class="vchip"><span class="vc-l">Stop-Loss</span><span class="vc-v">{curr}{sr["Support (50D)"]:.2f}</span></div><div class="vchip"><span class="vc-l">Target</span><span class="vc-v">{curr}{sr["Resistance (50D)"]:.2f}</span></div><div class="vchip"><span class="vc-l">Risk/Reward</span><span class="vc-v">{rr:.1f}×</span></div></div></div>', unsafe_allow_html=True)
    elif vtype=="sell":
        st.markdown(f'<div class="verd vs"><div class="vt vt-s">🔴 SELL / AVOID</div><div class="vbody"><b>{rc} of {sc_} indicators bearish</b> — weighted score <b>{rp}%</b>. Dominant trend is <b>downward</b>.</div><div class="v-chips"><div class="vchip"><span class="vc-l">Current Price</span><span class="vc-v">{curr}{la["Close"]:.2f}</span></div><div class="vchip"><span class="vc-l">Key Support</span><span class="vc-v">{curr}{sr["Support (50D)"]:.2f}</span></div><div class="vchip"><span class="vc-l">52W Low</span><span class="vc-v">{curr}{sr["52W Low"]:.2f}</span></div></div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verd vw"><div class="vt vt-w">⏳ WAIT & WATCH</div><div class="vbody">Mixed — <b>{bc} bullish</b> vs <b>{rc} bearish</b> out of {sc_} indicators. Wait for breakout above <b>{curr}{sr["Resistance (50D)"]:.2f}</b> or below <b>{curr}{sr["Support (50D)"]:.2f}</b>.</div></div>', unsafe_allow_html=True)

    # Why this analysis is right
    reasons = build_reasoning(sig_list, pat_list, vtype, la, bp, rp, sc_)
    r_html = "".join(f'<div class="reason-item"><span class="reason-icon">{ico}</span><div><span style="font-size:0.62rem;color:#444;text-transform:uppercase;letter-spacing:0.8px;display:block;margin-bottom:2px">{cat}</span><span>{txt}</span></div></div>' for ico,cat,txt in reasons)
    st.markdown(f'<div class="reason-box"><div class="reason-title">Why this analysis is right — Key reasoning</div>{r_html}</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer"><strong>⚠️ DISCLAIMER — For Analysis Only. Invest at Your Own Risk.</strong><br>Technical analysis is based on historical price data. It does NOT guarantee future results. Always consult a SEBI-registered advisor before investing.</div>', unsafe_allow_html=True)
    return df, info

# ════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown('<div class="sec-label">Search — Stock / Index / Commodity</div>', unsafe_allow_html=True)
    s1,s2,s3 = st.columns([3,1,1])
    with s1:
        chosen = st.selectbox("Select",[" — Select a stock, index or commodity —"]+ALL_LABELS,index=0,label_visibility="collapsed")
    with s2:
        period = st.selectbox("Period",["1d","5d","1mo","3mo","6mo","1y","2y"],index=4,label_visibility="collapsed")
    with s3:
        interval = st.selectbox("Interval",["15m","30m","1h","4h","1d","1wk"],index=4,label_visibility="collapsed")
    go_btn = st.button("⚡  Run Analysis")

    if go_btn:
        if chosen.startswith(" —"):
            st.warning("Please select a stock from the dropdown.")
        else:
            sel_name,sel_ticker,sel_exch=None,None,None
            for n,t,e in SYMBOL_DB:
                if f"{n} ({t})"==chosen: sel_name,sel_ticker,sel_exch=n,t,e; break
            if sel_ticker:
                with st.spinner(f"Analysing {sel_name} ..."):
                    try:
                        render_analysis(sel_name,sel_ticker,sel_exch,period,interval)
                    except Exception as ex:
                        st.error(f"Analysis failed: {str(ex)}")
                        if "15m" in interval or "30m" in interval or "1h" in interval or "4h" in interval:
                            st.info("ℹ️ Intraday intervals (15m/30m/1h/4h) only work with short periods. Try: 15m→'5d', 30m→'1mo', 1h→'1mo', 4h→'3mo'")
    else:
        st.markdown("""<div style="text-align:center;padding:3rem 1rem 2rem">
          <div style="font-size:0.65rem;color:#2A2A2A;letter-spacing:3px;text-transform:uppercase;margin-bottom:0.8rem">Professional Terminal</div>
          <div style="font-size:2rem;font-weight:800;color:#FFF;letter-spacing:-0.6px;margin-bottom:0.6rem">What do you want to analyse?</div>
          <p style="font-size:0.92rem;color:#666;line-height:1.8;max-width:500px;margin:0 auto">
            Select any stock, index, or commodity from the dropdown above.<br>Includes 200+ NSE stocks, mid-caps, small-caps, commodities and indices.
          </p></div>""", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        feats=[("01","200+ Symbols","NSE large/mid/small caps, BSE, Godawari Power, Suzlon, RVNL, HAL, Defence stocks, and more."),
               ("02","10+ Indicators","RSI, MACD, EMA, Bollinger Bands, Stochastic, ADX, Volume — fully automated."),
               ("03","Why Analysis is Right","Every result includes a plain-English reasoning section explaining the top factors."),
               ("04","Fundamental Analysis","P/E, ROE, Debt, Growth, Margins — fundamental verdict from official data."),
               ("05","Multi-Timeframe","15m, 30m, 1h, 4h, 1d, 1w — for every trading style."),
               ("06","Trade Planner","Full trade plan: entry, stop-loss, targets, risk/reward, rules, and more.")]
        for i,(num,title,desc) in enumerate(feats):
            col=[c1,c2,c3][i%3]
            col.markdown(f'<div class="feat-card"><div class="feat-num">{num}</div><div class="feat-title">{title}</div><div class="feat-desc">{desc}</div></div>', unsafe_allow_html=True)
            if i==2: st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: FUNDAMENTAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Fundamental Analysis":
    st.markdown("""<div style="background:#0A0A0A;border:1px solid #1E1E1E;border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1.2rem">
      <div style="font-size:0.95rem;font-weight:700;color:#FFF;margin-bottom:0.3rem">◉ Fundamental Analysis</div>
      <div style="font-size:0.82rem;color:#888;line-height:1.6">
        Analyses valuation, profitability, debt, growth, and analyst targets using official data from exchanges,
        company filings, and verified financial databases. NSE/BSE stocks only — commodities and indices are excluded.
      </div></div>""", unsafe_allow_html=True)

    f1,f2 = st.columns([3,1])
    with f1:
        f_chosen = st.selectbox("Select Stock",[" — Select a stock —"]+[f"{n} ({t})" for n,t,e in SYMBOL_DB if e in ["NSE","BSE"]],index=0,label_visibility="collapsed")
    with f2:
        fa_btn = st.button("◉  Run Fundamental Analysis", use_container_width=True)

    if fa_btn:
        if f_chosen.startswith(" —"):
            st.warning("Please select a stock.")
        else:
            sel_name,sel_ticker,sel_exch=None,None,None
            for n,t,e in SYMBOL_DB:
                if f"{n} ({t})"==f_chosen: sel_name,sel_ticker,sel_exch=n,t,e; break
            if sel_ticker:
                with st.spinner(f"Loading fundamental data for {sel_name} from official sources..."):
                    try:
                        t_obj = yf.Ticker(sel_ticker)
                        info  = t_obj.info
                        if not info or len(info)<5:
                            st.error("Could not retrieve fundamental data. This may be a mid/small-cap stock with limited coverage.")
                            st.stop()

                        fund_data,pb,bb,pnb,fverd,ftype,sector,industry,rec,tgt,price = fundamental_analysis(sel_ticker,info)
                        curr = "₹" if ".NS" in sel_ticker or ".BO" in sel_ticker else ""
                        name = info.get("longName",sel_name)
                        mktcap = info.get("marketCap",0)

                        # Header
                        st.markdown(f"""<div class="tk-wrap">
                          <div><div class="tk-name">{name}</div><div class="tk-sym">{sel_ticker} · Fundamental</div></div>
                          <div style="display:flex;gap:6px;flex-wrap:wrap;margin-left:auto">
                            {"".join(f'<span class="sector-badge">{x}</span>' for x in [sector,industry] if x)}
                          </div>
                        </div>""", unsafe_allow_html=True)

                        # Grid
                        keys = list(fund_data.keys()); half = len(keys)//2+len(keys)%2
                        g1,g2 = keys[:half], keys[half:]
                        row1 = "".join(f'<div class="fund-cell"><div class="fund-cell-label">{k}</div><div class="fund-cell-value">{fund_data[k][0]}</div><div class="fund-cell-sub">{fund_data[k][1]}</div></div>' for k in g1)
                        row2 = "".join(f'<div class="fund-cell"><div class="fund-cell-label">{k}</div><div class="fund-cell-value">{fund_data[k][0]}</div><div class="fund-cell-sub">{fund_data[k][1]}</div></div>' for k in g2)
                        st.markdown(f'<div style="background:#0A0A0A;border:1px solid #1E1E1E;border-radius:14px;padding:1.1rem;margin-bottom:1rem"><div class="fund-grid">{row1+row2}</div></div>', unsafe_allow_html=True)

                        # Industry/market trend note
                        sector_notes = {
                            "Technology":      "Indian IT sector faces margin pressure from wage inflation and weak US IT spending. Cloud, AI projects remain growth pockets.",
                            "Financial Services":"RBI maintaining rates. Credit growth stable at 14–16%. NBFCs seeing strong disbursals. Watch NPA trends.",
                            "Consumer Defensive":"FMCG seeing rural recovery. Urban premium segment growing. Input costs stabilising.",
                            "Healthcare":      "Pharma exports to US recovering. Domestic formulations growing steadily. API pricing normalised.",
                            "Energy":          "Energy transition underway. Renewables getting govt support. Traditional energy still strong on volume.",
                            "Basic Materials": "Steel, metals cyclical. Infra push boosting volumes. China slowdown headwind for pricing.",
                            "Industrials":     "Strong order books. Defence indigenisation, PLI schemes driving growth. Railway capex surging.",
                            "Consumer Cyclical":"Auto sector strong. 2-wheelers recovering. EV transition creating near-term uncertainty.",
                            "Real Estate":     "Residential demand strong in tier-1. Office absorption recovering. Interest rate sensitivity key.",
                            "Utilities":       "Power demand growing. Renewable capacity addition accelerating. Regulated returns stable.",
                            "Communication Services":"Telecom ARPU improving. OTT growth slowing. 5G capex cycle peaking.",
                        }
                        snote = sector_notes.get(sector,"Sector trend data not available for this industry. Monitor quarterly results and management commentary.")

                        # Market context
                        try:
                            nifty = yf.Ticker("^NSEI").history(period="1mo",interval="1d")
                            if not nifty.empty:
                                nifty_chg = (nifty["Close"].iloc[-1]-nifty["Close"].iloc[0])/nifty["Close"].iloc[0]*100
                                market_str = f"Nifty 50 is {'up' if nifty_chg>0 else 'down'} {abs(nifty_chg):.1f}% in the past month — {'broad market is supportive' if nifty_chg>0 else 'broad market headwind'}."
                            else: market_str = "Market context unavailable."
                        except: market_str = "Market context unavailable."

                        st.markdown(f"""<div style="background:#0A0A0A;border:1px solid #1E1E1E;border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1rem">
                          <div style="font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#555;margin-bottom:0.7rem">Industry & Market Context</div>
                          <div style="font-size:0.84rem;color:#AAA;line-height:1.7;margin-bottom:0.6rem">
                            <span style="color:#666;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.8px">Sector: </span>{sector or "N/A"} &nbsp;·&nbsp;
                            <span style="color:#666;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.8px">Industry: </span>{industry or "N/A"}
                          </div>
                          <div style="font-size:0.84rem;color:#AAA;line-height:1.7;margin-bottom:0.6rem">{snote}</div>
                          <div style="font-size:0.82rem;color:#666;line-height:1.6;padding-top:0.6rem;border-top:1px solid #1A1A1A">{market_str}</div>
                        </div>""", unsafe_allow_html=True)

                        # Analyst rec
                        if rec:
                            rec_map={"buy":"Strong Buy","strong_buy":"Strong Buy","hold":"Hold","sell":"Sell","underperform":"Underperform","outperform":"Outperform"}
                            rec_label=rec_map.get(rec.lower(),rec.title())
                            rec_col="#4ADE80" if "buy" in rec.lower() else ("#F87171" if "sell" in rec.lower() or "under" in rec.lower() else "#FBBF24")
                            tgt_str=f"  ·  Consensus Target: {curr}{tgt:.0f}" if tgt else ""
                            st.markdown(f'<div style="background:#141414;border:1px solid #1E1E1E;border-radius:10px;padding:0.7rem 1rem;margin-bottom:1rem;display:flex;align-items:center;gap:12px"><div style="font-size:0.62rem;color:#555;text-transform:uppercase;letter-spacing:1px">Analyst Consensus</div><div style="font-size:0.9rem;font-weight:700;color:{rec_col}">{rec_label}{tgt_str}</div></div>', unsafe_allow_html=True)

                        # Fundamental verdict
                        fvc_class={"bull":"fv-bull","bear":"fv-bear","neu":"fv-neu"}.get(ftype,"fv-neu")
                        ftc_class={"bull":"fv-tb","bear":"fv-ts","neu":"fv-tw"}.get(ftype,"fv-tw")
                        all_points = [("▲ Bullish",p,"#4ADE80") for p in pb]+[("▼ Bearish",p,"#F87171") for p in bb]+[("◐ Neutral",p,"#FBBF24") for p in pnb]
                        pts_html="".join(f'<div class="fv-point"><span style="font-size:0.72rem;color:{c};flex-shrink:0;min-width:72px">{l}</span><span>{txt}</span></div>' for l,txt,c in all_points)
                        st.markdown(f"""<div class="fund-verdict {fvc_class}">
                          <div class="fv-title {ftc_class}">{fverd}</div>
                          <div style="font-size:0.62rem;color:#555;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:0.9rem">
                            {len(pb)} bullish factors · {len(bb)} bearish factors · {len(pnb)} neutral factors — based on official financial data
                          </div>
                          {pts_html}
                        </div>""", unsafe_allow_html=True)

                        st.markdown('<div class="disclaimer"><strong>⚠️ Fundamental data sourced from Yahoo Finance (NSE/BSE filings, company reports, analyst consensus).</strong> Data may have a lag of up to 24 hours. Always verify with the BSE/NSE official website or SEBI filings. This is NOT investment advice.</div>', unsafe_allow_html=True)

                    except Exception as ex:
                        st.error(f"Fundamental analysis failed: {str(ex)}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: TRADE PLANNER
# ════════════════════════════════════════════════════════════════════════════
elif page == "Trade Planner":
    st.markdown('<div style="background:#0A0A0A;border:1px solid #1E1E1E;border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1.2rem"><div style="font-size:0.95rem;font-weight:700;color:#FFF;margin-bottom:0.3rem">◎ Trade Planner</div><div style="font-size:0.82rem;color:#888;line-height:1.6">Build a personalised trade plan. Add stocks, define your risk profile, trade duration, and position type. Get entry, stop-loss, targets, and trade rules.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Step 1 — Add to Watchlist</div>', unsafe_allow_html=True)
    wa1,wa2 = st.columns([3,1])
    with wa1:
        wl_chosen=st.selectbox("Watchlist Stock",[" — Select stock —"]+ALL_LABELS,index=0,label_visibility="collapsed")
    with wa2:
        if st.button("+ Add", use_container_width=True):
            if not wl_chosen.startswith(" —"):
                for n,t,e in SYMBOL_DB:
                    if f"{n} ({t})"==wl_chosen:
                        if not any(w["ticker"]==t for w in st.session_state["watchlist"]):
                            st.session_state["watchlist"].append({"name":n,"ticker":t,"exch":e})
                            st.success(f"Added {n}")
                        else: st.info(f"{n} already in watchlist")
                        break
    if not st.session_state["watchlist"]:
        st.markdown('<div style="background:#0A0A0A;border:1px dashed #1E1E1E;border-radius:12px;padding:1.5rem;text-align:center;color:#333;font-size:0.85rem">Watchlist empty. Add stocks above.</div>', unsafe_allow_html=True)
        st.stop()
    for idx,w in enumerate(st.session_state["watchlist"]):
        wc1,wc2=st.columns([5,1])
        with wc1: st.markdown(f'<div style="background:#111;border:1px solid #1E1E1E;border-radius:12px;padding:0.8rem 1.1rem;display:flex;align-items:center;gap:12px;margin-bottom:6px"><div class="wl-icon">{w["name"][:2].upper()}</div><div><div style="font-size:0.9rem;font-weight:600;color:#FFF">{w["name"]}</div><div style="font-size:0.7rem;color:#555;font-family:monospace">{w["ticker"]} · {w["exch"]}</div></div></div>', unsafe_allow_html=True)
        with wc2:
            if st.button("Remove", key=f"rm_{idx}"): st.session_state["watchlist"].pop(idx); st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Step 2 — Define Trade Setup</div>', unsafe_allow_html=True)
    wl_names=[f"{w['name']} ({w['ticker']})" for w in st.session_state["watchlist"]]
    plan_stock=st.selectbox("Plan for",wl_names,label_visibility="collapsed")
    plan_entry=next((w for w in st.session_state["watchlist"] if f"{w['name']} ({w['ticker']})"==plan_stock),None)
    tp1,tp2,tp3=st.columns(3)
    with tp1:
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.5rem">Risk Appetite</div>', unsafe_allow_html=True)
        risk=st.radio("Risk",["Conservative","Moderate","Aggressive"],key="risk_ap",label_visibility="collapsed")
    with tp2:
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.5rem">Trade Duration</div>', unsafe_allow_html=True)
        duration=st.radio("Duration",["Intraday (same day)","Swing (2–10 days)","Positional (1–3 months)","Long Term (6m+)"],key="dur",label_visibility="collapsed")
    with tp3:
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.5rem">Position Type</div>', unsafe_allow_html=True)
        pos_type=st.radio("Position",["Long (Buy)","Short (Sell)"],key="pos",label_visibility="collapsed")
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.5rem;margin-top:0.8rem">Stop-Loss Style</div>', unsafe_allow_html=True)
        sl_type=st.radio("SL",["Pre-defined (fixed)","Trailing Stop-Loss","Open Position (no SL)"],key="sl",label_visibility="collapsed")
    capital_input=st.number_input("Capital to deploy (₹)",min_value=1000,max_value=10000000,value=100000,step=5000)
    if st.button("📋  Generate Trade Plan",use_container_width=True) and plan_entry:
        with st.spinner(f"Building plan for {plan_entry['name']} ..."):
            try:
                dur_map={"Intraday (same day)":("5d","15m"),"Swing (2–10 days)":("3mo","1d"),"Positional (1–3 months)":("1y","1d"),"Long Term (6m+)":("2y","1wk")}
                pp,pi=dur_map.get(duration,("1y","1d"))
                t_obj=yf.Ticker(plan_entry["ticker"])
                df=t_obj.history(period=pp,interval=pi)
                if df.empty: st.error("No data."); st.stop()
                df=compute(df)
                sig_list=get_signals(df);pat_list=get_patterns(df)
                bs,rs,tt,bc,rc,sc_=score(sig_list);verd_,vtype=verdict(bs,rs,tt)
                la=df.iloc[-1];curr="₹" if ".NS" in plan_entry["ticker"] else ""
                price=la["Close"];atr=la["ATR"]
                bp=int(bs/tt*100) if tt else 0;rp=int(rs/tt*100) if tt else 0
                rp_vals={"Conservative":{"sl_mult":1.0,"t1":1.5,"t2":2.5,"t3":4.0,"pos_pct":0.03},"Moderate":{"sl_mult":1.5,"t1":2.0,"t2":3.5,"t3":6.0,"pos_pct":0.05},"Aggressive":{"sl_mult":2.0,"t1":2.5,"t2":4.5,"t3":8.0,"pos_pct":0.08}}[risk]
                is_long=pos_type=="Long (Buy)"
                sl_atr=atr*rp_vals["sl_mult"]
                if is_long: sl=price-sl_atr;t1=price+atr*rp_vals["t1"];t2=price+atr*rp_vals["t2"];t3=price+atr*rp_vals["t3"]
                else: sl=price+sl_atr;t1=price-atr*rp_vals["t1"];t2=price-atr*rp_vals["t2"];t3=price-atr*rp_vals["t3"]
                sl=max(sl,0.01)
                rpx=abs(price-sl)
                rr1=abs(t1-price)/rpx if rpx>0 else 0;rr2=abs(t2-price)/rpx if rpx>0 else 0;rr3=abs(t3-price)/rpx if rpx>0 else 0
                qty=max(1,int(capital_input*rp_vals["pos_pct"]/rpx)) if rpx>0 else 1
                exp=qty*price;ml=qty*rpx;mg=qty*abs(t3-price)
                if is_long and bp>=55: talign=(f"✅ Trend aligned — {bp}% of indicators BULLISH","#4ADE80")
                elif is_long and bp<45: talign=(f"⚠️ Counter-trend — {rp}% indicators BEARISH. Trade with caution.","#F87171")
                elif not is_long and rp>=55: talign=(f"✅ Trend aligned — {rp}% indicators BEARISH","#4ADE80")
                elif not is_long and rp<45: talign=(f"⚠️ Counter-trend — {bp}% indicators BULLISH. Trade with caution.","#F87171")
                else: talign=("◐ Mixed signals — use strict risk management","#FBBF24")
                vbd = {"buy":"tb-buy","sell":"tb-sell","wait":"tb-wait"}.get(vtype,"tb-wait")
                rb  = {"Conservative":"rb-low","Moderate":"rb-med","Aggressive":"rb-high"}.get(risk,"rb-med")
                pos_label = "LONG" if is_long else "SHORT"
                stock_name = plan_entry['name']
                talign_color = talign[1]
                talign_text  = talign[0]

                # Pre-build metrics grid
                metrics_data = [
                    ("Entry Price",      f"{curr}{price:.2f}",           "Current market price"),
                    ("Stop-Loss",        f"{curr}{sl:.2f}",              f"{abs(((sl-price)/price)*100):.1f}% away"),
                    ("Quantity",         f"{qty} shares",                 "Based on risk capital"),
                    ("Total Exposure",   f"{curr}{exp:,.0f}",            f"{exp/capital_input*100:.1f}% of capital"),
                    ("Max Loss",         f"{curr}{ml:,.0f}",             f"{ml/capital_input*100:.1f}% of capital"),
                    ("Max Gain (T3)",    f"{curr}{mg:,.0f}",              "At full Target 3"),
                ]
                metrics_html = "".join(
                    f'<div class="tp-cell"><div class="tp-cell-label">{l}</div>'
                    f'<div class="tp-cell-value">{v}</div>'
                    f'<div class="tp-cell-sub">{s}</div></div>'
                    for l,v,s in metrics_data
                )

                # Pre-build targets
                def dir_sign(tgt_price):
                    if (is_long and tgt_price > price) or (not is_long and tgt_price < price):
                        return "+"
                    return "-"

                targets_data = [
                    ("Target 1", t1,  rr1, "Book 50%",     "#4ADE80"),
                    ("Target 2", t2,  rr2, "Book 30%",     "#22C55E"),
                    ("Target 3", t3,  rr3, "Let 20% run",  "#16A34A"),
                    ("Stop-Loss", sl, 0,   "Exit all",     "#F87171"),
                ]
                targets_html = ""
                for tl, tp_, tr_, tn_, tc_ in targets_data:
                    pct_ = abs(((tp_ - price) / price) * 100)
                    sign_ = dir_sign(tp_)
                    rr_str_ = f"  R/R {tr_:.1f}x" if tr_ > 0 else ""
                    targets_html += (
                        f'<div class="tgt-row">'
                        f'<div class="tgt-dot" style="background:{tc_}"></div>'
                        f'<div style="font-size:0.78rem;color:#888;min-width:60px">{tl}</div>'
                        f'<div style="font-size:0.9rem;font-weight:700;color:#FFF;font-family:monospace">{curr}{tp_:.2f}</div>'
                        f'<div style="font-size:0.75rem;color:{tc_};font-weight:600;margin-left:6px">{sign_}{pct_:.1f}%{rr_str_}</div>'
                        f'<div style="flex:1"></div>'
                        f'<div style="font-size:0.72rem;color:#555">{tn_}</div>'
                        f'</div>'
                    )

                # Pre-build rules
                rules_map = {
                    "Conservative": [
                        "Enter only after candle close confirms direction",
                        f"Hard stop-loss at {curr}{sl:.2f} — do NOT widen it",
                        "Book 50% at T1, trail SL to breakeven",
                        "Book 30% at T2, let 20% run to T3",
                        "Never risk more than 2-3% of total capital per trade",
                        "Avoid trading near major news or earnings events",
                    ],
                    "Moderate": [
                        "Enter on RSI + MACD + price action alignment",
                        f"Initial stop-loss {curr}{sl:.2f} — trail after T1 is hit",
                        "Take 40% off at T1, move SL to cost price",
                        "Take 30% at T2, ride remaining 30% to T3",
                        "5% capital deployment per trade maximum",
                        "Review position daily — exit if thesis breaks",
                    ],
                    "Aggressive": [
                        "Enter on breakout above resistance or below support",
                        f"Wide stop at {curr}{sl:.2f} — built to handle volatility",
                        "Scale out 25% at T1, hold strong for T2 and T3",
                        "Pyramid into winners only — never average into losers",
                        "Maximum 8% capital per single trade",
                        "Use GTT or bracket orders to automate exits",
                    ],
                }
                rules_list = rules_map.get(risk, [])
                rules_html = "".join(
                    f'<div class="rule-item"><span class="rule-dot">→</span><span>{r}</span></div>'
                    for r in rules_list
                )

                # Pre-build SL method note
                sl_notes = {
                    "Pre-defined (fixed)":    f"Fixed SL at {curr}{sl:.2f}. Place a GTT / stop-market order immediately at entry.",
                    "Trailing Stop-Loss":      f"Start trailing SL after T1. Trail by 1x ATR ({curr}{atr:.2f}) below each higher close.",
                    "Open Position (no SL)":   "No stop-loss chosen. This is high risk. Monitor continuously and exit manually on reversal.",
                }
                sl_note_text = sl_notes.get(sl_type, "")

                # Render the complete plan
                st.markdown(f"""
                <div style="background:#0A0A0A;border:1px solid #1E1E1E;border-radius:16px;padding:1.4rem 1.6rem">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;padding-bottom:1rem;border-bottom:1px solid #1C1C1C">
                    <span class="tp-badge {vbd}">{verd_}</span>
                    <span class="tp-badge" style="background:#141414;color:#AAA;border:1px solid #2A2A2A">{pos_label}</span>
                    <span class="risk-badge {rb}">{risk}</span>
                    <div style="margin-left:auto">
                      <div style="font-size:1.05rem;font-weight:700;color:#FFF">{stock_name}</div>
                    </div>
                  </div>
                  <div style="background:#141414;border-radius:10px;padding:0.65rem 1rem;margin-bottom:1rem;font-size:0.84rem;color:{talign_color};font-weight:500">
                    {talign_text}
                  </div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:1rem">
                    {metrics_html}
                  </div>
                  <div style="font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#555;margin-bottom:0.6rem">
                    Price Targets
                  </div>
                  {targets_html}
                  <div style="font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#555;margin-top:1rem;margin-bottom:0.6rem">
                    Trade Rules
                  </div>
                  {rules_html}
                  <div style="background:#141414;border-radius:10px;padding:0.65rem 1rem;margin-top:1rem;font-size:0.82rem;color:#AAA">
                    <span style="font-size:0.6rem;color:#555;text-transform:uppercase;letter-spacing:0.8px;display:block;margin-bottom:3px">
                      Stop-Loss Method
                    </span>
                    {sl_note_text}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="disclaimer"><strong>Warning: Trade plan generated using technical and mathematical models only. This is NOT financial advice. Invest at your own risk.</strong></div>', unsafe_allow_html=True)
            except Exception as ex:
                st.error(f"Trade plan failed: {str(ex)}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: SEARCH HISTORY
# ════════════════════════════════════════════════════════════════════════════
elif page == "Search History":
    history=st.session_state["history"]
    if not history:
        st.markdown('<div style="text-align:center;padding:5rem 1rem"><div style="font-size:1.5rem;font-weight:700;color:#FFF;margin-bottom:0.5rem">No history yet</div><div style="font-size:0.88rem;color:#555">Run your first analysis from the Dashboard</div></div>', unsafe_allow_html=True)
    else:
        hc1,hc2=st.columns([3,1])
        with hc1: st.markdown(f'<div style="font-size:0.84rem;color:#555;margin-bottom:0.8rem">{len(history)} analyses this session</div>', unsafe_allow_html=True)
        with hc2:
            if st.button("Clear All",use_container_width=True): st.session_state["history"]=[]; st.rerun()
        buys=sum(1 for h in history if h["verdict"]=="BUY");sells=sum(1 for h in history if h["verdict"]=="SELL");waits=sum(1 for h in history if h["verdict"]=="WAIT")
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Total",len(history));m2.metric("BUY",buys);m3.metric("SELL",sells);m4.metric("WAIT",waits)
        st.markdown("<hr>", unsafe_allow_html=True)
        for h in reversed(history):
            vc={"BUY":"#4ADE80","SELL":"#F87171"}.get(h["verdict"],"#FBBF24")
            vbg={"BUY":"#0A2010","SELL":"#200A0A"}.get(h["verdict"],"#201A0A")
            vbd={"BUY":"#1A4020","SELL":"#401A1A"}.get(h["verdict"],"#402A1A")
            cc="#4ADE80" if "+" in h["chg"] else "#F87171"
            st.markdown(f'<div class="hist-item"><div style="width:44px;height:44px;border-radius:10px;background:{vbg};border:1px solid {vbd};color:{vc};display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:800;flex-shrink:0">{h["verdict"]}</div><div style="flex:1;min-width:120px"><div style="font-size:0.9rem;font-weight:600;color:#FFF">{h["name"]}</div><div style="font-size:0.72rem;color:#555;font-family:monospace">{h["ticker"]} · {h["period"]} · {h["time"]}</div></div><div style="text-align:right"><div style="font-size:0.95rem;font-weight:700;color:#FFF">{h["price"]}</div><div style="font-size:0.78rem;color:{cc};font-weight:600">{h["chg"]}</div></div><div style="text-align:right;min-width:90px"><div style="font-size:0.6rem;color:#333;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px">Bull / Bear</div><div style="font-size:0.82rem;font-weight:700"><span style="color:#4ADE80">{h["bp"]}%</span> <span style="color:#333">/</span> <span style="color:#F87171">{h["rp"]}%</span></div></div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: TEAM & SHARING
# ════════════════════════════════════════════════════════════════════════════
elif page == "Team & Sharing":
    st.markdown('<div style="background:#0A0A0A;border:1px solid #1E1E1E;border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1.2rem"><div style="font-size:0.95rem;font-weight:700;color:#FFF;margin-bottom:0.3rem">⊞ Team & Sharing</div><div style="font-size:0.82rem;color:#888;line-height:1.6">Share Ace-Trade with your team. Each member gets their own login with email and username access. To add new team members, edit the USERS dictionary at the top of app.py.</div></div>', unsafe_allow_html=True)

    # Current team
    st.markdown('<div class="sec-label">Current Team Members</div>', unsafe_allow_html=True)
    for uname, udata in USERS.items():
        role_cls = "role-owner" if udata["role"]=="Owner" else "role-analyst"
        st.markdown(f"""<div class="team-card">
          <div class="team-avatar">{udata["name"][:2].upper()}</div>
          <div style="flex:1">
            <div style="font-size:0.9rem;font-weight:600;color:#FFF">{udata["name"]}</div>
            <div style="font-size:0.72rem;color:#555;font-family:monospace">{udata["email"]}</div>
          </div>
          <span style="font-size:0.7rem;color:#555;font-family:monospace">@{uname}</span>
          <span class="role-badge {role_cls}">{udata["role"]}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">How to Add a New Team Member</div>', unsafe_allow_html=True)
    st.markdown("""<div style="background:#0A0A0A;border:1px solid #1E1E1E;border-radius:14px;padding:1.3rem 1.5rem">
      <div style="font-size:0.85rem;color:#AAA;line-height:1.8;margin-bottom:1rem">
        Open your <code style="background:#141414;padding:2px 6px;border-radius:4px;color:#FFF;font-family:monospace">app.py</code> file and find the <code style="background:#141414;padding:2px 6px;border-radius:4px;color:#FFF;font-family:monospace">USERS</code> dictionary near the top. Add a new entry:
      </div>
      <div style="background:#141414;border-radius:10px;padding:1rem 1.2rem;font-family:monospace;font-size:0.82rem;color:#4ADE80;line-height:1.8">
        "newuser": &#123;<br>
        &nbsp;&nbsp;"password": "theirpassword",<br>
        &nbsp;&nbsp;"email": "newuser@yourdomain.com",<br>
        &nbsp;&nbsp;"name": "Their Name",<br>
        &nbsp;&nbsp;"role": "Analyst"<br>
        &#125;
      </div>
      <div style="font-size:0.82rem;color:#666;margin-top:1rem;line-height:1.7">
        After saving, restart the app (<code style="background:#141414;padding:2px 6px;border-radius:4px;color:#AAA;font-family:monospace">Ctrl+C</code> then <code style="background:#141414;padding:2px 6px;border-radius:4px;color:#AAA;font-family:monospace">streamlit run app.py</code>).
        Your teammate can then log in with either their <strong style="color:#CCC">username</strong> or their <strong style="color:#CCC">email address</strong>.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">How to Share the Tool</div>', unsafe_allow_html=True)
    st.markdown("""<div style="background:#0A0A0A;border:1px solid #1E1E1E;border-radius:14px;padding:1.3rem 1.5rem">
      <div style="font-size:0.85rem;color:#AAA;line-height:2">
        <div style="display:flex;gap:12px;padding:0.4rem 0;border-bottom:1px solid #141414"><span style="color:#FFF;font-weight:600;min-width:180px">On the same WiFi/network:</span><span>Your teammate opens their browser and goes to <code style="background:#141414;padding:2px 6px;border-radius:4px;color:#4ADE80;font-family:monospace">http://YOUR-PC-IP:8501</code></span></div>
        <div style="display:flex;gap:12px;padding:0.4rem 0;border-bottom:1px solid #141414"><span style="color:#FFF;font-weight:600;min-width:180px">Find your IP:</span><span>Open Command Prompt → type <code style="background:#141414;padding:2px 6px;border-radius:4px;color:#4ADE80;font-family:monospace">ipconfig</code> → look for IPv4 Address</span></div>
        <div style="display:flex;gap:12px;padding:0.4rem 0;border-bottom:1px solid #141414"><span style="color:#FFF;font-weight:600;min-width:180px">Deploy online (free):</span><span>Upload to <code style="background:#141414;padding:2px 6px;border-radius:4px;color:#4ADE80;font-family:monospace">share.streamlit.io</code> — free hosting, accessible from anywhere</span></div>
        <div style="display:flex;gap:12px;padding:0.4rem 0"><span style="color:#FFF;font-weight:600;min-width:180px">Mobile access:</span><span>Once online, the tool works on mobile browsers — no app install needed</span></div>
      </div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown(f"""<div style="max-width:700px">
      <div style="margin-bottom:1.5rem">{logo("lg")}</div>
      <p style="font-size:0.92rem;color:#888;line-height:1.8;margin-bottom:1.4rem">
        Ace-Trade is a professional technical and fundamental analysis terminal for serious traders.
        200+ NSE/BSE stocks, commodities, and indices. 10+ indicators. Complete trade planning.
      </p>
      <div style="background:#111;border:1px solid #1E1E1E;border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem">
        <div style="font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#444;margin-bottom:0.9rem">What this tool covers</div>
        {"".join(f'<div style="display:flex;gap:10px;padding:0.45rem 0;border-bottom:1px solid #1A1A1A;font-size:0.86rem"><span style="color:#FFF;min-width:16px">→</span><span style="color:#888">{t}</span></div>' for t in ["Technical — EMA 20/50/200, Golden Cross, Death Cross, RSI, MACD, Stochastic, Bollinger Bands, ATR, ADX, Volume","Fundamental — P/E, P/B, ROE, Debt/Equity, Revenue Growth, Earnings Growth, Free Cash Flow, Analyst Targets","Chart Patterns — Double Top/Bottom, Triangles, Breakouts, Rally/Drop Streaks","Trade Planner — Entry, Stop-Loss (fixed/trailing), 3 Targets, Risk/Reward, Position Sizing","Analysis Reasoning — Plain-English explanation of why the verdict is what it is","Team Access — Multi-user login with username and email authentication"])}
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div class="disclaimer" style="max-width:700px;font-size:0.84rem">
      <strong>⚠️ FULL DISCLAIMER</strong><br><br>
      Ace-Trade is for <strong>educational and analysis purposes only</strong>. It is NOT a SEBI-registered investment advisor.<br><br>
      <strong>Invest at Your Own Risk.</strong> Technical and fundamental signals are based on historical data and mathematical models.
      They do not predict the future. Always consult a <strong>SEBI-registered financial advisor</strong> before investing.
      The creators of Ace-Trade accept <strong>no liability</strong> for any financial losses.
      <strong>You are solely responsible for all investment decisions.</strong>
    </div>""", unsafe_allow_html=True)
