# Inspiration - Inspiracija
## https://github.com/bicycle-orooro/my-lotto-ai/tree/main



"""
cd /
streamlit run app.py
"""


 
import pandas as pd
import streamlit as st
import random
import os
import logging
import streamlit.components.v1 as components 
import altair as alt 
import re 
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
import numpy as np
from PIL import Image

CSV_FILE = "/data/loto7hh_4596_k29_up.csv"
SEED = 39
random.seed(SEED)
np.random.seed(SEED)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
MAX_NUM = 39
PICK_COUNT = 7

# ==========================================
# 0. Page basic settings
# ==========================================
st.set_page_config(page_title="AI Loto prediktor", page_icon="🎲")

# ==========================================
# 📈 Google Analytics
# ==========================================
def inject_ga(tracking_id):
    if tracking_id == "G-XXXXXXXXXX":
        return
        
    ga_js = f"""
    <script>
        var parentDoc = window.parent.document;
        if (!parentDoc.getElementById("ga-script")) {{
            var gaScript = parentDoc.createElement("script");
            gaScript.id = "ga-script";
            gaScript.async = true;
            gaScript.src = "https://www.googletagmanager.com/gtag/js?id={tracking_id}";
            parentDoc.head.appendChild(gaScript);

            var gaInline = parentDoc.createElement("script");
            gaInline.innerHTML = `
                window.dataLayer = window.dataLayer || [];
                function gtag(){{dataLayer.push(arguments);}}
                gtag('js', new Date());
                gtag('config', '{tracking_id}');
            `;
            parentDoc.head.appendChild(gaInline);
        }}
    </script>
    """
    components.html(ga_js, width=0, height=0)

# ==========================================
# 🎨 Smartphon app UI/UX optimization (Custom CSS)
# ==========================================
def apply_mobile_layout():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 450px !important;
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
        }
        
        h1 { font-size: 22px !important; font-weight: 800 !important; }
        h2 { font-size: 18px !important; font-weight: 700 !important; }
        h3 { font-size: 15px !important; font-weight: 600 !important; opacity: 0.8; }
        
        .lotto-ball {
            display: inline-block;
            width: 36px;       
            height: 36px;
            border-radius: 50%;
            text-align: center;
            line-height: 36px;
            font-size: 14px;   
            font-weight: bold;
            margin-right: 4px; 
            margin-bottom: 5px;
            box-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        [data-testid="stMetricValue"] { font-size: 20px !important; font-weight: 700 !important; }
        [data-testid="stMetricLabel"] { font-size: 12px !important; }
        
        .stButton>button {
            border-radius: 10px;
            font-weight: bold;
            font-size: 16px;
            padding: 10px;
        }
        
        button[data-baseweb="tab"] {
            font-size: 16px !important;
            font-weight: bold !important;
            padding-top: 10px !important;
            padding-bottom: 10px !important;
        }
        
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

def get_lotto_ball_html(number):
    if number <= 10: bg_color = "#fbc400"; text_color = "black"
    elif number <= 20: bg_color = "#69c8f2"; text_color = "black"
    elif number <= 30: bg_color = "#ff7272"; text_color = "white"
    elif number <= 40: bg_color = "#aaaaaa"; text_color = "white"
    else: bg_color = "#b0d840"; text_color = "black"
    return f'<div class="lotto-ball" style="background-color: {bg_color}; color: {text_color};">{number}</div>'

# ==========================================
# 📊 Data load
# ==========================================
@st.cache_data(ttl=600)
def load_data():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_FILE)
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        num_cols = []
        for i in range(1, 8):
            key = f"num{i}"
            if key in cols_lower:
                num_cols.append(cols_lower[key])
        if len(num_cols) == 7:
            out = df[num_cols].copy()
            out.columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7']
            out = out.apply(pd.to_numeric, errors='coerce').dropna().astype(int)
            out = out.iloc[::-1].reset_index(drop=True)
            out.insert(0, 'draw_no', np.arange(len(out), 0, -1, dtype=int))
            return out

        # 🎯 8번째 칸 'bonus' 추가
        df = pd.read_csv(CSV_FILE, header=None, names=['draw_no', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna().astype(int)
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except:
        return pd.DataFrame()

# ==========================================
# 🤖 AI based prediction algorithm
# ==========================================
def generate_ai_numbers(df, num_sets=1, fixed_nums=[], excluded_nums=[]):
    if df.empty: return [[1, 2, 3, 4, 5, 6, 7] for _ in range(num_sets)]
    all_nums = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7']].values.flatten()
    recent_nums = df.head(10)[['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7']].values.flatten()
    
    all_counts = pd.Series(all_nums).value_counts()
    recent_counts = pd.Series(recent_nums).value_counts()
    
    weights = {i: 1.0 for i in range(1, MAX_NUM + 1)}
    for num in range(1, MAX_NUM + 1):
        if num in all_counts: weights[num] += all_counts[num] * 0.02
        if num in recent_counts: weights[num] += recent_counts[num] * 0.5
        if num in excluded_nums: weights[num] = 0.0  
        if num in fixed_nums: weights[num] = 0.0  

    weight_list = [weights[i] for i in range(1, MAX_NUM + 1)]
    generated_sets = []
    for _ in range(num_sets):
        selected = fixed_nums.copy() 
        while len(selected) < PICK_COUNT:
            pick = random.choices(range(1, MAX_NUM + 1), weights=weight_list, k=1)[0]
            if pick not in selected: selected.append(pick)
        selected.sort() 
        generated_sets.append(selected)
    return generated_sets

# ==========================================
# 📊 Statistics analysis full set
# ==========================================
def show_statistics_dashboard(df):
    st.header("📊 Izvestaj analize loto podataka")
    if df.empty:
        st.warning("Nema podataka.")
        return

    recent = df.head(10)
    all_nums = recent[['num1','num2','num3','num4','num5','num6','num7']].values.flatten()
    counts = pd.Series(all_nums).value_counts().sort_index()

    with st.container(border=True):
        st.subheader("🔥 Najcesci brojevi u poslednjih 10 kola (TOP 5)")
        hot = counts.sort_values(ascending=False).head(5)
        hot_html = "".join([f"<div style='display:inline-block; text-align:center; margin-right:8px;'>{get_lotto_ball_html(num)}<br><span style='font-size:11px; opacity:0.6;'>{hot[num]}회</span></div>" for num in hot.index])
        st.markdown(hot_html, unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("❄️ Brojevi bez pojave u poslednjih 10 kola")
        missing = [n for n in range(1, MAX_NUM + 1) if n not in counts.index]
        missing_html = "".join([get_lotto_ball_html(num) for num in missing[:7]])
        st.markdown(missing_html, unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("⚖️ Odnos parnih/neparnih i prosecna suma")
        even = sum(n % 2 == 0 for n in all_nums)
        odd = sum(n % 2 == 1 for n in all_nums)
        sums = recent[['num1','num2','num3','num4','num5','num6','num7']].sum(axis=1)

        odd_rate = round((odd/(10 * PICK_COUNT))*100)
        even_rate = round((even/(10 * PICK_COUNT))*100)
        avg_sum = round(sums.mean(), 1)

        metrics_html = f"""
        <div style="display: flex; justify-content: space-between; text-align: center; padding: 5px 0;">
            <div style="flex: 1;">
                <div style="font-size: 12px; opacity: 0.7; margin-bottom: 4px;">Neparni</div>
                <div style="font-size: 20px; font-weight: 700;">{odd_rate}%</div>
            </div>
            <div style="flex: 1; border-left: 1px solid rgba(128,128,128,0.2); border-right: 1px solid rgba(128,128,128,0.2);">
                <div style="font-size: 12px; opacity: 0.7; margin-bottom: 4px;">Parni</div>
                <div style="font-size: 20px; font-weight: 700;">{even_rate}%</div>
            </div>
            <div style="flex: 1;">
                <div style="font-size: 12px; opacity: 0.7; margin-bottom: 4px;">Prosecna suma</div>
                <div style="font-size: 20px; font-weight: 700;">{avg_sum}</div>
            </div>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("📈 Raspodela po opsezima")
        bins = {"1~10": 0, "11~20": 0, "21~30": 0, "31~39": 0}
        for n in all_nums:
            if n <= 10: bins["1~10"] += 1
            elif n <= 20: bins["11~20"] += 1
            elif n <= 30: bins["21~30"] += 1
            else: bins["31~39"] += 1
            
        df_bins = pd.DataFrame({"Opseg": list(bins.keys()), "Pojava": list(bins.values())})
        bar_chart = alt.Chart(df_bins).mark_bar(color='#ff7272', cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Opseg', sort=None, title=None),
            y=alt.Y('Pojava', title=None),
            tooltip=['Opseg', 'Pojava']
        ).properties(height=200)
        st.altair_chart(bar_chart, use_container_width=True)

    with st.container(border=True):
        st.subheader("➕ Kretanje sume u poslednjih 10 kola")
        df_sums = pd.DataFrame({
            "Kolo": recent['draw_no'].astype(str),
            "Suma": sums.values
        }).iloc[::-1] 
        
        line_chart = alt.Chart(df_sums).mark_line(color='#69c8f2', point=True).encode(
            x=alt.X('Kolo', sort=None, title=None),
            y=alt.Y('Suma', scale=alt.Scale(zero=False), title=None),
            tooltip=['Kolo', 'Suma']
        ).properties(height=200)
        st.altair_chart(line_chart, use_container_width=True)

# ==========================================
# 🖥️ Main web screen configuration
# ==========================================
inject_ga("G-G0KYYZPQ2L")
apply_mobile_layout()

st.title("🎲 AI Loto aplikacija")

df = load_data()

if not df.empty:
    latest_draw = df['draw_no'].iloc[0]
    next_draw = latest_draw + 1 
    
    tab1, tab2 = st.tabs(["🔮 Predikcija i statistika", "🏆 Provera tiketa"])
    
    # ----------------------------------------
    # [탭 1] Existing prediction and statistics screen
    # ----------------------------------------
    with tab1:
        st.info(f"✅ Model je obradio kolo **{latest_draw}** (sledece: **{next_draw}**)")
        
        with st.container(border=True):
            with st.expander("🛠️ Tvoja pravila (fiksni/iskljuceni brojevi)"):
                all_numbers = list(range(1, MAX_NUM + 1))
                fixed_nums = st.multiselect("📌 Fiksni brojevi (max 5)", all_numbers, max_selections=5)
                excluded_nums = st.multiselect("❌ Iskljuceni brojevi", all_numbers)
                
                intersect = set(fixed_nums) & set(excluded_nums)
                if intersect:
                    st.error(f"⚠️ Isti broj je i fiksan i iskljucen: {', '.join(map(str, intersect))}")

        if st.button("✨ Generisi NEXT kombinaciju", type="primary", use_container_width=True):
            if intersect:
                st.warning("Prvo ukloni konflikt fiksnih i iskljucenih brojeva.")
            else:
                with st.spinner('Racunam optimalne kombinacije...'):
                    ai_sets = generate_ai_numbers(df, num_sets=1, fixed_nums=fixed_nums, excluded_nums=excluded_nums)
                    
                    with st.container(border=True):
                        st.subheader("🎯 AI predlozene kombinacije")
                        share_text = f"🤖 AI loto predlog (kolo {next_draw})\n\n"
                        
                        for i, num_set in enumerate(ai_sets):
                            st.caption("**NEXT kombinacija**")
                            balls_html = "".join([get_lotto_ball_html(num) for num in num_set])
                            st.markdown(f'<div style="margin-bottom: 12px;">{balls_html}</div>', unsafe_allow_html=True)
                            share_text += f"NEXT: {', '.join(map(str, num_set))}\n"
                        
                        share_text += "\ngenerisanje AI brojeva"
                        
                        st.write("---")
                        st.write("📲 **Podeli brojeve**")
                        st.code(share_text, language="text")

        st.write("---")
        show_statistics_dashboard(df)
        
        with st.expander("📊 Baza prethodnih kola"):
            st.dataframe(df, use_container_width=True)

    # ----------------------------------------
    # [탭 2] 🌟 Winning check (Bonus number and 2nd place determination logic applied)
    # ----------------------------------------
    with tab2:
        st.write("") 
        
        check_method = st.radio(
            "🔍 Izaberi nacin provere:",
            ("⌨️ Rucni unos brojeva", "📷 QR skeniranje (kamera)"),
            horizontal=True
        )
        
        # [모드 1] QR code scan
        if check_method == "📷 QR skeniranje (kamera)":
            with st.container(border=True):
                st.subheader("📷 Skeniranje QR koda")
                st.caption("Usmeri kameru na QR kod sa tiketa.")
                st.info("💡 Ako se pali prednja kamera, prebaci na zadnju.")
                if cv2 is None:
                    st.error("OpenCV (cv2) nije instaliran, QR skeniranje nije dostupno.")
                    st.stop()
                
                img_file_buffer = st.camera_input("Skeniraj QR kod")

                if img_file_buffer is not None:
                    image = Image.open(img_file_buffer)
                    img_array = np.array(image)
                    
                    detector = cv2.QRCodeDetector()
                    qr_data, bbox, _ = detector.detectAndDecode(img_array)
                    
                    if qr_data:
                        if "v=" in qr_data:
                            v_string = qr_data.split("v=")[1]
                            chunks = re.split(r'[mq]', v_string) 
                            
                            try:
                                qr_draw_no = int(chunks[0])
                                target_draw_data = df[df['draw_no'] == qr_draw_no]
                                
                                st.success(f"✅ QR skeniranje uspesno! (kolo {qr_draw_no})")
                                
                                if target_draw_data.empty:
                                    st.warning(f"😅 Kolo {qr_draw_no} jos nije u lokalnoj bazi.")
                                else:
                                    win_nums = target_draw_data.iloc[0][['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7']].tolist()
                                    
                                    st.write("---")
                                    st.write(f"### 🎯 Rezultat kola {qr_draw_no}")
                                    win_html = "".join([get_lotto_ball_html(num) for num in win_nums])
                                    st.markdown(f"**Izvuceni brojevi:**<br>{win_html}", unsafe_allow_html=True)
                                    st.write("")
                                    
                                    games = [c for c in chunks[1:] if len(c) == 14] 
                                    for idx, game_str in enumerate(games):
                                        my_nums = [int(game_str[i:i+2]) for i in range(0, 14, 2)]
                                        match_count = len(set(my_nums) & set(win_nums))
                                        
                                        my_html = ""
                                        for num in sorted(my_nums):
                                            if num in win_nums:
                                                my_html += get_lotto_ball_html(num)
                                            else:
                                                my_html += f'<div class="lotto-ball" style="background-color: #333333; color: gray;">{num}</div>'
                                        
                                        result_text = "Bez dobitka 😢"
                                        if match_count == 7: result_text = "1. nagrada 🎉"
                                        elif match_count == 6: result_text = "2. nagrada 🎊"
                                        elif match_count == 5: result_text = "3. nagrada 🎊"
                                        elif match_count == 4: result_text = "4. nagrada 👍"
                                        elif match_count == 3: result_text = "5. nagrada 💰"
                                        
                                        st.markdown(f"**{chr(65+idx)} igra [{result_text}]**<br>{my_html}", unsafe_allow_html=True)
                            except:
                                st.error("QR je procitan, ali format nije podrzan.")
                        else:
                            st.error("Ovo nije podrzan QR format tiketa.")
                    else:
                        st.warning("QR nije pronadjen. Pokusaj ponovo uz bolji fokus/svetlo.")

        # [모드 2] Manual input
        else:
            with st.container(border=True):
                st.subheader("⌨️ Rucna provera brojeva")
                check_draw = st.number_input("📌 Unesi broj kola", min_value=1, max_value=latest_draw, value=latest_draw)
                user_input = st.text_input("Primer: 7, 14, 20, 33, 38, 39, 2", placeholder="Unesi 7 brojeva")
                
                if st.button("Proveri rezultat", type="primary", use_container_width=True):
                    if not user_input:
                        st.warning("Unesi brojeve.")
                    else:
                        try:
                            my_nums = [int(x.strip()) for x in user_input.split(',')]
                            if len(my_nums) != PICK_COUNT:
                                st.error(f"Unesi tacno {PICK_COUNT} brojeva. (uneseno: {len(my_nums)})")
                            elif len(set(my_nums)) != PICK_COUNT:
                                st.error("Imas duplikate u unosu.")
                            elif any(n < 1 or n > MAX_NUM for n in my_nums):
                                st.error(f"Brojevi moraju biti u opsegu 1-{MAX_NUM}.")
                            else:
                                target_draw_data = df[df['draw_no'] == check_draw]
                                if target_draw_data.empty:
                                    st.error("Nema podataka za to kolo.")
                                else:
                                    win_nums = target_draw_data.iloc[0][['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7']].tolist()
                                    match_count = len(set(my_nums) & set(win_nums))
                                    
                                    st.write("---")
                                    st.subheader(f"✅ Rezultat za kolo {check_draw}")
                                    
                                    st.caption("Izvuceni brojevi")
                                    win_html = "".join([get_lotto_ball_html(num) for num in win_nums])
                                    st.markdown(f"{win_html}", unsafe_allow_html=True)
                                    
                                    st.caption("Tvoji brojevi")
                                    my_html = ""
                                    for num in sorted(my_nums):
                                        if num in win_nums: my_html += get_lotto_ball_html(num) 
                                        else: my_html += f'<div class="lotto-ball" style="background-color: #333333; color: gray;">{num}</div>'
                                    st.markdown(my_html, unsafe_allow_html=True)
                                    
                                    st.write("---")
                                    if match_count == 7: 
                                        st.success("🎉 Cestitam! 1. nagrada! (7 pogodaka)")
                                        st.balloons()
                                    elif match_count == 6: 
                                        st.success("🎉 Cestitam! 2. nagrada! (6 pogodaka)")
                                        st.balloons()
                                    elif match_count == 5: 
                                        st.success("🎉 Cestitam! 3. nagrada! (5 pogodaka)")
                                    elif match_count == 4: 
                                        st.info("👍 4. nagrada! (4 pogotka)")
                                    elif match_count == 3: 
                                        st.info("💰 5. nagrada! (3 pogotka)")
                                    else: 
                                        st.error(f"😢 Bez dobitka. (pogodaka: {match_count})")
                        except ValueError:
                            st.error("Unesi samo brojeve odvojene zarezima.")
else:
    st.error("CSV fajl nije pronadjen.")




"""
🤖 AI loto predlog (kolo 4597)

NEXT: 8, 9, x, y, z, 23, 26

generisanje AI brojeva
"""



"""
Streamlit jednostranična aplikacija: 
dva taba (predikcija + statistika / provera tiketa), 
mobilno sužen CSS, učitavanje istorije iz CSV-a, 
„predlog brojeva“, Altair grafici, opciono OpenCV za QR.


Tehnike / metode po blokovima

st.cache_data(ttl=600)
   Kešira load_data() ~10 min 
   — manje citanja diska pri rerun-u.
load_data()
   Pokušaj 1: 
   kolone num1…num7 (case-insensitive); 
   numerika, obrnut red (iloc[::-1]), 
   pa uvešta draw_no kao arange(len, 0, -1) 
   — nije uvek isto kao kolona iz CSV-a. 
   Pokušaj 2: 
   header=None, fiksna imena kolona uključujući 
   draw_no iz fajla.
generate_ai_numbers()
   Nije ML model: 
   ponderisana slučajnost. 
   Težine = baza 1.0 + frekvencija u celoj istoriji 
   x 0.02 + frekvencija u poslednjih 10 kola x 0.5; 
   fiksni/isključeni ruše težinu na 0. 
   Zatim random.choices uz težine dok ne skupi 
   7 različitih (fiksni prvo), sortiranje.
show_statistics_dashboard()
   Poslednjih 10 kola: 
   top 5 „vrelih“, „hladni“ (nema pojave), 
   parni/neparni %, prosečna suma, 
   histogram opsega 1-10…31-39, 
   linijski grafik sume po kolu — Altair.
Tab 2 QR
   st.camera_input → PIL → numpy → cv2.QRCodeDetector; 
   očekuje URL sa v= i poseban string (split po m/q), 
   izvlačenje kola i blokova od 14 cifara po kombinaciji.
Tab 2 ručno
   number_input za kolo, tekstualni unos 7 brojeva, 
   presek skupova sa izvučenim, poruke za „nagrade“ po broju pogodaka.
inject_ga / components.html
   Injekcija gtag skripte u parent dokument (GA merenje).




Dobre strane

Jasna podela: UI, podaci, „generator“, statistika, provera.

Keš nad CSV-om smanjuje opterećenje.

Robustniji učitavanje kad postoje imenovane num* kolone.

UI: lopte po opsegu boja, sužen layout za telefon, tabovi.

Fiksni / isključeni brojevi sa proverom preseka.

Ručna provera ima osnovnu validaciju (broj, opseg, duplikati).




Slabe / rizične strane

„AI“ u imenu: 
algoritam je heuristika + slučaj; 
nema treniranja modela, nema evaluacije predikcije 
— očekivanje „pameti“ može biti zavaravajuće.

except: 
bez tipa u load_data i u QR bloku — guta greške, teže debugovanje.

QR format 
izgleda prilagođen određenom tiketu; treba provera da li v= / mq split uopšte važi.

Fiksni u generate_ai_numbers: 
težina 0 za fiksne je suvišna ako su već u selected, ali nije bug; 
bitnije je da je sve determinisano seed-om (SEED = 39) 
— ista sesija daje isti raspored slučajnosti dok se seed/state ne promeni.




Aplikacija je Streamlit vizuelizacija 
+ statistika poslednjih kola 
+ ponderisani random generator 
+ provera tiketa (ručno sigurnije nego QR bez potvrde formata). 
Jače strane su UX i brz pregled istorije; 
slabije su semantička reč „AI“, konzistentnost draw_no/CSV, 
široki except, putanja i verovatno neusaglašenost QR/pravila nagrada sa stvarnim tiketom.
"""
