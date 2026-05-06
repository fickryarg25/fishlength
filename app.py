import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
import io
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(page_title="Aquaculture Management", layout="wide")
st.title("🌊 Aquaculture Management System")

# Membuat 2 Tab
tab_measure, tab_feeding = st.tabs(["🐟 Measure Fish", "📊 Feeding Projection"])


# ==========================================
# TAB 1: FISH MEASUREMENT (KODE LAMA ANDA)
# ==========================================
with tab_measure:
    st.header("Batch Fish Processor")
    
    # --- CONFIGURATION ---
    CSV_FILENAME = "fish_measurements.csv"

    SPECIES_DB = {
        "Silver Pompano (Trachinotus blochii)": {
            "a": 0.0263, "b": 2.83, 
            "note": "Measure Total Length (Tip to Tail)"
        },
        "Spiny Lobster (P. homarus) - Total Length": {
            "a": 0.035, "b": 2.89, 
            "note": "Measure Horns to Tail Fan."
        }
    }

    def resize_image(image, width):
        (h, w) = image.shape[:2]
        r = width / float(w)
        dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def estimate_weight(length_cm, a, b):
        return a * (length_cm ** b)

    def reset_for_next_image():
        st.session_state["clicks"] = []
        st.session_state["image_index"] += 1
        st.session_state["component_key"] += 1
        st.rerun()

    with st.sidebar:
        st.header("Settings")
        resize_width = st.slider("Image Display Size", min_value=400, max_value=1200, value=700, step=50)
        species = st.selectbox("Select Species", list(SPECIES_DB.keys()))
        selected_fish = SPECIES_DB[species]
        ruler_val = st.number_input("Ruler Reference (cm)", value=1.0, step=0.1)
        
        st.divider()
        st.header("Results & Uniformity")
        
        if "results" not in st.session_state: st.session_state["results"] = []

        if st.session_state["results"]:
            df = pd.DataFrame(st.session_state["results"])
            
            if len(df) > 1:
                mean_length = df["Length (cm)"].mean()
                std_length = df["Length (cm)"].std()
                if mean_length > 0:
                    uniformity_score = (1 - (std_length / mean_length)) * 100
                else:
                    uniformity_score = 0.0
                
                df["Batch_Mean_Length (cm)"] = round(mean_length, 2)
                df["Uniformity (%)"] = round(uniformity_score, 2)
                st.metric("Batch Uniformity", f"{uniformity_score:.1f}%")
            else:
                df["Batch_Mean_Length (cm)"] = df["Length (cm)"]
                df["Uniformity (%)"] = 100.0
                st.metric("Batch Uniformity", "100.0%")
            
            st.dataframe(df, hide_index=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV Data", data=csv, file_name=CSV_FILENAME, mime="text/csv")
            
            if st.button("Clear All Data"):
                st.session_state["results"] = []
                st.rerun()

    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if "image_index" not in st.session_state: st.session_state["image_index"] = 0
    if "clicks" not in st.session_state: st.session_state["clicks"] = []
    if "component_key" not in st.session_state: st.session_state["component_key"] = 0
    if "last_resize_width" not in st.session_state: st.session_state["last_resize_width"] = resize_width

    if st.session_state["last_resize_width"] != resize_width:
        st.session_state["clicks"] = []
        st.session_state["component_key"] += 1
        st.session_state["last_resize_width"] = resize_width

    if uploaded_files:
        if st.session_state["image_index"] >= len(uploaded_files):
            st.success("✅ All images processed!")
            if st.button("Start Over"):
                st.session_state["image_index"] = 0
                st.session_state["results"] = []
                st.session_state["clicks"] = []
                st.session_state["component_key"] = 0
                st.rerun()
        else:
            current_file = uploaded_files[st.session_state["image_index"]]
            file_bytes = np.asarray(bytearray(current_file.read()), dtype=np.uint8)
            img_original = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            img_display = resize_image(img_rgb, resize_width)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"Image: {current_file.name}")
                img_with_dots = img_display.copy()
                clicks = st.session_state["clicks"]
                
                if len(clicks) >= 2:
                    cv2.line(img_with_dots, clicks[0], clicks[1], (255, 0, 0), 2)
                    cv2.circle(img_with_dots, clicks[0], 5, (255, 0, 0), -1)
                    cv2.circle(img_with_dots, clicks[1], 5, (255, 0, 0), -1)
                elif len(clicks) == 1:
                     cv2.circle(img_with_dots, clicks[0], 5, (255, 0, 0), -1)

                fish_count = 0
                for i in range(2, len(clicks), 2):
                    pt_head = clicks[i]
                    cv2.circle(img_with_dots, pt_head, 5, (0, 255, 0), -1)
                    if i + 1 < len(clicks):
                        pt_tail = clicks[i+1]
                        cv2.circle(img_with_dots, pt_tail, 5, (0, 255, 0), -1)
                        cv2.line(img_with_dots, pt_head, pt_tail, (0, 255, 0), 2)
                        fish_count += 1
                        cv2.putText(img_with_dots, f"#{fish_count}", (pt_head[0], pt_head[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                dynamic_key = f"clicker_{st.session_state['component_key']}"
                value = streamlit_image_coordinates(img_with_dots, key=dynamic_key)
                
                if value is not None:
                    new_point = (value["x"], value["y"])
                    if not clicks or clicks[-1] != new_point:
                        st.session_state["clicks"].append(new_point)
                        st.rerun()

            with col2:
                st.subheader("Instructions")
                n = len(clicks)
                if n == 0: st.markdown("🔴 **Click 1:** Ruler Start")
                elif n == 1: st.markdown("🔴 **Click 2:** Ruler End")
                elif n >= 2:
                    if (n - 2) % 2 == 0: st.markdown(f"🟢 **Click:** Fish #{ (n-2)//2 + 1 } Head")
                    else: st.markdown(f"🟢 **Click:** Fish #{ (n-2)//2 + 1 } Tail")
                
                if n >= 4 and n % 2 == 0:
                    ruler_px = np.linalg.norm(np.array(clicks[0]) - np.array(clicks[1]))
                    px_per_cm = ruler_px / ruler_val
                    current_fish_data = []
                    count = 1
                    for i in range(2, len(clicks), 2):
                        pt1 = np.array(clicks[i])
                        pt2 = np.array(clicks[i+1])
                        fish_px = np.linalg.norm(pt1 - pt2)
                        len_cm = fish_px / px_per_cm
                        w_g = estimate_weight(len_cm, selected_fish['a'], selected_fish['b'])
                        current_fish_data.append({"Filename": current_file.name, "Species": species, "Fish_ID": count, "Length (cm)": round(len_cm, 2), "Weight (g)": round(w_g, 2)})
                        count += 1
                    
                    if st.button(f"Save {count-1} Fish & Next", type="primary"):
                        st.session_state["results"].extend(current_fish_data)
                        reset_for_next_image()

                if n > 0:
                    if st.button("Undo Last Click"):
                        st.session_state["clicks"].pop()
                        st.session_state["component_key"] += 1
                        st.rerun()


# ==========================================
# TAB 2: FEEDING PROJECTION (TAB BARU)
# ==========================================
with tab_feeding:
    st.header("14-Day Feeding Projection")
    
    # 1. Fungsi penentu Feed Rate berdasarkan tabel standar Anda
    def get_feed_rate(mbw):
        if mbw < 80.60: return 5.0
        elif mbw < 125.40: return 4.5
        elif mbw < 202.20: return 3.8
        elif mbw < 250.20: return 3.5
        elif mbw < 300.70: return 3.3
        else: return 3.0

    # 2. Layout Input (Dibuat mirip screenshot Anda)
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1: cage_code = st.text_input("Cage Code", "C1F")
    with c2: start_date = st.date_input("Start Date", datetime.date.today())
    with c3: start_doc = st.number_input("Start DOC", value=57, step=1)
    with c4: population = st.number_input("Population", value=905, step=1)
    with c5: current_mbw = st.number_input("Current MBW (g)", value=49.0, step=1.0)
    with c6: target_mbw = st.number_input("Target MBW (g)", value=80.0, step=1.0)
    with c7: feeding_days = st.number_input("Feeding Days", value=13, step=1)

    st.write("") # Spacing

    # 3. Logika Perhitungan
    if st.button("📊 Generate Feeding Plan", type="primary"):
        # Hitung Average Daily Gain
        adg = (target_mbw - current_mbw) / feeding_days
        
        data = []
        accumulative = 0.0
        
        # Looping untuk setiap hari feeding
        for i in range(feeding_days):
            doc = start_doc + i
            date = start_date + datetime.timedelta(days=i)
            
            # Proyeksi MBW bertambah sesuai ADG
            proj_mbw = current_mbw + (adg * i)
            
            # Hitung Biomassa & Pakan
            biomass = (population * proj_mbw) / 1000.0
            fr = get_feed_rate(proj_mbw)
            std_feed = biomass * (fr / 100.0)
            
            accumulative += std_feed
            
            data.append({
                "DOC": doc,
                "DATE": date.strftime("%d %b %Y"),
                "POPULATION": population,
                "PROJ. MBW (G)": f"{proj_mbw:.2f}",
                "ADG (G/DAY)": f"{adg:.2f}",
                "TARGET BIOMASS": f"{biomass:.2f}",
                "FEED RATE %": f"{fr:.1f}",
                "STANDARD FEED (KG)": f"{std_feed:.2f}",
                "ACCUMULATIVE": f"{accumulative:.2f}"
            })
            
        # Looping selesai, tambahkan 1 hari otomatis untuk Fasting Day
        fasting_doc = start_doc + feeding_days
        fasting_date = start_date + datetime.timedelta(days=feeding_days)
        
        data.append({
            "DOC": fasting_doc,
            "DATE": fasting_date.strftime("%d %b %Y"),
            "POPULATION": "-",
            "PROJ. MBW (G)": "-",
            "ADG (G/DAY)": "-",
            "TARGET BIOMASS": "-",
            "FEED RATE %": "-",
            "STANDARD FEED (KG)": "-",
            "ACCUMULATIVE": "FASTING DAY (SAMPLING)"
        })
        
        # 4. Tampilkan Tabel
        df_feed = pd.DataFrame(data)
        
        # Styling Pandas untuk memberikan warna pada tabel jika diinginkan
        st.dataframe(df_feed, use_container_width=True, hide_index=True)
        
        # 5. Fitur Download ke Excel (.xlsx)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_feed.to_excel(writer, index=False, sheet_name='Feeding Plan')
            
            # Merapikan lebar kolom di Excel
            worksheet = writer.sheets['Feeding Plan']
            for i, col in enumerate(df_feed.columns):
                column_len = max(df_feed[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, column_len)

        st.download_button(
            label="📥 Download Feeding Plan (Excel)",
            data=buffer.getvalue(),
            file_name=f"Feeding_Plan_{cage_code}_DOC{start_doc}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
