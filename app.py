import streamlit as st
import cv2
import numpy as np
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CONFIGURATION ---
CSV_FILENAME = "fish_measurements.csv"

# Species Database
SPECIES_DB = {
    "Silver Pompano (Trachinotus blochii)": {
        "a": 0.023, "b": 2.854, 
        "note": "Measure Total Length"
    },
    "Spiny Lobster (Panulirus homarus) - TL": {
        "a": 0.0035, "b": 2.89, 
        "note": "Measure Total Length/Body Length"
    }
}

# --- HELPER FUNCTIONS ---
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

# --- MAIN APP ---
st.set_page_config(page_title="Multi-Fish Measurer", layout="wide")
st.title("🐟 Multi-Fish Batch Processor")

# 1. SIDEBAR
with st.sidebar:
    st.header("Settings")
    
    # [FIX] DYNAMIC RESIZE SLIDER
    # This allows you to adjust the image size to fit your specific laptop screen
    resize_width = st.slider("Image Display Size", min_value=400, max_value=1200, value=700, step=50, help="Reduce this if the image looks cropped/zoomed.")
    
    species = st.selectbox("Select Species", list(SPECIES_DB.keys()))
    selected_fish = SPECIES_DB[species]
    st.caption(selected_fish['note'])
    
    ruler_val = st.number_input("Ruler Reference (cm)", value=1.0, step=0.1)
    
    st.divider()
    st.header("Results")
    
    if "results" not in st.session_state: st.session_state["results"] = []

    if st.session_state["results"]:
        df = pd.DataFrame(st.session_state["results"])
        st.dataframe(df, hide_index=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name=CSV_FILENAME, mime="text/csv")
    else:
        st.info("No measurements yet.")

# 2. FILE UPLOADER
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if "image_index" not in st.session_state: st.session_state["image_index"] = 0
if "clicks" not in st.session_state: st.session_state["clicks"] = []
if "component_key" not in st.session_state: st.session_state["component_key"] = 0
if "last_resize_width" not in st.session_state: st.session_state["last_resize_width"] = resize_width

# [FIX] Reset clicks if user changes the slider (otherwise dots will be in wrong place)
if st.session_state["last_resize_width"] != resize_width:
    st.session_state["clicks"] = []
    st.session_state["component_key"] += 1
    st.session_state["last_resize_width"] = resize_width

# 3. MAIN LOGIC
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
        
        # [FIX] Use the slider value here
        img_display = resize_image(img_rgb, resize_width)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"Image: {current_file.name}")
            
            # Draw Dots/Lines
            img_with_dots = img_display.copy()
            clicks = st.session_state["clicks"]
            
            # Calibration (Blue)
            if len(clicks) >= 2:
                cv2.line(img_with_dots, clicks[0], clicks[1], (255, 0, 0), 2)
                cv2.circle(img_with_dots, clicks[0], 5, (255, 0, 0), -1)
                cv2.circle(img_with_dots, clicks[1], 5, (255, 0, 0), -1)
            elif len(clicks) == 1:
                 cv2.circle(img_with_dots, clicks[0], 5, (255, 0, 0), -1)

            # Fish (Green)
            fish_count = 0
            for i in range(2, len(clicks), 2):
                pt_head = clicks[i]
                cv2.circle(img_with_dots, pt_head, 5, (0, 255, 0), -1)
                
                if i + 1 < len(clicks):
                    pt_tail = clicks[i+1]
                    cv2.circle(img_with_dots, pt_tail, 5, (0, 255, 0), -1)
                    cv2.line(img_with_dots, pt_head, pt_tail, (0, 255, 0), 2)
                    fish_count += 1
                    cv2.putText(img_with_dots, f"#{fish_count}", (pt_head[0], pt_head[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Interactive Component
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
                if (n - 2) % 2 == 0:
                    st.markdown(f"🟢 **Click:** Fish #{ (n-2)//2 + 1 } Head")
                else:
                    st.markdown(f"🟢 **Click:** Fish #{ (n-2)//2 + 1 } Tail")
            
            if n >= 4 and n % 2 == 0:
                st.divider()
                st.write(" **Measurements:**")
                
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
                    
                    st.write(f"**#{count}:** {len_cm:.2f} cm | {w_g:.0f} g")
                    
                    current_fish_data.append({
                        "Filename": current_file.name,
                        "Species": species,
                        "Fish_ID": count,
                        "Length (cm)": round(len_cm, 2),
                        "Weight (g)": round(w_g, 2)
                    })
                    
                    label = f"#{count} {len_cm:.1f}cm"
                    cv2.putText(img_with_dots, label, (clicks[i][0], clicks[i][1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    count += 1
                
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_with_dots, cv2.COLOR_RGB2BGR))
                if is_success:
                    st.download_button("📥 Download Image", buffer.tobytes(), f"marked_{current_file.name}", "image/png")

                st.divider()
                if st.button(f"Save {count-1} Fish & Next", type="primary"):
                    st.session_state["results"].extend(current_fish_data)
                    reset_for_next_image()

            if n > 0:
                if st.button("Undo Last Click"):
                    st.session_state["clicks"].pop()
                    st.session_state["component_key"] += 1
                    st.rerun()
else:
    st.info("👆 Please upload images to begin.")
