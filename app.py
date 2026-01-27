import streamlit as st
import cv2
import numpy as np
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CONFIGURATION ---
RESIZE_WIDTH = 800  
CSV_FILENAME = "fish_measurements.csv"

# Species Database (a and b values)
SPECIES_DB = {
    "Silver Pompano (Trachinotus blochii)": {
        "a": 0.0263, "b": 2.83, 
        "note": "Measure Total Length (Tip to Tail)"
    },
    "Spiny Lobster (Panulirus homarus)": {
        "a": 0.0021, "b": 2.77, 
        "note": "Measure Carapace ONLY (Horns to start of Tail)"
    }
}

# --- HELPER FUNCTIONS ---
def resize_image(image, width):
    """Resizes image to fixed width while keeping aspect ratio"""
    (h, w) = image.shape[:2]
    r = width / float(w)
    dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def estimate_weight(length_cm, a, b):
    return a * (length_cm ** b)

def reset_for_next_image():
    """Clears clicks and moves to next image index"""
    st.session_state["clicks"] = []
    st.session_state["image_index"] += 1
    st.session_state["component_key"] += 1
    st.rerun()

# --- MAIN APP ---
st.set_page_config(page_title="Batch Fish Measurer", layout="wide")
st.title("🐟 Batch Fish Processor")

# 1. SIDEBAR: SETTINGS & DATA
with st.sidebar:
    st.header("1. Settings")
    species = st.selectbox("Select Species", list(SPECIES_DB.keys()))
    selected_fish = SPECIES_DB[species]
    st.caption(selected_fish['note'])
    
    ruler_val = st.number_input("Ruler Reference (cm)", value=1.0, step=0.1, help="The real distance between your first two clicks (e.g., 7cm to 8cm is 1.0)")
    
    st.divider()
    st.header("3. Results")
    
    if "results" not in st.session_state:
        st.session_state["results"] = []

    if st.session_state["results"]:
        df = pd.DataFrame(st.session_state["results"])
        st.dataframe(df, hide_index=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name=CSV_FILENAME, mime="text/csv")
    else:
        st.info("No measurements yet.")

# 2. FILE UPLOADER
uploaded_files = st.file_uploader("Upload Images (Select Multiple)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Initialize Session State
if "image_index" not in st.session_state: st.session_state["image_index"] = 0
if "clicks" not in st.session_state: st.session_state["clicks"] = []
if "component_key" not in st.session_state: st.session_state["component_key"] = 0

# 3. PROCESSING LOOP
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
        
        # Resize for display
        img_display = resize_image(img_rgb, RESIZE_WIDTH)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"Processing: {current_file.name} ({st.session_state['image_index'] + 1}/{len(uploaded_files)})")
            
            # --- DRAWING LOGIC ---
            img_with_dots = img_display.copy()
            
            for i, (x, y) in enumerate(st.session_state["clicks"]):
                color = (255, 0, 0) if i < 2 else (0, 255, 0)
                cv2.circle(img_with_dots, (x, y), 5, color, -1)
                if i == 1:
                    cv2.line(img_with_dots, st.session_state["clicks"][0], (x, y), (255, 0, 0), 2)
                if i == 3:
                    cv2.line(img_with_dots, st.session_state["clicks"][2], (x, y), (0, 255, 0), 2)

            # --- INTERACTIVE IMAGE ---
            dynamic_key = f"clicker_{st.session_state['component_key']}"
            value = streamlit_image_coordinates(img_with_dots, key=dynamic_key)
            
            if value is not None:
                new_point = (value["x"], value["y"])
                if not st.session_state["clicks"] or st.session_state["clicks"][-1] != new_point:
                    st.session_state["clicks"].append(new_point)
                    st.rerun()

        with col2:
            st.subheader("Instructions")
            clicks_count = len(st.session_state["clicks"])
            
            if clicks_count == 0: st.markdown("🔴 **Click 1:** Ruler Start")
            elif clicks_count == 1: st.markdown("🔴 **Click 2:** Ruler End")
            elif clicks_count == 2: st.markdown("🟢 **Click 3:** Fish Nose")
            elif clicks_count == 3: st.markdown("🟢 **Click 4:** Fish Tail")
            elif clicks_count >= 4:
                st.success("Measurement Complete!")
                
                # --- CALCULATION ---
                pts = st.session_state["clicks"]
                ruler_px = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
                px_per_cm = ruler_px / ruler_val
                fish_px = np.linalg.norm(np.array(pts[2]) - np.array(pts[3]))
                length_cm = fish_px / px_per_cm
                weight_g = estimate_weight(length_cm, selected_fish['a'], selected_fish['b'])
                
                # Display Results on Screen
                st.metric("Length", f"{length_cm:.2f} cm")
                st.metric("Weight", f"{weight_g:.0f} g")
                
                # --- BURN TEXT INTO IMAGE FOR DOWNLOAD ---
                final_img = img_with_dots.copy()
                label_text = f"L: {length_cm:.2f}cm | W: {weight_g:.0f}g"
                
                # Draw a black rectangle background for text visibility
                text_pos = (pts[2][0], pts[2][1] - 10) # Position above fish nose
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(final_img, (text_pos[0], text_pos[1] - h - 5), (text_pos[0] + w, text_pos[1] + 5), (0,0,0), -1)
                
                # Draw the text
                cv2.putText(final_img, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Convert to PNG for download
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
                
                if is_success:
                    st.download_button(
                        label="📥 Download this Image",
                        data=buffer.tobytes(),
                        file_name=f"measured_{current_file.name}",
                        mime="image/png"
                    )

                # --- SAVE & NEXT ---
                st.divider()
                if st.button("Save Data & Next Image", type="primary"):
                    st.session_state["results"].append({
                        "Filename": current_file.name,
                        "Species": species,
                        "Length (cm)": round(length_cm, 2),
                        "Weight (g)": round(weight_g, 2)
                    })
                    reset_for_next_image()

            if clicks_count > 0:
                if st.button("Undo Last Click"):
                    st.session_state["clicks"].pop()
                    st.session_state["component_key"] += 1
                    st.rerun()

else:
    st.info("👆 Please upload images to begin.")
