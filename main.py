# app.py — Streamlit YOLOv11 viewer with enhancements (Grayscale CLAHE + Lab-L CLAHE)
import streamlit as st
import numpy as np
import cv2, random
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv11 Floor-Plan Viewer", layout="wide")

# --------------------- helpers ---------------------
@st.cache_resource
def load_model(path: str):
    return YOLO(path)

def class_color(cid: int):
    random.seed(int(cid) + 12345)
    return tuple(int(x) for x in np.array([random.randrange(60,255) for _ in range(3)]))

def ensure_odd(v: int, minv: int = 3):
    v = int(v)
    if v < minv: v = minv
    if v % 2 == 0: v += 1
    return v

def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def draw_transparent_boxes(img_bgr, boxes, alpha=0.5):
    overlay = img_bgr.copy()
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)
        for bb, cl in zip(xyxy, cls):
            x1, y1, x2, y2 = [int(round(v)) for v in bb]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), class_color(cl), thickness=-1)
    return cv2.addWeighted(overlay, float(alpha), img_bgr, 1 - float(alpha), 0.0)

def detections_to_text(result):
    if result.boxes is None or len(result.boxes) == 0:
        return "Detections: 0"
    names = result.names
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls  = result.boxes.cls.cpu().numpy().astype(int)
    lines = [f"{i:02d}. {names.get(int(c), str(int(c)))}\t{cf:.2f}\t[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
             for i, ((x1,y1,x2,y2), cf, c) in enumerate(zip(xyxy, conf, cls), start=1)]
    return "Detections: " + str(len(lines)) + "\n" + "\n".join(lines)

# --------------------- enhancement pipeline ---------------------
def enhance_image(
    img_bgr,
    gray_first: bool,
    clahe_mode: str,             # "None", "Grayscale CLAHE", "Lab-L CLAHE"
    clahe_clip: float, clahe_grid: int,
    use_bc: bool, alpha_gain: float, beta_bias: float,
    use_unsharp: bool, us_amount: float, us_radius: float,
    use_thresh: bool, th_block: int, th_C: int,
    use_morph: bool, morph_k: int, morph_iters: int,
    use_invert: bool
):
    base = img_bgr.copy()
    work_gray = None

    if gray_first:
        work_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    # --- CLAHE ---
    if clahe_mode == "Grayscale CLAHE":
        if work_gray is None:
            work_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_grid), int(clahe_grid)))
        work_gray = clahe.apply(work_gray)
    elif clahe_mode == "Lab-L CLAHE":
        lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_grid), int(clahe_grid)))
        L_eq = clahe.apply(L)
        lab_eq = cv2.merge([L_eq, a, b])
        base = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # --- Brightness/Contrast ---
    if use_bc:
        if work_gray is not None:
            work_gray = cv2.convertScaleAbs(work_gray, alpha=float(alpha_gain), beta=float(beta_bias))
        else:
            base = cv2.convertScaleAbs(base, alpha=float(alpha_gain), beta=float(beta_bias))

    # --- Unsharp mask ---
    if use_unsharp:
        if work_gray is not None:
            blur = cv2.GaussianBlur(work_gray, (0,0), sigmaX=float(us_radius), sigmaY=float(us_radius))
            work_gray = cv2.addWeighted(work_gray, 1.0 + float(us_amount), blur, -float(us_amount), 0)
        else:
            blur = cv2.GaussianBlur(base, (0,0), sigmaX=float(us_radius), sigmaY=float(us_radius))
            base = cv2.addWeighted(base, 1.0 + float(us_amount), blur, -float(us_amount), 0)

    # --- Adaptive threshold (binarize) ---
    if use_thresh:
        if work_gray is None:
            work_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        bs = ensure_odd(int(th_block), 3)
        work_gray = cv2.adaptiveThreshold(
            work_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, int(th_C)
        )

    # Merge back to color if we were in gray path
    if work_gray is not None:
        base = to_bgr(work_gray)

    # --- Morphological OPEN (clean speckles/lines) ---
    if use_morph:
        k = max(1, int(morph_k))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        base = cv2.morphologyEx(base, cv2.MORPH_OPEN, kernel, iterations=int(morph_iters))

    # --- Invert ---
    if use_invert:
        base = cv2.bitwise_not(base)

    return base

# --------------------- UI ---------------------
st.title("YOLOv11 — Split View with Floor-Plan Enhancers")
left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png","bmp","tif","tiff","webp"])
    model_path = st.text_input("Model (.pt)", value="weights/best.pt")
    conf = st.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
    iou  = st.slider("IoU",        0.0, 1.0, 0.45, 0.01)
    alpha= st.slider("Box transparency (alpha)", 0.0, 1.0, 0.50, 0.05)
    detect_on_enh = st.checkbox("Run detection on enhanced image", value=True)

    with st.expander("Enhancements", expanded=True):
        gray_first  = st.checkbox("Convert to Grayscale first", value=True)
        clahe_mode  = st.selectbox("CLAHE mode", ["None", "Grayscale CLAHE", "Lab-L CLAHE"], index=1)
        c1,c2 = st.columns(2)
        with c1:
            clahe_clip = st.slider("CLAHE clipLimit", 1.0, 6.0, 2.0, 0.1)
            use_bc     = st.checkbox("Brightness/Contrast", value=False)
            alpha_gain = st.slider("Contrast gain (alpha)", 0.5, 2.0, 1.2, 0.05)
        with c2:
            clahe_grid = st.slider("CLAHE tileGridSize", 4, 16, 8, 1)
            beta_bias  = st.slider("Brightness bias (beta)", -64, 64, 0, 1)

        use_unsharp = st.checkbox("Unsharp mask (sharpen)", value=True)
        us_amount   = st.slider("Unsharp amount", 0.0, 2.0, 0.8, 0.05)
        us_radius   = st.slider("Unsharp radius (sigma)", 0.3, 5.0, 1.2, 0.1)

        use_thresh  = st.checkbox("Adaptive threshold (binarize)", value=False)
        th_block    = st.slider("Adaptive block size (odd)", 3, 51, 25, 2)
        th_C        = st.slider("Adaptive C", -15, 15, 5, 1)

        use_morph   = st.checkbox("Morphological clean-up (OPEN)", value=False)
        morph_k     = st.slider("Morph kernel size", 1, 9, 3, 1)
        morph_iters = st.slider("Morph iterations", 0, 5, 1, 1)

        use_invert  = st.checkbox("Invert B/W", value=False)

    run = st.button("Run")

with right:
    st.subheader("Result")
    image_slot = st.empty()
    det_text_slot = st.empty()

# --------------------- run pipeline ---------------------
def run_once():
    if uploaded is None:
        st.warning("Upload an image to run inference.")
        return

    # read image (BGR)
    file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Failed to decode image. Try a different file.")
        return

    # enhance for visualization
    enh_bgr = enhance_image(
        img_bgr,
        gray_first, clahe_mode, clahe_clip, int(clahe_grid),
        use_bc, alpha_gain, beta_bias,
        use_unsharp, us_amount, us_radius,
        use_thresh, int(th_block), int(th_C),
        use_morph, int(morph_k), int(morph_iters),
        use_invert
    )

    # detection input
    det_input = enh_bgr if detect_on_enh else img_bgr

    # run model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    res = model.predict(det_input, conf=float(conf), iou=float(iou), verbose=False)[0]

    # overlay transparent boxes on the *enhanced* view (clean look, no labels)
    out_bgr = draw_transparent_boxes(enh_bgr, res.boxes, alpha=float(alpha))
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    # show image + detections text
    image_slot.image(out_rgb, caption="Enhanced + transparent boxes", use_container_width=True)
    det_text_slot.text_area(
        "Detections (class, conf, [x1,y1,x2,y2])",
        value=detections_to_text(res),
        height=240,
    )

# Trigger run on button click or auto-run when image changes
if run and uploaded is not None:
    run_once()
elif uploaded is not None and "auto_run_done" not in st.session_state:
    # auto-run once after first upload
    st.session_state["auto_run_done"] = True
    run_once()
else:
    # show placeholder
    pass
