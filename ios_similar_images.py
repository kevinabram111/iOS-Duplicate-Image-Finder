
import os
import streamlit as st
from PIL import Image
import imagehash
import pandas as pd
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import base64
from io import BytesIO
import time
import zipfile
import tempfile

# --- Settings ---
SIMILARITY_THRESHOLD = 5
PAGE_SIZE = 10
NUM_WORKERS = max(multiprocessing.cpu_count() - 1, 1)

# --- New: ZIP -> .xcassets extraction ---
def extract_xcassets_from_zip(uploaded_zip):
    """Return (temp_root, list_of_found_xcassets_paths)."""
    tmpdir = tempfile.mkdtemp(prefix="xcassets_")
    zip_path = os.path.join(tmpdir, "upload.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.read())
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)
    found = []
    for dirpath, dirnames, filenames in os.walk(tmpdir):
        for d in dirnames:
            if d.endswith(".xcassets"):
                found.append(os.path.join(dirpath, d))
    return tmpdir, found

# --- Helper Functions ---
def get_image_paths(folder):
    image_paths = []
    for root, _, files in os.walk(folder):
        pngs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '@3x' in f]
        if not pngs:
            pngs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # fallback
        if pngs:
            largest_file = max(pngs, key=lambda f: os.path.getsize(os.path.join(root, f)))
            image_paths.append(os.path.join(root, largest_file))
    return image_paths

def compute_hash(path):
    try:
        img = Image.open(path)
        if img.mode == 'P':
            img = img.convert('RGBA')
        img = img.resize((256, 256)).convert('L')
        return path, imagehash.phash(img)
    except Exception as e:
        print(f"Error hashing {path}: {e}")
        return path, None

def compare_pair(args):
    (path1, hash1), (path2, hash2) = args
    if hash1 is None or hash2 is None:
        return None
    diff = hash1 - hash2
    if diff <= SIMILARITY_THRESHOLD:
        similarity = 100 - (diff / 64 * 100)
        size1 = os.path.getsize(path1) / 1024
        size2 = os.path.getsize(path2) / 1024
        return {
            'Image A': path1,
            'Image B': path2,
            'Similarity %': round(similarity, 2),
            'Image A Size (KB)': round(size1, 1),
            'Image B Size (KB)': round(size2, 1),
            'Suggested Keep': 'A' if size1 < size2 else 'B'
        }
    return None

def encode_image_base64(path, max_size=(100, 100)):
    try:
        img = Image.open(path)
        img.thumbnail(max_size)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode()
        return f"<img src='data:image/png;base64,{encoded}' width='100'>"
    except:
        return ""

def find_similar_images_parallel(folder):
    paths = get_image_paths(folder)
    hashes = {}

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_path = {executor.submit(compute_hash, path): path for path in paths}
        for future in as_completed(future_to_path):
            path, hash_val = future.result()
            hashes[path] = hash_val

    hash_pairs = list(combinations(hashes.items(), 2))
    results = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(compare_pair, pair) for pair in hash_pairs]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    results.sort(key=lambda x: -x['Similarity %'])
    return results

# --- Streamlit App ---
st.set_page_config(page_title="iOS Duplicate Image Finder", layout="wide")
st.title("🖼️ iOS Duplicate Image Finder")

# New: Allow user to attach/upload a ZIP containing .xcassets
with st.expander("📎 Attach .xcassets (ZIP) — optional", expanded=False):
    uploaded = st.file_uploader("Upload a ZIP that contains one or more .xcassets directories", type=["zip"], accept_multiple_files=False)
    selected_assets_dir = None
    if uploaded is not None:
        with st.spinner("Extracting and locating .xcassets…"):
            tmp_root, xcassets_dirs = extract_xcassets_from_zip(uploaded)
        if not xcassets_dirs:
            st.error("No .xcassets directories were found inside the ZIP.")
        else:
            if len(xcassets_dirs) == 1:
                selected_assets_dir = xcassets_dirs[0]
                st.success(f"Found one .xcassets: {selected_assets_dir}")
            else:
                choice = st.selectbox("Multiple .xcassets found — choose one", xcassets_dirs)
                selected_assets_dir = choice
                st.success(f"Using: {selected_assets_dir}")

# If user uploaded, use that; else fall back to local ./Assets.xcassets
base_dir = os.path.dirname(os.path.abspath(__file__))
default_assets_dir = os.path.join(base_dir, "Assets.xcassets")
assets_dir = selected_assets_dir if selected_assets_dir else default_assets_dir


if not os.path.isdir(assets_dir):
    st.stop()
else:
    if 'duplicates' not in st.session_state:
        st.session_state.duplicates = []

    if st.button("🔍 Scan for Duplicates"):
        with st.spinner("Parallel scanning in progress..."):
            st.session_state.duplicates = find_similar_images_parallel(assets_dir)

    if st.session_state.duplicates:
        df = pd.DataFrame(st.session_state.duplicates)

        # === Keep original UI controls unchanged ===
        similarity_filter = st.slider("Minimum similarity %", 0, 100, 90)
        search_term = st.text_input("Search image name (optional):")
        sort_option = st.selectbox("Sort by", ["Similarity %", "Image A Size (KB)", "Image B Size (KB)"])

        df = df[df['Similarity %'] >= similarity_filter]
        if search_term:
            df = df[df['Image A'].str.contains(search_term) | df['Image B'].str.contains(search_term)]

        df = df.sort_values(by=sort_option, ascending=False)

        total_pages = max((len(df) + PAGE_SIZE - 1) // PAGE_SIZE, 1)

        if total_pages > 1:
            current_page = st.slider("📄 Page", min_value=1, max_value=total_pages, value=1, key="current_page")
        else:
            current_page = 1

        start_idx = (current_page - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, len(df))
        current_data = df.iloc[start_idx:end_idx]

        st.success(f"✅ Showing {len(current_data)} of {len(df)} visually similar image pairs.")

        for _, row in current_data.iterrows():
            img_cols = st.columns([1, 1])
            with img_cols[0]:
                st.image(row['Image A'], caption=f"A - {os.path.basename(row['Image A'])} ({row['Image A Size (KB)']} KB)", use_column_width=True)
            with img_cols[1]:
                st.image(row['Image B'], caption=f"B - {os.path.basename(row['Image B'])} ({row['Image B Size (KB)']} KB)", use_column_width=True)
            st.markdown(f"**Similarity: {row['Similarity %']}% | Suggested Keep: {row['Suggested Keep']}**")
            st.markdown("---")

        # === Keep original export-to-HTML feature unchanged ===
        if not df.empty:
            df_copy = df.copy()
            def encode_image_base64(path, max_size=(100, 100)):
                try:
                    img = Image.open(path)
                    img.thumbnail(max_size)
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    import base64 as _b64
                    encoded = _b64.b64encode(buffered.getvalue()).decode()
                    return f"<img src='data:image/png;base64,{encoded}' width='100'>"
                except:
                    return ""
            df_copy['Image A Preview'] = df_copy['Image A'].apply(lambda x: encode_image_base64(x))
            df_copy['Image B Preview'] = df_copy['Image B'].apply(lambda x: encode_image_base64(x))

            report_html = df_copy.to_html(
                escape=False, index=False,
                columns=[
                    'Image A Preview', 'Image B Preview',
                    'Image A', 'Image B',
                    'Similarity %', 'Image A Size (KB)', 'Image B Size (KB)', 'Suggested Keep'
                ]
            )

            st.download_button(
                label="📄 Download HTML Report with Previews",
                data=report_html,
                file_name="duplicate_images_report.html",
                mime="text/html"
            )
