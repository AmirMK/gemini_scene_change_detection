# app.py
import json
import re
import io
import uuid
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html
from google import genai
from google.genai import types
from google.cloud import storage

# ---------------- MODEL CALL ----------------
def scene_change_detection(video_file_url, client, model="gemini-2.5-flash"):
    system_instruction = '''
           I have a video that I need you to analyze for ad placement by detecting scene changes, 
           also known as shot boundaries. I need to identify the 10 best scene changes across the 
           entire movie, which are the best potential points for ad placement as they minimize 
           interruptions for viewers. These scene changes should be selected from all parts of the movie: 
           the beginning, middle, and the very end. Make sure you distribute the selected scenes evenly across 
           the entire movie.
           For each of these scene changes, please provide:

            timestamp: The exact timestamps indicating where the scene change occurs. Make sure that the timestamp of scenes are matched those in the original movie,
            reflecting its position accurately. The timestamps must exactly match those in the original movie.

            reason: The reason why this is a scene change and why it is a good location for ad placement. the reason 
            should be very specific. Summarize the story after and before the scene and explain why 
            between these two scenes is a good place for an ad.

            summary: A brief summary of the scene before the change.

            transition_feeling: The main feeling that the transition makes in viewers like excitement, peace, fear, etc.

            transition_type: The method used to switch from one scene to another like cuts, fades, dissolves, etc.

            narrative_type: The main role or significance of the scene in the storyline like pivotal, climatic, conflict, etc.

            dialogue_intensity: The amount and intensity of dialogue in the scene like monologue, dialogue, narration, debate, etc.

            characters_type: The types of the most important character involved in the scene transition like protagonist, antagonist, supporting, etc.

            scene_categories:  Classification of the scene before the change into the categories such as action, drama, comedy, etc.
          '''      

    response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "reason": {"type": "string"},
                "transition_feeling": {"type": "string"},
                "transition_type": {"type": "string"},
                "narrative_type": {"type": "string"},
                "dialogue_intensity": {"type": "string"},
                "characters_type": {"type": "string"},
                "scene_categories": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": [
                "timestamp", "reason", "transition_feeling", "transition_type",
                "narrative_type", "characters_type", "scene_categories"
            ],
        },
    }

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="video:"),
                types.Part.from_uri(file_uri=video_file_url, mime_type="video/mp4"),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        response_mime_type="application/json",
        response_schema=response_schema,
        system_instruction=[types.Part.from_text(text=system_instruction)],
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    try:
        response = client.models.generate_content(
            model=model, contents=contents, config=generate_content_config
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Generation failed: {e}")
        return []

# ---------------- HELPERS ----------------
def gs_to_https(gs_uri: str) -> str:
    assert gs_uri.startswith("gs://"), "Expected a gs:// URI"
    bucket, path = gs_uri[5:].split("/", 1)
    return f"https://storage.googleapis.com/{bucket}/{path}"

@st.cache_data(ttl=300)
def list_public_mp4s(bucket_name: str = "video_demo_test"):
    anon = storage.Client.create_anonymous_client()
    blobs = anon.list_blobs(bucket_name)
    gs_uris, https_urls, labels = [], [], []
    for b in blobs:
        if b.name.lower().endswith(".mp4"):
            gs_uri = f"gs://{bucket_name}/{b.name}"
            gs_uris.append(gs_uri)
            https_urls.append(gs_to_https(gs_uri))
            labels.append(b.name.split("/")[-1])
    return gs_uris, https_urls, labels

def parse_timestamp_to_seconds(ts: str) -> int:
    ts = (ts or "").strip()
    if re.fullmatch(r"\d+(\.\d+)?", ts):
        return int(float(ts))
    parts = [p for p in ts.split(":") if p != ""]
    if len(parts) == 3:
        h, m, s = parts
        return int(float(h) * 3600 + float(m) * 60 + float(s))
    if len(parts) == 2:
        m, s = parts
        return int(float(m) * 60 + float(s))
    if len(parts) == 1 and parts[0]:
        return int(float(parts[0]))
    return 0

def video_embed_html(src: str, start_seconds: int, height: int = 340) -> str:
    """Embed a <video> that seeks to start_seconds and stays paused until user clicks play."""
    vid = f"vid_{uuid.uuid4().hex}"
    s = max(0, int(start_seconds))
    return f"""
    <video id="{vid}" src="{src}" controls preload="auto" playsinline
           style="width:100%; height:{height}px; background:#000;"></video>
    <script>
      (function() {{
        const v = document.getElementById("{vid}");
        const target = {s};

        function seekAndPause() {{
          try {{
            if (!isNaN(v.duration)) {{
              v.currentTime = target;
              setTimeout(() => {{ if (!v.paused) v.pause(); }}, 0);
            }}
          }} catch (e) {{}}
        }}
        v.addEventListener('loadedmetadata', seekAndPause, {{ once: true }});
        v.addEventListener('canplay', () => {{ if (!v.paused) v.pause(); }});
        v.addEventListener('play', () => {{
          if (Math.abs(v.currentTime - target) > 0.5) {{
            v.currentTime = target;
          }}
        }});
      }})();
    </script>
    """

def render_timed_video(src: str, start_seconds: int, height: int = 340):
    html(video_embed_html(src, start_seconds, height=height), height=height + 12, scrolling=False)

# Store parameters/results in session so UI persists across re-runs
def ensure_state():
    ss = st.session_state
    ss.setdefault("project_id", "vertex-ai-search-v2")
    ss.setdefault("model", "gemini-2.5-flash")
    ss.setdefault("selected_uri", "")
    ss.setdefault("base_display_url", "")
    ss.setdefault("last_results", None)

# Callback to run detection and persist results
def run_detection():
    ss = st.session_state
    if not ss.selected_uri:
        st.warning("Please select a video or enter a custom URL.")
        return
    ss.base_display_url = gs_to_https(ss.selected_uri) if ss.selected_uri.startswith("gs://") else ss.selected_uri
    client = genai.Client(vertexai=True, project=ss.project_id, location="global")
    with st.spinner("Analyzing scene changes..."):
        results = scene_change_detection(ss.selected_uri, client, model=ss.model)
    if isinstance(results, list) and results:
        ss.last_results = results
    else:
        ss.last_results = None
        st.warning("No results returned.")

# ---------------- APP ----------------
st.set_page_config(page_title="Scene Change Detection", layout="wide")
st.title("🎬 Scene Change Detection (Gemini 2.5)")

ensure_state()

with st.sidebar:
    st.header("Settings")

    # Project ID (bound to session)
    st.text_input(
        "GCP Project ID",
        key="project_id",
        help="Used to initialize the Vertex Gemini client.",
    )

    # Model (bound to session)
    st.selectbox(
        "Model",
        options=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        key="model",
    )

    # Video selection
    gs_uris, https_urls, labels = list_public_mp4s("video_demo_test")
    if gs_uris:
        idx = st.selectbox(
            "Choose a video from GCS (public)",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
        )
        gs_choice = gs_uris[idx]
    else:
        st.warning("No .mp4 files found in bucket `video_demo_test`.")
        gs_choice = ""

    st.write("— or —")
    custom_url = st.text_input(
        "Custom video URL (HTTP(S) or gs://...)",
        placeholder="e.g., gs://video_demo_test/wakeup_princess.mp4 "
                    "or https://storage.googleapis.com/video_demo_test/...",
    )

    # Decide which to use and store in session (so callback sees it)
    st.session_state.selected_uri = custom_url.strip() if custom_url.strip() else gs_choice

    # Run button triggers callback that writes results into session
    st.button("Run detection", type="primary", use_container_width=True, on_click=run_detection)

# ---------- MAIN: Render results if present (persists across re-runs) ----------
if st.session_state.last_results:
    st.subheader("Results")
    for i, item in enumerate(st.session_state.last_results, start=1):
        ts = str(item.get("timestamp", "0")).strip()
        seconds = parse_timestamp_to_seconds(ts)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"**Scene {i} — start @ {ts}**")
            render_timed_video(st.session_state.base_display_url, seconds, height=340)
        with col2:
            st.markdown(f"**Scene {i} — metadata**")
            st.json(item)

    # Always render the download button when results exist (download triggers a rerun, but state persists)
    df = pd.DataFrame(st.session_state.last_results)
    if "timestamp" in df.columns:
        df = df[["timestamp"] + [c for c in df.columns if c != "timestamp"]]
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    base_name = (st.session_state.base_display_url or "results").split("/")[-1].replace(".mp4", "")
    st.sidebar.download_button(
        label="⬇️ Download results CSV",
        data=csv_bytes,
        file_name=f"{base_name}_scene_changes.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("Configure options in the sidebar and click **Run detection** to see results.")
