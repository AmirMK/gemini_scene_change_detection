# 🎬 Scene Change Detection App

This is a Streamlit app that uses **Google’s Gemini models** to analyze videos stored in a Google Cloud Storage bucket and detect **scene changes** (shot boundaries).  
The app identifies the **best points for ad placement** across a movie or video and returns detailed metadata (timestamp, reason, transition type, narrative type, etc.).  

![Demo](app.gif)

On the UI:
- **Sidebar** lets you select:
  - GCP project ID (default: `vertex-ai-search-v2`)
  - Gemini model (`gemini-2.5-flash`, `gemini-2.5-flash-lite`, or `gemini-2.5-pro`)
  - Video file from a public GCS bucket or a custom URL
- **Main area** displays results:
  - Each scene change is shown in a row with:
    - Left: the video starting from the detected timestamp (paused until user plays)
    - Right: structured JSON metadata for that scene
- A **Download CSV** button in the sidebar lets you export all results.

---

## 🚀 Run Locally

### 1. Clone this repo
```bash
git clone <your-repo-url>
cd scene_detection
```

### 2. Install dependencies
Make sure you have Python **3.9+** installed. Then, from the project folder, run:

```bash
pip install -r requirements.txt
```

### 3. Authenticate with Google Cloud
The app needs credentials to call Gemini on Vertex AI and access videos in GCS.  
Run the following command and follow the login steps in your browser:

```bash
gcloud auth application-default login
```

### 4. Launch the app
Start the Streamlit server by running:

```bash
streamlit run app.py
```

### 5. Open in browser
After starting the server, Streamlit will show a local URL such as:

http://localhost:8501

Open this link in your web browser to access the app.
