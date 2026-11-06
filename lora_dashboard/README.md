---
title: LoRa Hill-Terrain Explorer
emoji: 📡
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# LoRa Hill-Terrain Explorer

Interactive companion to the MSc class project paper *"Experimental Characterization of LoRa Propagation and Machine Learning Approaches for Path Loss Modeling Across Diverse Hill Terrains in Nepal"* (Peter Karki, Softwarica College / Coventry University).

**Tabs:** Overview · Delivery explorer · Path loss · Model results · Model lab (live grouped validation) · Dead-zone map (synthetic hill-valley terrain, isometric 3-D and 2-D) · Method & limits.

## Data
By default the app reads the raw logs from `LoRa_data/`, looking next to `app.py` or one folder up (set `LORA_DATA` to use another path). If that folder isn't found, it falls back to `data/`, which holds the anonymised field data: 61 complete stops, 5,490 packet slots and 30 lost locations. Site names and GPS coordinates are removed; only distances and elevation differences are kept.

## Run it

**Hugging Face Spaces (public link).** Create a new Space with the Gradio SDK, then upload `app.py`, `requirements.txt`, `README.md` and the `data/` folder.

**Google Colab.**
```python
!pip -q install gradio plotly
# upload lora_dashboard.zip using the Files panel, then:
!unzip -q -o lora_dashboard.zip -d /content
%cd /content/lora_dashboard
import app
app.demo.launch(share=True)
```

**Your laptop.**
```bash
pip install -r requirements.txt
python app.py        # then open http://127.0.0.1:7860
```

## Limits
This is a research prototype, not a planning tool. The dead-zone map applies a depth-3 tree trained on 61 stops to an invented terrain. The tree sees distance, elevation difference and angle, but not the terrain between the antennas. The line-of-sight layer is a separate geometric check.
