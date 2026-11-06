# Experimental Characterization of LoRa Propagation and Machine Learning Approaches for Path Loss Modeling Across Diverse Hill Terrains in Nepal

**Peter Karki** · Coventry ID 17624929 · ST7072CEM Machine Learning · Softwarica College of IT and E-Commerce

Field measurements of 866 MHz LoRa links at 61 stops in three hill-terrain case studies near Kathmandu. At each stop, the instrument swept 6 spreading factors × 3 transmit powers × 5 packets (90 slots). The project compares regression and classification models under validation grouped by stop and leave-one-case-out.

**Main results**
- A depth-3 decision tree on spreading factor and power predicts delivery with a balanced accuracy of 0.68, and 0.67 on an unseen terrain.
- No path loss model beats the mean (RMSE 11.1 dB). The pooled exponent is n = 0.14; Case 1 alone gives n = 2.10.
- The cheapest working setting saves a median 97% of transmit energy compared with SF12 at 14 dBm.

## Repository structure
```
.
├── LoRa_ST7072CEM_Student.ipynb   analysis notebook (run this)
├── LoRa_data/                     raw field data
│   ├── raw/site_A..D_roamer_log.csv   one Roamer log per Station site
│   ├── stop_features.csv              distance, elevation, land cover per stop
│   ├── stations.csv                   Station positions
│   └── README.txt
├── dashboard/                     interactive Gradio dashboard
│   ├── app.py, loader.py, requirements.txt
│   ├── data/                      anonymised tables (used if LoRa_data/ is missing)
│   └── screenshots/
├── paper/
│   ├── PeterKarki_17624929.pdf    final paper (IEEE format)
│   ├── main.tex                   LaTeX source
│   └── figures/                   all figures and photos used in the paper
├── docs/DASHBOARD_ONE_SHOT_SPEC.md   full dashboard specification
└── requirements.txt
```

## How to run

### 1. Install
Python 3.10–3.13 is recommended.
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Notebook
Open `LoRa_ST7072CEM_Student.ipynb` in Jupyter, VS Code or Google Colab and choose **Run All** (about 3 minutes). It reads `LoRa_data/` with relative paths, so keep the notebook and the folder side by side.

In **Google Colab**, upload the notebook, zip the `LoRa_data` folder as `LoRa_data.zip`, and upload the zip to the Files panel. The first data cell unzips it.

### 3. Dashboard
```bash
cd dashboard
python app.py
```
Open http://127.0.0.1:7860.

Tabs:
- Overview
- Delivery explorer
- Path loss
- Settings advisor
- Model lab
- Dead-zone map
- Field map (local only)
- Results & method

### 4. Paper
`paper/PeterKarki_17624929.pdf` is the submitted paper. To rebuild it, upload `paper/` to Overleaf (or run `pdflatex main.tex` twice).

## Video
Screencast of the code and the dashboard: *add link here*.

## Data and privacy
The raw logs contain GPS positions of the measurement stops. The paper and the public dashboard views withhold site names and coordinates; the dashboard's Field map tab is meant for local use only.

## Acknowledgement
Generative AI tools assisted with code, figures and editing. The experiment, field work, analysis and conclusions are the author's own.
