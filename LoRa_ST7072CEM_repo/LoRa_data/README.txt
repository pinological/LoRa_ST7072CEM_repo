LoRa hill-terrain field data (ST7072CEM, Peter Karki)

raw/site_X_roamer_log.csv  One file per Station site (A-D). One row per packet slot logged by the Roamer
                           (status OK or TIMEOUT), plus one row per failed handshake (status NOACK).
stop_features.csv          One row per complete stop: distance, antenna heights, SRTM elevations,
                           elevation difference and angle, land cover and manual line-of-sight label.
stations.csv               Station position for each site.

Cases: Case 1 = Site A, Case 2 = Site B, Case 3 = Sites C and D.
These files contain GPS coordinates; do not share them publicly.
