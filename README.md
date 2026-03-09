# Automatisierung KI-basierte Analyse von Schwingungsspektren (PHM 2010)

Dieses Projekt automatisiert die in der PDF beschriebene KI-basierte Auswertung:
**CSV (Vibration) → STFT-Spektrogramme → CNN → Nudging (🟢/🔴)**

Datengrundlage: PHM Data Challenge 2010 (Kaggle).

## Ziel-Output (Nudging)
- 🟢 GREEN / OK: Klasse `uniform`
- 🔴 RED / DEFECT: Klassen `severe` oder `rapid`

## Setup
```bash
pip install -r requirements.txt
