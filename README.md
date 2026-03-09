# Automatisierung KI-basierte Analyse von Schwingungsspektren (PHM 2010)

Dieses Projekt automatisiert die KI-basierte Auswertung von Vibrationsdaten aus dem IoT-/Predictive-Maintenance-Umfeld:

**CSV (Vibration) → STFT-Spektrogramme → CNN → Nudging (🟢/🔴)**

Datengrundlage: [PHM Data Challenge 2010](https://www.kaggle.com/datasets/rabahba/phm-data-challenge-2010) (Kaggle).

## Ziel-Output (Nudging)
- 🟢 **OK**: Klasse `uniform` – normaler Verschleiß
- 🔴 **DEFECT**: Klassen `severe` oder `rapid` – kritischer Verschleiß, Wartung empfohlen

## Pipeline-Übersicht

```
data/raw/*.csv
    └─► STFT-Spektrogramm (3 Kanäle: vx, vy, vz → RGB)
            └─► SimpleCNN (3× Conv + GlobalAvgPool + FC)
                    └─► Nudging: OK / DEFECT
```

## Schnellstart (One-Command)

```bash
export KAGGLE_TOKEN="dein_kaggle_api_token"
bash run_pipeline.sh
```

Das Script erledigt automatisch:
1. Pakete installieren
2. PHM2010-Datensatz herunterladen (via kagglehub)
3. CNN-Modell trainieren (10 Epochen)
4. Inference auf Testdaten ausführen
5. Report & Ergebnisse speichern

## Manuelles Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt kagglehub
```

### Training
```bash
export KAGGLE_TOKEN="dein_token"
python3 -m src.cli train --out models/stft_cnn.pt --epochs 10
```

### Inference
```bash
python3 -m src.cli predict --model models/stft_cnn.pt --split test
```

## Ergebnisse

Nach Inference werden folgende Dateien unter `results/` gespeichert:

| Datei | Inhalt |
|---|---|
| `predictions.csv` | Klassenwahrscheinlichkeiten + Nudge je Messung |
| `summary.json` | Gesamtstatistik (OK/DEFECT-Verteilung) |
| `report.md` | Lesbarer Kurzbericht |
| `part_summary.csv` | Nudge-Entscheidung aggregiert je Cutter |

**Beispielergebnis (Testset c2, c3, c5):**
- Gesamtdateien: 1890
- 🟢 OK: 596 (31.5%)
- 🔴 DEFECT: 1294 (68.5%)

## Projektstruktur

```
project/
├── src/
│   ├── cli.py            # Einstiegspunkt (train / predict)
│   ├── config.py         # Hyperparameter & Pfade
│   ├── data_download.py  # Kaggle-Download via kagglehub
│   ├── data_load.py      # CSV-Laden & Signalextraktion
│   ├── features_stft.py  # STFT → RGB-Spektrogramm
│   ├── model_cnn.py      # SimpleCNN-Architektur
│   ├── train.py          # Training-Loop & Label-Vergabe
│   ├── inference.py      # Inference + Nudging + Report
│   └── decision.py       # Nudging-Logik (OK / DEFECT)
├── results/              # Generierte Ausgaben
├── run_pipeline.sh       # One-Command-Script
└── requirements.txt
```
