#!/bin/bash
# =============================================================
# run_pipeline.sh  –  Schritte 1–3 automatisch ausführen
# Ausführen im Terminal aus dem Ordner: project/
#   bash run_pipeline.sh
# =============================================================

set -e  # Bricht bei Fehler sofort ab

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "================================================"
echo " Schritt 0: Umgebung prüfen & Pakete installieren"
echo "================================================"

# Virtuelle Umgebung aktivieren (falls vorhanden)
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ .venv aktiviert"
else
    echo "⚠ Kein .venv gefunden – nutze System-Python"
fi

# Fehlende Pakete nachinstallieren
pip install -r requirements.txt kagglehub -q
echo "✓ Pakete installiert"

echo ""
echo "================================================"
echo " Schritt 1: Dateinamen prüfen (Label-Heuristik)"
echo "================================================"

python3 - <<'PYCHECK'
import sys
from pathlib import Path

# Prüft: wie heißen die CSV-Dateien im Datensatz?
data_dir = Path("data/raw")
csvs = list(data_dir.rglob("*.csv"))

if not csvs:
    print("  ⚠ Noch keine Daten – werden beim Training heruntergeladen.")
else:
    print(f"  Gefundene CSVs: {len(csvs)}")
    for f in sorted(csvs)[:5]:
        print(f"    Beispiel: {f.name}")
    # Prüfe ob Klassenname im Dateinamen steht
    classes = ("uniform", "severe", "rapid")
    has_classname = any(cls in f.stem.lower() for f in csvs for cls in classes)
    if has_classname:
        print("  ✓ Klassenname ist direkt im Dateinamen – Labels werden korrekt erkannt.")
    else:
        print("  ✓ Label-Vergabe via Dateiindex (Heuristik: <100=uniform, 100-199=severe, ≥200=rapid)")
PYCHECK

echo ""
echo "================================================"
echo " Schritt 2: Modell neu trainieren"
echo "================================================"

python3 -m src.cli train \
    --out models/stft_cnn.pt \
    --epochs 10 \
    --batch-size 16

echo "✓ Modell gespeichert unter: models/stft_cnn.pt"

echo ""
echo "================================================"
echo " Schritt 3: Inference auf Testdaten"
echo "================================================"

python3 -m src.cli predict \
    --model models/stft_cnn.pt \
    --split test

echo ""
echo "================================================"
echo " Ergebnis:"
echo "================================================"
python3 - <<'PYRESULT'
import json
from pathlib import Path

summary_path = Path("results/summary.json")
if summary_path.exists():
    with open(summary_path) as f:
        s = json.load(f)
    print(f"  Gesamtdateien: {s['total_predicted']}")
    print(f"  🟢 OK:         {s['ok']}  ({s['ok_rate']:.1%})")
    print(f"  🔴 DEFECT:     {s['defect']}  ({1-s['ok_rate']:.1%})")
    print(f"  Übersprungen:  {s['skipped_files']}")
    if s['ok_rate'] == 1.0:
        print("\n  ⚠ Warnung: 100% OK-Rate – Label-Funktion nochmals prüfen!")
    else:
        print("\n  ✓ Plausible Verteilung – Modell scheint zu funktionieren.")
else:
    print("  Keine summary.json gefunden.")
PYRESULT

echo ""
echo "✅ Pipeline komplett. Ergebnisse in: results/"
