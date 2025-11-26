#!/usr/bin/env bash
# Run continual learning with all 3 methods and compare

echo "=================================="
echo "CONTINUAL LEARNING DEMO"
echo "=================================="

echo ""
echo "[1/3] Training with Finetune (baseline)..."
python3 main.py --mode continual --method finetune --epochs 2 --batch 128

echo ""
echo "[2/3] Training with Experience Replay..."
python3 main.py --mode continual --method ER --epochs 2 --batch 128

echo ""
echo "[3/3] Training with EWC..."
python3 main.py --mode continual --method EWC --epochs 2 --batch 128

echo ""
echo "=================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=================================="
