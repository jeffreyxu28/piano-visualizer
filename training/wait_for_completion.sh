#!/bin/bash
LOG="training/train_log.txt"
MAX_ITER=280   # 280 * 30s ≈ 140 minutes safety ceiling
i=0
while [ $i -lt $MAX_ITER ]; do
  if grep -q "Done. best_piano_model.pt is ready" "$LOG" 2>/dev/null; then
    echo "=== TRAINING COMPLETE ==="
    tail -60 "$LOG"
    exit 0
  fi
  if grep -qE "Traceback \(most recent call last\)|RuntimeError|FileNotFoundError|CUDA out of memory" "$LOG" 2>/dev/null; then
    echo "=== TRAINING ERROR DETECTED ==="
    tail -80 "$LOG"
    exit 1
  fi
  sleep 30
  i=$((i+1))
done
echo "=== TIMED OUT WAITING (140 min ceiling reached) ==="
tail -60 "$LOG"
exit 2
