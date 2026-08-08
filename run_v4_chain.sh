#!/bin/bash
# ru_stop v4: same as v3 plus real recordings of this room simply being a room.
#
# v3 hears the user — 22 of 24 held-out utterances against v2's 0 — but also
# fires on ordinary household noise, which a speech recogniser confirms contains
# no speech at all. Its negatives cover generic corpora and the near-silence the
# speaker leaves while playing; what they do not cover is this room's own
# background. That is the single change here: nothing else moves, so whatever
# the acceptance test says can be attributed to it.
#
# Positives are reused untouched (15 GB already on disk, ~3 h of work), so this
# is negatives plus training.
#
#   setsid ./run_v4_chain.sh < /dev/null > /dev/null 2>&1 &
set -u
cd /home/kelstar/projects/smart-home/voice/wakework || exit 1

F=training/output/ru_stop/mww_features
LOG=log_ru_stop_v4_chain.txt
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }

log "=== v4 chain start (pid $$) ==="

if [ ! -s "$F/positive_train.npy" ] || [ ! -s "$F/positive_val.npy" ]; then
  log "positives missing - refusing to rebuild them here"
  exit 1
fi

# Keep v3 so the two can be compared, and so there is something to fall back to.
if [ -f ru_stop_mww.tflite ] && [ ! -f ru_stop_v3_mww.tflite ]; then
  cp ru_stop_mww.tflite ru_stop_v3_mww.tflite
  cp ru_stop_mww.json   ru_stop_v3_mww.json
  log "kept v3 as ru_stop_v3_mww.*"
fi

log "dropping negative arrays so they are rebuilt with the room clips"
rm -f "$F/negative_train.npy" "$F/negative_val.npy"

log "features (negatives only)"
./.venv/bin/python train_mww.py --config ru_stop.yaml --phase features \
    >> log_ru_stop_v4_features.txt 2>&1
for n in positive_train positive_val negative_train negative_val; do
  if [ ! -s "$F/$n.npy" ]; then
    log "MISSING $n.npy after features - not training"
    exit 1
  fi
done
ls -la "$F"/*.npy >> "$LOG"

log "training"
./.venv/bin/python train_mww.py --config ru_stop.yaml --phase train \
    >> log_ru_stop_v4_train.txt 2>&1 || { log "train failed"; exit 1; }

log "exporting"
./.venv/bin/python train_mww.py --config ru_stop.yaml --phase export \
    >> log_ru_stop_v4_train.txt 2>&1 || { log "export failed"; exit 1; }

log "CHAIN DONE"
