#!/bin/bash
# Finish the ru_stop v3 run: negatives, then a validation set that carries the
# same channel as training, then train and export.
#
# Launch with setsid, not just nohup. The first attempt was nohup'd and still
# died mid-run when the controlling session went away: nohup only ignores
# SIGHUP, and a process group killed outright takes the job with it. setsid
# puts this in its own session so nothing upstream can reach it.
#
#   setsid ./run_v3_chain.sh < /dev/null > /dev/null 2>&1 &
#
# positive_train.npy from the earlier attempt is kept and reused -- the
# features phase skips arrays that already exist, so this resumes rather than
# restarting three hours of work. positive_val.npy is deliberately removed
# first: the copy on disk was built without augmentation, and training stops on
# val_loss with restore_best_weights, so a clean-synthesis validation set would
# select for exactly the failure being fixed.
set -u
cd /home/kelstar/projects/smart-home/voice/wakework || exit 1

F=training/output/ru_stop/mww_features
log() { echo "$(date +%H:%M:%S) $*" >> log_ru_stop_v3_chain.txt; }

log "=== chain start (pid $$) ==="

if [ ! -s "$F/positive_train.npy" ]; then
  log "positive_train.npy missing - would have to redo positives; stopping"
  exit 1
fi

if [ -s "$F/positive_val.npy" ]; then
  log "removing unaugmented positive_val.npy so it is rebuilt through the channel"
  rm -f "$F/positive_val.npy"
fi

log "features (positives are reused; negatives and validation are built now)"
./.venv/bin/python train_mww.py --config ru_stop.yaml --phase features \
    >> log_ru_stop_v3_features2.txt 2>&1
for n in positive_train positive_val negative_train negative_val; do
  if [ ! -s "$F/$n.npy" ]; then
    log "MISSING $n.npy after features - not training"
    exit 1
  fi
done
ls -la "$F"/*.npy >> log_ru_stop_v3_chain.txt

log "training"
./.venv/bin/python train_mww.py --config ru_stop.yaml --phase train \
    >> log_ru_stop_v3_train.txt 2>&1 || { log "train failed"; exit 1; }

log "exporting"
./.venv/bin/python train_mww.py --config ru_stop.yaml --phase export \
    >> log_ru_stop_v3_train.txt 2>&1 || { log "export failed"; exit 1; }

log "CHAIN DONE"
