#!/bin/bash
# One readable line per half-minute, so a multi-hour run is visibly alive.
#
#   tail -f voice/wakework/PROGRESS.txt
#
# The training logs are tqdm carriage-return spam interleaved with TensorFlow
# warnings, which is unreadable and gives no sense of whether anything is
# stuck. This pulls out the one thing that matters — what stage, how far, how
# long left — and says plainly when a stage ends.
set -u
cd "$(dirname "$0")" || exit 1

CHAIN_PID=${1:?usage: progress.sh <chain-pid>}
OUT=PROGRESS.txt

last=""
while kill -0 "$CHAIN_PID" 2>/dev/null; do
  line=""
  # Whichever run's log was written to most recently — hardcoding a version
  # meant the indicator happily reported a finished run's last line while a
  # new one was underway.
  newest=$(ls -t log_ru_stop_v*_features*.txt log_ru_stop_v*_train.txt 2>/dev/null | head -1)
  if [ -n "$newest" ]; then
    # Last tqdm bar in the file: stage name, percent, elapsed<remaining.
    bar=$(tail -c 4000 "$newest" 2>/dev/null | tr '\r' '\n' | grep -a "|" | tail -1)
    [ -n "$bar" ] && line="$bar"
    case "$newest" in
      *_train.txt)
        # Keras prints epochs rather than tqdm bars; those matter more here.
        ep=$(grep -aE "^Epoch |val_loss" "$newest" 2>/dev/null | tail -1)
        [ -n "$ep" ] && line="$ep"
        ;;
    esac
  fi

  stage=$(echo "$line" | sed -n 's/^\[\([a-z_]*\)\].*/\1/p')
  pct=$(echo "$line" | grep -o '[0-9]\+%' | head -1)
  eta=$(echo "$line" | grep -o '\[[0-9:]*<[0-9:?]*' | tr -d '[' | head -1)

  if [ -n "$stage" ]; then
    msg="$stage  ${pct:-?}  ${eta:-}"
  else
    msg="${line:-working}"
  fi

  if [ "$msg" != "$last" ]; then
    echo "$(date +%H:%M:%S)  $msg" >> "$OUT"
    last="$msg"
  fi
  sleep 30
done

echo "$(date +%H:%M:%S)  chain finished — see log_ru_stop_v3_chain.txt" >> "$OUT"
