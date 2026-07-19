#!/usr/bin/env bash
# validate_model.sh URL FILENAME LABEL PARAMS
# Downloads a Q4_0 GGUF (unless present), runs a coherence prompt via llama-cli
# on the IME build, benches pp128/tg32 on IME and the RVV control, appends one
# result row, then deletes the GGUF to respect the 7.7 GiB / no-swap board.
set -u
URL="$1"; FN="$2"; LABEL="$3"; PARAMS="$4"
MODELS=~/llama-x60/models
IME=~/x60-ime; RVV=~/x60-rvv
RESULTS=~/model_validation.tsv
LOG=~/mv_${LABEL}.log
M="$MODELS/$FN"
[ -f "$RESULTS" ] || echo -e "label\tparams\tsize_MiB\tcoherent\tuse_ime1\tpp128_ime\ttg32_ime\tpp128_rvv\ttg32_rvv" > "$RESULTS"

echo "==== $LABEL ($PARAMS) ====" | tee "$LOG"
if [ ! -f "$M" ]; then
  echo "-- downloading $FN" | tee -a "$LOG"
  if ! wget -q --show-progress -O "$M.part" "$URL" 2>>"$LOG"; then
    echo "DL_FAIL"; echo -e "${LABEL}\t${PARAMS}\tNA\tDL_FAIL\tNA\tNA\tNA\tNA\tNA" >> "$RESULTS"; rm -f "$M.part"; exit 0
  fi
  mv "$M.part" "$M"
fi
SZ=$(( $(stat -c%s "$M") / 1048576 ))
echo "-- size ${SZ} MiB; free:" | tee -a "$LOG"; free -h | head -2 | tee -a "$LOG"

# 1) coherence (IME) -- capture whether it emits >20 non-prompt chars, finite run
CLI=$(LD_LIBRARY_PATH=$IME taskset -c 0-7 $IME/llama-cli -m "$M" -t 8 -c 1024 -n 40 \
      -no-cnv -st --simple-io --no-display-prompt \
      -p "Explain what RISC-V vector extensions are, in two short sentences." 2>>"$LOG")
echo "-- cli output:" | tee -a "$LOG"; echo "$CLI" | tee -a "$LOG"
USEIME=$(grep -o "use_ime1: [01]" "$LOG" | tail -1 | awk '{print $2}')
OUTCHARS=$(printf "%s" "$CLI" | tr -d '[:space:]' | wc -c)
COH=$([ "$OUTCHARS" -gt 20 ] && echo yes || echo NO)

# 2) bench IME
bench(){ LD_LIBRARY_PATH=$1 taskset -c 0-7 $1/llama-bench -m "$M" -p 128 -n 32 -t 8 2>>"$LOG" \
         | awk -F'|' '/pp128/{n=split($8,a,"±");gsub(/ /,"",a[1]);pp=a[1]} /tg32/{n=split($8,a,"±");gsub(/ /,"",a[1]);tg=a[1]} END{print pp"\t"tg}'; }
echo "-- bench IME"  | tee -a "$LOG"; IMEB=$(bench "$IME")
echo "-- bench RVV"  | tee -a "$LOG"; RVVB=$(bench "$RVV")
echo -e "${LABEL}\t${PARAMS}\t${SZ}\t${COH}\t${USEIME:-?}\t${IMEB}\t${RVVB}" | tee -a "$RESULTS"

# 3) reclaim RAM/disk unless it's the pre-existing 0.5b baseline (keep that one)
[ "$FN" = "qwen2.5-0.5b-instruct-q4_0.gguf" ] || rm -f "$M"
echo "MV_DONE $LABEL"
