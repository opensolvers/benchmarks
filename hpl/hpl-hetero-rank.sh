#!/bin/bash
NX=${HPL_NX100:-8}
ATH=${HPL_A100_THREADS:-1}
r=${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}
if (( r < NX )); then
  export OMP_NUM_THREADS=1
  export FLEXIBLAS="${HPL_X100_BLAS:-OPENBLAS_DUAL}"
  c=$(( r % 8 ))
  exec taskset -c "$c" "$@"
else
  export OMP_NUM_THREADS=$ATH
  export FLEXIBLAS="${HPL_A100_BLAS:-OPENBLAS_ZVL1024B}"
  echo $$ > /proc/set_ai_thread
  ar=$(( r - NX ))
  if (( ATH <= 1 )); then
    c=$(( 8 + (ar % 8) ))
    exec taskset -c "$c" "$@"
  else
    exec taskset -c 8-15 "$@"
  fi
fi
