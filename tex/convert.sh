#!/bin/sh
# Runs convert.tex's jobs in N parallel pdflatex processes (default: all cores).
# All the work is TeX; this script only starts the processes.   usage: tex/convert.sh [outdir/]
cd "$(dirname "$0")" || exit 1
PATH=$PATH:/Library/TeX/texbin; export PATH
OUT=${1:-w/}; export OUT
N=${N:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
mkdir -p "$OUT"
t0=$(date +%s)
{ echo tok
  for L in 0 1 2 3 4 5 6 7 8 9 10 11; do echo "fc $L"; echo "mproj $L"; done
  for L in 0 1 2 3 4 5 6 7 8 9 10 11; do echo "qkv $L"; done
  for k in 0 1 2 3 4 5 6 7; do echo "head $k"; done
  for a in 0 25 50 75 100 125 150 175; do b=$((a+25)); [ $b -gt 197 ] && b=197; echo "wte $a $b"; done
  for L in 0 1 2 3 4 5 6 7 8 9 10 11; do echo "aproj $L"; echo "ln $L"; done
  echo misc
} | xargs -P "$N" -L 1 sh -c '
  j=$1; shift; a=""; n=cvj-$j; for x; do a="$a{$x}"; n=$n-$x; done
  max_print_line=1000000 pdflatex -no-shell-escape -interaction=batchmode -jobname=$n \
    "\\def\\cvout{$OUT}\\def\\cvjobs{\\cvjob{$j}$a}\\input convert" >/dev/null 2>&1
  if grep -aq "^!" $n.log || ! grep -aq "convert done" $n.log; then echo "FAILED: $n (see tex/$n.log)"; exit 255; fi
  echo "  $(grep -ao "\[[a-z0-9]* [0-9]*s\]" $n.log | tr "\n" " ")"; rm -f $n.log' sh || exit 1
echo "done in $(( $(date +%s) - t0 )) s with $N processes: $(ls "$OUT" | wc -l | tr -d " ") files in $OUT"
