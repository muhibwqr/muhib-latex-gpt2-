#!/bin/sh
# Compile the demo. All GPT-2 math runs inside pdflatex; these only enlarge TeX's memory tables.
cd "$(dirname "$0")"
exec env save_size=5000000 extra_mem_top=20000000 pdflatex -no-shell-escape -interaction=nonstopmode "${1:-muhib.tex}"
