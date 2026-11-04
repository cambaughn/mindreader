#!/usr/bin/env bash
# Runs ON the pod (via run.sh). Installs the few Linux deps training needs (torch comes with the
# image) and downloads + converts the FLARE weights there, with the same checksums as the engine's setup.sh.
set -euo pipefail
cd /workspace/mindreader
python3 - <<'PY'
import torch, sys
ok = torch.cuda.is_available()
print(f"torch {torch.__version__}  cuda {ok}  {torch.cuda.get_device_name(0) if ok else ''}")
sys.exit(0 if ok else 1)
PY
pip install -q "numpy>=2,<3" "sentencepiece==0.2.2" opencv-python-headless six packaging

cd third_party/silent-dictation
mkdir -p models/flare_large
fetch() {  # url destination sha256 (resumes; skips when already present and correct)
  local url="$1" dest="$2" sum="$3"
  if [[ -f "$dest" && "$(sha256sum "$dest" | cut -d' ' -f1)" == "$sum" ]]; then echo "ok     $dest"; return; fi
  echo "fetch  $dest"
  curl -fL --retry 3 -C - -sS -o "$dest.part" "$url"
  [[ "$(sha256sum "$dest.part" | cut -d' ' -f1)" == "$sum" ]] || { echo "checksum mismatch for $dest" >&2; exit 1; }
  mv "$dest.part" "$dest"
}
AVSR_REV=182b62837773ab01052d4ac21ef1d2203ea7d267
fetch "https://raw.githubusercontent.com/mpc001/auto_avsr/$AVSR_REV/spm/unigram/unigram5000.model" \
  models/unigram5000.model 2c1d648ccf5fdce6612ecbea2ffbd1cab5aabc90458781186175c3911c4bdb1e
fetch "https://raw.githubusercontent.com/mpc001/auto_avsr/$AVSR_REV/spm/unigram/unigram5000_units.txt" \
  models/unigram5000_units.txt 29001d967ad290e28f6f1d7d47e07e314354bf67d3b1f494e936fa2fed9a9122
if [[ ! -f models/flare_large/model.pt ]]; then
  FLARE_REV=6187ad650dc1786a8efc8b6bfbee37bbf62c4b2e
  fetch "https://huggingface.co/Anony94/FLARE-checkpoints/resolve/$FLARE_REV/exp_22_2_large/finetuned/checkpoints/checkpoint_last.pt" \
    models/flare_large/checkpoint_last.pt ceaaf364ab47cda05dddc05dc109752cede57c989d7f9481b765e29e227cbf25
  PYTHONPATH="$PWD" python3 -m silentdict.flare.convert models/flare_large/checkpoint_last.pt models/flare_large/model.pt
  rm -f models/flare_large/checkpoint_last.pt
fi
cd /workspace/mindreader
PYTHONPATH=third_party/silent-dictation python3 -c "import silentdict.train, mindreader.evaluate; print('imports ok')"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
