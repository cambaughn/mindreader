#!/usr/bin/env bash
# Runs ON the pod (via run.sh). Installs the few Linux deps training needs; torch comes with the image.
set -euo pipefail
cd /workspace/mindreader
python3 - <<'PY'
import torch, sys
ok = torch.cuda.is_available()
print(f"torch {torch.__version__}  cuda {ok}  {torch.cuda.get_device_name(0) if ok else ''}")
sys.exit(0 if ok else 1)
PY
pip install -q "numpy>=2,<3" "sentencepiece==0.2.2" opencv-python-headless six packaging
PYTHONPATH=third_party/silent-dictation python3 -c "import silentdict.train, mindreader.evaluate; print('imports ok')"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
