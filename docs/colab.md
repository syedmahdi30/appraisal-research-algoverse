# Running on Colab (A100) via the VS Code Colab extension

The primary experiments (Stage A/C) need an NVIDIA GPU. We use a Colab **A100** runtime
connected to VS Code. Your local files sync to the runtime through the extension, so the
repo is already present at the working directory — no GitHub clone needed.

## The one thing to remember: runtimes are ephemeral
Colab gives you a fresh machine each session and wipes it when it recycles (idle timeout
or session end). That means every session you must re-install packages and re-set the
token, and anything you want to KEEP (probes, metrics) must live on Google Drive.
`scripts/colab_bootstrap.py` handles all of this.

## One-time setup
1. **HF_TOKEN in Colab Secrets** — click the 🔑 key icon in the Colab sidebar → Add new
   secret → name `HF_TOKEN`, value = your token, toggle **Notebook access** ON.
   (Never paste the token into code.)
2. **Accept the Gemma license** once on the Hub: https://huggingface.co/google/gemma-3-4b-it

## Each session

IMPORTANT: `drive.mount` and `userdata.get` (Colab Secrets) ONLY work in a notebook CELL —
they crash inside a `!python` subprocess (`'NoneType' object has no attribute 'kernel'`).
So mount Drive and load the token in a CELL first; every later `!python` inherits both.

```python
# Cell 1 — clone + enter the repo (%cd with %, so `import src` works)
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>

# Cell 2 — mount Drive + load token (a CELL, needs the kernel)
from google.colab import userdata, drive
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")   # token from the Colab key icon
drive.mount("/content/drive")
```
```bash
# Cell 3 — deps + persist small dirs to Drive (inherits HF_TOKEN + the mounted drive)
!python scripts/colab_bootstrap.py --drive

# Cell 4 — prove the model path works (boots Gemma, one forward pass, hooks fire)
!python scripts/smoke_test.py

# Cell 5 — data (crowd-enVENT free; EMOTIC gated, from your Drive)
!python scripts/download_data.py --dataset crowd-envent

# Cell 6 — run a stage
!python -m src.experiments.stage_a_text --config config/stage_a.yaml
```

With `--drive`, only the SMALL persistent dirs are symlinked into
`/content/drive/MyDrive/algoverse-appraisal/`: `results/` and `data/processed/`. Raw data
(`data/raw/`) stays on Colab's fast local disk — reading large image sets from mounted
Drive is slow, so we keep the source archive in Drive and extract it locally each session.

## EMOTIC (gated, zip in your MyDrive)
The EMOTIC zip lives in `MyDrive`. Keep it there (persistent) and extract to fast local
disk each session:

```bash
# extract the Drive zip into data/raw/emotic/ (local, fast) — swap in your filename
!python scripts/download_data.py --dataset emotic --archive /content/drive/MyDrive/EMOTIC.zip
```

To inspect the archive's layout WITHOUT extracting (useful before wiring the loader):

```bash
!unzip -l /content/drive/MyDrive/EMOTIC.zip | head -40
```

## Notes
- The A100 has 40 GB VRAM; Gemma-3-4B uses ~8 GB, so there's ample headroom.
- `colab_bootstrap.py` keeps Colab's CUDA-matched torch (requirements pin only a lower
  bound), so pip won't swap torch and break CUDA.
- If `smoke_test.py` fails on `is_multimodal`, check the installed `transformer-lens`
  version (needs >=3.2.1 for the Gemma3 multimodal hotfix) and re-run.
