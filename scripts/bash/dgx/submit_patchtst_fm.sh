#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=a100
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=tsfm
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

#!/bin/bash
#!/bin/bash
#!/bin/bash
set -euo pipefail

mkdir -p jobs

VENV_DIR=".venv-patchtst-fm"
PYTHON_BIN="$VENV_DIR/bin/python"
TSFM_ZIP="granite-tsfm-main.zip"
TSFM_DIR="granite-tsfm-main"
TSFM_ABS_DIR="$(pwd)/$TSFM_DIR"

if [ ! -x "$PYTHON_BIN" ] || ! PYTHONPATH="$TSFM_ABS_DIR" "$PYTHON_BIN" -c "import hydra, tsfm_public, tsfm_public.toolkit" 2>/dev/null; then
    echo "Setting up $VENV_DIR (missing or incomplete)..."
    rm -rf "$VENV_DIR"
    uv venv "$VENV_DIR"

    if [ ! -d "$TSFM_DIR" ]; then
        if [ ! -f "$TSFM_ZIP" ]; then
            echo "ERROR: $TSFM_ZIP not found."
            exit 1
        fi
        unzip -q "$TSFM_ZIP"
    fi

    # Install runtime deps directly (skip installing granite-tsfm itself — its
    # static `packages = ["tsfm_public", "tsfmhfdemos"]` list in pyproject.toml
    # doesn't include subpackages like tsfm_public.toolkit, so the built wheel
    # is broken). We expose the source tree via PYTHONPATH instead.
    uv pip install --python "$PYTHON_BIN" \
        "transformers[torch]>=4.57.6,<5" \
        datasets deprecated "urllib3>=1.26.19" "numpy<3" \
        "torch>=2.10,<2.11" "scikit-learn<2.0.0" "pandas>=2.3.3,<4" \
        "filelock>=3.20.3" "einops>=0.7" \
        hydra-core omegaconf matplotlib
fi

PYTHONPATH="$TSFM_ABS_DIR" srun "$PYTHON_BIN" -u -m src.baselines.run_foundation --multirun \
    model=patchtst_fm \
    dataset=cer_bis \
    model.probabilistic=true \
    dataset.context_length=144,336,672,1440,2500 \
    dataset.prediction_length=96