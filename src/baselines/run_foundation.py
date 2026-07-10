import hydra
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from src.baselines.foundation_runner import run_foundation_eval, _save_forecast_plot
from src.baselines.adapters import get_adapter


@hydra.main(config_path="../../configs", config_name="config_foundation", version_base=None)
def main(cfg):
    adapter = get_adapter(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dump the full resolved config (all settings for this run)
    OmegaConf.save(cfg, output_dir / "config_resolved.yaml")

    run_foundation_eval(cfg, adapter, output_dir, save_plot_fn=_save_forecast_plot)


if __name__ == "__main__":
    main()