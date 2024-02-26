from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


def convert_to_serializable(fields):
    """Recursively convert dataclass fields to a serializable format."""
    if isinstance(fields, dict):
        return {key: convert_to_serializable(value) for key, value in fields.items()}
    elif isinstance(fields, list):
        return [convert_to_serializable(value) for value in fields]
    elif isinstance(fields, Path):
        return str(fields)  # Convert Path to string
    return fields


d = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
SAVE_PATH = Path(__file__).parent.resolve() / "runs" / d
SAMPLE_PATH = SAVE_PATH / "samples"
SAMPLE_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH = SAVE_PATH / "plots"
PLOT_PATH.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = SAVE_PATH / "config" / "config.yaml"
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    shuffle_data: bool = True
    epochs: int = 500
    num_samples_plot: int = 100
    sample_path: Path = SAMPLE_PATH
    plot_path: Path = PLOT_PATH
    config_path: Path = CONFIG_PATH


@dataclass
class DatasetConfig:
    num_samples: int = 1000
    data_dim: tuple = (3, 16, 16)
    test_split: float = 0.2


@dataclass
class OptimizerConfig:
    lr: float = 1e-3


@dataclass
class VAEConfig:
    latent_dim: int = 2


@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)

    def to_dict(self) -> dict:
        """Convert the dataclass instance to a dictionary, with Path objects as strings."""
        return convert_to_serializable(asdict(self))
