from pathlib import Path

import matplotlib.pyplot as plt
import torch

from baselines.vae_torch import VAESprites

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

if __name__ == "__main__":
    model_path = ROOT_DIR / "models" / "vae_torch" / "1708011778_vae_sprites.pt"
    vae = VAESprites()

    vae.load_state_dict(torch.load(model_path))

    image = vae.plot_grid_in_latent_space()

    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT_DIR / "static" / "vae_torch_latent_space_example.jpg", dpi=300)
    plt.show()
