import logging

import equinox as eqx
import helpers as h
import trainer
import yaml
from config import Config
from jax import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


def main(cfg):

    model_key, data_key, optim_key, loss_key, train_key = random.split(
        random.PRNGKey(cfg.train.seed), num=5
    )
    train_dl, test_dl = h.return_dataloader(cfg, key=data_key)
    optim = h.return_optim(cfg, key=optim_key)
    vae = h.return_vae(cfg, key=model_key)
    opt_state = optim.init(eqx.filter(vae, eqx.is_inexact_array))
    loss = h.return_loss(cfg, key=loss_key)

    t = trainer.Trainer(cfg)
    vae = t.train(vae, loss, optim, opt_state, train_dl, test_dl, key=train_key)


if __name__ == "__main__":
    cfg = Config()
    cfg_dict = cfg.to_dict()
    with open(cfg.train.config_path, "w") as file:
        yaml.dump(cfg_dict, file, allow_unicode=True)
    main(cfg)
