import matplotlib.pyplot as plt
import torchvision


def plot_grid(batch, plot_title, save_path, figsize=(15, 0.75 * 15), nrow=8):
    """
    Plot a grid of images

    Args:
        batch: torch.Tensor of shape (B, C, H, W)
        plot_title: str
        save_path: str
        figsize: tuple
        nrow: int
            Number of images in each row
    """

    fig, axs = plt.subplots(figsize=figsize)
    print(axs)
    grid = torchvision.utils.make_grid(batch, nrow=nrow)
    axs.imshow(grid.permute(1, 2, 0))
    axs.set_title(plot_title)
    axs.axis("off")
    fig.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close(fig)


def plot_process(batch, plot_title, save_path, figsize=(15, 0.75 * 15)):
    """
    Plot a batch of sequence of images. To be used when you want to visualize the sampling/corruption process.

    Args:
        batch: torch.Tensor of shape (B, T, C, H, W)
        plot_title: str
        save_path: str
        figsize: tuple
    """
    shape = batch.shape
    batch = batch.reshape(shape[0] * shape[1], *shape[2:])
    nrows = shape[1]
    plot_grid(batch, plot_title, save_path, figsize, nrow=nrows)


def plot_losses(train_loss, test_loss=None, plot_title="Losses", save_path=None):
    """
    Plot the training and test losses

    Args:
        train_loss: list
            List of training losses
        test_loss: list
            List of test losses
        plot_title: str
        save_path: str
    """
    fig, ax = plt.subplots(figsize=(15, 0.75 * 15))
    ax.plot(train_loss, label="Train loss", color="red")
    if test_loss is not None:
        ax.plot(test_loss, label="Test loss", color="blue")
    ax.set_title(plot_title)
    ax.set_xlabel("Epochs")
    ax.legend()
    fig.savefig(save_path, dpi=500, bbox_inches="tight")
