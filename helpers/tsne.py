import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize

def draw_tsne(
    X,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    cmap: str = "viridis",
    figsize=(8, 6),
    point_size: int = 20,
    annotate: bool = False,
    save_path: str = './tsne.pdf',
):
    # convert to numpy
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    if X_np.ndim != 2:
        raise ValueError(f"X must be 2D (M,N), got shape {X_np.shape}")

    M = X_np.shape[0]
    if M == 0:
        raise ValueError("X contains zero instances")

    # compute tsne
    tsne = TSNE(n_components=2, perplexity=min(perplexity, max(2, M-1)),
                max_iter=n_iter, random_state=random_state, init="pca")
    Y = tsne.fit_transform(X_np)  # (M,2)

    # colors by index mapped to colormap
    indices = np.arange(M)
    if M == 1:
        norm_vals = np.array([0.0])
    else:
        norm_vals = indices.astype(float) / (M - 1)  # in [0,1]
        # norm_vals = (indices // 90) / (indices.max() // 90)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(1-norm_vals)

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off') 
    point_size = np.array([20] * M)
    point_size[[0, 45, 90, 135, 180, 225, 270, 315]] = 180
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=colors, s=point_size, edgecolors="none")
    # ax.set_xlabel("t-SNE 1")
    # ax.set_ylabel("t-SNE 2")
    # ax.set_title("t-SNE of latents")

    # optional annotate
    if annotate:
        for i, (x, y) in enumerate(Y):
            ax.text(x, y, str(i), fontsize=6, color="black", ha="center", va="center")

    # colorbar showing index mapping
    # sm = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=max(0, M - 1)), cmap=cmap_obj)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax)
    # cbar.set_label("instance index")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", transparent=True, pad_inches=0,)

    return fig, ax, Y


# sample = np.arange(0, 360)[:, None].repeat(3, 1)
# draw_tsne(sample)

sample = np.load('/workspaces/diffusion/helpers/tsne.npz')
sample_middle_layer = sample['middle']
sample_last_layer = sample['last']
draw_tsne(sample_middle_layer, save_path='/workspaces/diffusion/helpers/tsne.svg')