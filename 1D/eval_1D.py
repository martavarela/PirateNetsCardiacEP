import os

import ml_collections

import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from jaxpi.utils import restore_checkpoint

import models_1D_data_only as models
from utils_1D import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # u_ref, t_star, x_star, y_star, coords = get_dataset("2DAPplanar.mat")
    coords, t_star, x_star, u_ref, _ = get_dataset("1DAPplanar.mat")
    u0 = u_ref[0, :]

    # Restore model
    model = models.AlievPanfilov1D(config, u0, u_ref, t_star, x_star, coords)
    ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt")
    ckpt_path = os.path.abspath(ckpt_path)
    print(f"Restoring model from: {ckpt_path}")
    model.state = restore_checkpoint(model.state, ckpt_path, step=50000)
    params = model.state.params

    # Compute L2 error
    l2_error = model.compute_l2_error(params, u_ref)
    print("L2 relative error: {:.3e}".format(l2_error))

    u_pred = model.V_pred_fn(params, coords[:,0], coords[:,1])
    u_pred = u_pred.reshape(u_ref.shape)
    
    T, X = np.meshgrid(t_star, x_star)

    # plot
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(T, X, u_ref.T, shading='nearest', cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(T, X, u_pred, shading='nearest', cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(T, X, jnp.abs(u_ref.T - u_pred), shading='nearest', cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "V_t_x.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    # # Generate a video 
    # def save_2d_video(u_ref, u_pred, t_star, x_star, y_star, workdir, config):
    #     save_dir = os.path.join(workdir, "videos", config.wandb.name)
    #     os.makedirs(save_dir, exist_ok=True)
    #     video_path = os.path.join(save_dir, "dynamics.mp4")

    #     # Create figure
    #     fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    #     # Initialize plots
    #     vmin, vmax = np.min(u_ref), np.max(u_ref)

    #     im1 = ax[0].imshow(u_ref[0], extent=[y_star.min(), x_star.min(), x_star.max()],
    #                         origin="lower", cmap="jet", vmin=vmin, vmax=vmax)
    #     ax[0].set_title("Exact")
    #     ax[0].set_xlabel("y")
    #     ax[0].set_ylabel("x")

    #     im2 = ax[1].imshow(u_pred[0], extent=[y_star.min(), x_star.min(), x_star.max()],
    #                         origin="lower", cmap="jet", vmin=vmin, vmax=vmax)
    #     ax[1].set_title("Predicted")
    #     ax[1].set_xlabel("y")

    #     im3 = ax[2].imshow(np.abs(u_ref[0] - u_pred[0]), extent=[ x_star.min(), x_star.max()],
    #                         origin="lower", cmap="jet")
    #     ax[2].set_title("Absolute Error")
    #     ax[2].set_xlabel("y")

    #     plt.tight_layout()
    #     fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    #     fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    #     fig.colorbar(im3, ax=ax[2], fraction=0.046, pad=0.04)

    #     def update(frame):
    #         im1.set_array(u_ref[frame])
    #         im2.set_array(u_pred[frame])
    #         im3.set_array(np.abs(u_ref[frame] - u_pred[frame]))
    #         ax[0].set_title(f"Exact (t={t_star[frame]:.2f})")
    #         ax[1].set_title(f"Predicted (t={t_star[frame]:.2f})")
    #         ax[2].set_title(f"Absolute Error (t={t_star[frame]:.2f})")
    #         return im1, im2, im3

    #     ani = animation.FuncAnimation(fig, update, frames=len(t_star), interval=50)

    #     # Save animation
    #     writer = animation.FFMpegWriter(fps=20, bitrate=1800)
    #     ani.save(video_path, writer=writer)

    #     print(f"Video saved at: {video_path}")

    # # Call the function after computing `u_pred`
    # save_2d_video(u_ref, u_pred, t_star, x_star, workdir, config)
