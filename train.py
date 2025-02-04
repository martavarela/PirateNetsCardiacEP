import os
import time

from functools import partial
import jax
import jax.numpy as jnp
from jax import random, pmap
from jax.tree_util import tree_map
from sklearn.model_selection import train_test_split

import ml_collections
from absl import logging
import wandb

from jaxpi.samplers import SpaceSampler, BaseSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset

class DataSampler(BaseSampler):
    def __init__(self, coords, V, test_size, batch_size, train=True, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.V = V
        self.coords = coords

        # Perform train-test split
        coords_train, coords_test, V_train, _ = train_test_split(
            self.coords, V, test_size=test_size, random_state=42)
        self.V_train = jnp.array(V_train)  # conver to jnp array so that it can be indexed
        # Use either training or testing data
        self.selected_coords = coords_train if train else coords_test

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        idx = random.choice(
            key, self.selected_coords.shape[0], shape=(self.batch_size,))
        coords_batch = self.selected_coords[idx, :]
        V_batch = self.V_train[idx, :]
        batch = (coords_batch, V_batch)
        return batch

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()

    # Get dataset
    coords, t_star, x_star, y_star, z_star, V, W  = get_dataset("AP_sphere_planar_data.mat")
    V0 = V[0, :]
    
    # Define samplers: data and residual
    test_size = 0.2
    data_sampler = iter(DataSampler(coords, V, test_size, config.training.batch_size_per_device))
    res_sampler = iter(SpaceSampler(coords, config.training.batch_size_per_device))

    samplers = {
            "res": res_sampler,
            "data": data_sampler
        }
    # Initialize model
    model = models.AlievPanfilov3D(config, V0, V, t_star, x_star, y_star, z_star, coords)

    # Initialize evaluator
    evaluator = models.Evaluator(config, model)

    print("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        # new, included both data and res batches
        batch = {}
        for key, sampler in samplers.items():
            batch[key] = next(sampler)
        model.state = model.step(model.state, batch)
        # old, only res batches
        # batch = next(res_sampler)
        # model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, V)
                wandb.log(log_dict, step)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model
