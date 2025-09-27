import os
import re
import shutil
import time
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import fully_shard

from resiliency.goodput_measure import logging as goodput_logging
from resiliency.goodput_measure import constant as goodput_event

# --- Configuration ---
CHECKPOINT_DIR = "my_fsdp_checkpoint"
MAX_STEPS = 10000
SAVE_INTERVAL = 500
PRINT_INTERVAL = SAVE_INTERVAL // 5

class AppState(Stateful):
    """A stateful object that wraps the model, optimizer, and current training step."""
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.step = 0

    def state_dict(self):
        """Called by dcp.save() to get the state of the application."""
        model_state_dict, optim_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optim_state_dict,
            "step": self.step,
        }

    def load_state_dict(self, state_dict):
        """Called by dcp.load() to set the state of the application."""
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )
        self.step = state_dict["step"]

class ToyModel(nn.Module):
    """A simple model for demonstration purposes."""
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def find_latest_checkpoint_path(checkpoint_dir: str):
    """
    Finds the path to the latest checkpoint subfolder (e.g., 'step=N').

    Returns:
        The full path to the latest checkpoint subfolder, or None if not found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    latest_step = -1
    latest_path = None

    # Regex to match subfolders named 'step=NUMBER'
    pattern = re.compile(r"^step=(\d+)$")

    for subdir in os.listdir(checkpoint_dir):
        full_path = os.path.join(checkpoint_dir, subdir)
        if os.path.isdir(full_path):
            match = pattern.match(subdir)
            if match:
                step = int(match.group(1))
                if step > latest_step:
                    latest_step = step
                    latest_path = full_path

    return latest_path

def run_training():
    """The main worker function for training."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank, world_size)

    # 1. Initialize Model and Optimizer
    model = ToyModel().to(rank)
    model = fully_shard(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # 2. Create the Stateful object
    app_state = AppState(model, optimizer)

    # 3. Find and load the latest checkpoint if it exists
    start_step = 0
    latest_checkpoint_path = find_latest_checkpoint_path(CHECKPOINT_DIR)

    if latest_checkpoint_path:
        if rank == 0:
            print(f"Found latest checkpoint at '{latest_checkpoint_path}'. Resuming training.")

        dcp.load(
            state_dict={"app": app_state},
            checkpoint_id=latest_checkpoint_path,
        )
        start_step = app_state.step
        if rank == 0:
            print(f"Successfully loaded. Resuming from step {start_step}.")
    else:
        if rank == 0:
            print("No valid checkpoint subfolders found. Starting training from scratch.")

    if rank == 0:
        goodput_logging.log_event(goodput_event.CHECKPOINT_LOADED, step=start_step)

    # 4. Training Loop
    last_time, last_step = time.time(), start_step
    for step in range(start_step, MAX_STEPS):
        app_state.step = step + 1 # Update step before potential save

        # Dummy training step
        optimizer.zero_grad()
        input_data = torch.rand(8, 16, device="cuda") + rank
        loss = model(input_data).sum()
        loss.backward()
        optimizer.step()

        if rank == 0 and app_state.step % PRINT_INTERVAL == 0:
            curr_time, curr_step = time.time(), app_state.step
            print(f"Step {app_state.step}/{MAX_STEPS} | Loss: {loss.item():.4f} | Avg Step Time (Last {curr_step - last_step} Steps): {(curr_time - last_time)/(curr_step - last_step):.6f}s")
            last_time, last_step = curr_time, curr_step

        # 5. Periodically save a checkpoint in a versioned subfolder
        if (app_state.step % SAVE_INTERVAL == 0) or (app_state.step == MAX_STEPS):
            dist.barrier() # Ensure all ranks are synchronized before saving

            # Define the specific subfolder for this checkpoint
            save_dir = os.path.join(CHECKPOINT_DIR, f"step={app_state.step}")

            if rank == 0:
                print(f"\nSaving checkpoint to '{save_dir}'...")

            dcp.save(
                state_dict={"app": app_state},
                checkpoint_id=save_dir,
            )

            if rank == 0:
                goodput_logging.log_event(goodput_event.CHECKPOINT_SAVED, step=app_state.step)
                print("Checkpoint saved successfully.\n")

    cleanup()

if __name__ == "__main__":
    run_training()
