from data.load_data import get_segmented_audio
from data.data_utils import get_loaders

from train.model_cv import model_cv_lambdas

from typing import Union
import os
import torch
import numpy as np
from tqdm import tqdm


def train_model(
    audio_dirs: list[str],
    seg_dirs: list[str],
    model_dir: str,
    max_vocs: int = 5000,
    context_len=0.25,
    seed: Union[None, int] = 1234,
    shuffle_order=False,
    max_jobs: int = 4,
    batch_size: int = 32,
    n_epochs: int = 100,
    save_freq: int = 5,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    function for training a model. takes audio from
    audio_dirs, onsets and offsets from segmentation files
    in seg_dirs. cross-validates over the one hyperparameter of this model

    inputs
    --------
            audio_dirs: list of folders with audio
            seg_dirs: list of folders with segmentation decisions
            model_dir: location to save model checkpoints and plots
            max_vocs: max number of vocal chunks to train on
            context_len: context window to train model on
            seed: random seed for reproducibility
            shuffle_order: whether to shuffle audio files for gathering train data
            max_jobs: max number of jobs for dataloader
            batch_size: batch size during training
            n_epochs: max number of passes through the data during training
            save_freq: how often (in epochs) we want to checkpoint model
            device: device to train on. falls back to cpu if cuda is unavailable
    returns
    --------
            best model after hyperparameter cross-validation
    """

    n_cpu = os.cpu_count()

    n_jobs = min(max_jobs, n_cpu if n_cpu is not None else 0)

    assert len(audio_dirs) == len(seg_dirs), print(
        "Need the same number of audio dirs as segment dirs!"
    )
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    chunks_per_dir = max_vocs // len(audio_dirs)
    chunks = []

    for audio_dir, seg_dir in tqdm(
        zip(audio_dirs, seg_dirs), desc="Gathering training data", total=len(audio_dirs)
    ):
        audio, sr = get_segmented_audio(
            audio_dir,
            seg_dir,
            max_vocs=chunks_per_dir,
            context_len=context_len,
            seed=seed,
            training=True,
            extend=True,
            shuffle_order=shuffle_order,
        )
        chunks += audio

    print(f"Gathered {len(chunks)}/{max_vocs} allowed vocalizations")
    dt = 1 / sr
    dataloaders = get_loaders(
        np.stack(chunks, axis=0),
        num_workers=n_jobs,
        batch_size=batch_size,
        train_size=0.6,
        cv=True,
        seed=seed,
        dt=dt,
    )

    best_model = model_cv_lambdas(
        dls=dataloaders,
        dt=dt,
        n_epochs=n_epochs,
        lr=1e-3,
        n_kernels=15,
        expand_factor=10,
        n_layers=3,
        d_state=1,
        d_conv=4,
        tau=dt,
        model_path=model_dir,
        save_freq=save_freq,
        device=device,
    )

    return best_model


if __name__ == "__main__":
    pass
