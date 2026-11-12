# ouroboros

this branch is for cleaning up the code to make it more usable and presentable for other vocalization-focused individuals.

## to-do:

- [x] include third party files, to minimize external requirements
- [ ] remove all unnecessary files
- [x] add type specifications and function, class docstrings. follow format from mamba.py
- [ ] test performance to ensure (approximate) equivalence between these versions
- [x] ensure preprocessing and data loading is still easy

## devices

models, training and evaluation take a `device` argument (default `"cuda"`). If cuda is requested but
`torch.cuda.is_available()` is False, a `RuntimeWarning` is raised and the cpu is used instead; pass
`device="cpu"` to do this deliberately.

## tests

```
uv sync --group dev
uv run pytest
```

the suite runs on the cpu with tiny models (about 10 s); the gpu test is skipped when no gpu is present.

## comparisons

`comparisons/` has baselines with the same calling conventions as the main code, for comparing against Ouroboros on identical data:

- `comparisons/vae.py`: a convolutional VAE with the architecture and objective of the VAE from
  [Autoencoded Vocal Analysis](https://github.com/pearsonlab/autoencoded-vocal-analysis), trained on 128 x 128 log-mel spectrograms.
  `train_model(audio_dirs, seg_dirs, model_dir, ...)` mirrors `train.train_model.train_model`; the pieces
  (`VAE`, `get_spectrogram_loaders`, `train`, `save_model`/`load_model`, `eval_model_error`,
  `model_cv_precisions`) mirror their counterparts. `get_latents_from_audio` gives latent features for new vocalizations.
- `comparisons/acoustic_features.py`: acoustic features from [VocalPy](https://vocalpy.readthedocs.io) (SAT
  features summarized over time, or BioSound). `get_acoustic_features(audio_dirs, seg_dirs, ...)` takes the same data arguments,
  and `make_feature_pipeline` imputes, standardizes and optionally reduces the dimensionality of the features.

Tests that are slow (BioSound, parallel jobs) are skipped by default: `uv run pytest -m slow`.

### long recordings

Recordings are memory-mapped and only the segments that are used are read (`data/load_data.py`), for every model. The
acoustic features are also computed one audio file at a time, so memory does not grow with the number or length of
the recordings: `iter_acoustic_features` yields features per file, and `get_acoustic_features(..., cache_dir=...)`
saves each finished file's features so an interrupted run can be resumed. `iter_segmented_audio` streams the segmented audio
itself, one file at a time.
