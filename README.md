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
