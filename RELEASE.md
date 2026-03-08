# Release: nwf-vision

## Publish to PyPI

```bash
cd c:\nwf\libraries\nwf-vision
pip install build twine
python -m build
twine upload dist/*
```

## Git

```bash
git add examples/ notebooks/ README.md pyproject.toml .gitignore
git status
git commit -m "Add examples: split_cifar, ood_cifar_svhn, active_learning; notebooks; Colab badge; README with application areas"
git push origin main
```

## Colab

After push, the badge in README opens:
https://colab.research.google.com/github/romero19912017-ui/nwf-vision/blob/main/notebooks/split_cifar.ipynb
