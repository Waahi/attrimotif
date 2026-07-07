# Release checklist for attrimotif (SoftwareX submission, steps 3-5)

Steps 3 and 4 are outward actions on the author's own accounts (GitHub, PyPI,
Zenodo). This file makes them turnkey. Step 5 fills the real DOI/URLs back into
the manuscript.

Before starting, bump the version from `0.9.0` to `1.0.0` in all five places:
`pyproject.toml`, `src/attrimotif/__init__.py`, `CITATION.cff`, `codemeta.json`,
and `.zenodo.json`. Add your ORCID to `.zenodo.json` and `CITATION.cff`.

## Step 3 — GitHub push + CI

```bash
git init
git add -A
git commit -m "attrimotif v1.0.0"
git tag v1.0.0
git remote add origin https://github.com/Waahi/attrimotif.git
git push -u origin main --tags
```

Watch the Actions tab. The matrix (Python 3.9-3.12 x Linux/Windows/macOS) must be
green, including the netrd Portrait-Divergence parity test (netrd is installed
in CI). If that parity test fails beyond tolerance, no manuscript change is
required (the wording already hedges), but consider making `backend="netrd"`
the default in `compare.portrait_divergence`.

## Step 4 — PyPI + Zenodo + docs

```bash
# 4a. Build and publish to PyPI (requires a PyPI account + API token)
python -m build
python -m twine check dist/*
python -m twine upload dist/*

# 4b. Zenodo: turn on the GitHub-Zenodo integration for the repo, then publish
#     the v1.0.0 GitHub release. Zenodo mints a version DOI from .zenodo.json.
#     (Alternatively upload dist/ manually and paste the .zenodo.json metadata.)

# 4c. Docs to GitHub Pages
python -m pip install mkdocs
mkdocs gh-deploy
```

## Step 5 — final metadata pass (in manuscript.md)

Fill the Code metadata table with the real values:

- C1 -> `v1.0.0`
- C2 -> `https://github.com/Waahi/attrimotif`
- C3 -> the Zenodo version DOI, e.g. `10.5281/zenodo.XXXXXXX`
- C8 -> the published docs URL, e.g. `https://Waahi.github.io/attrimotif`
- C9 -> support email (already set)

Also sync the package-facing text to the released state:

- `README.md`: change `pip install attrimotif  # from PyPI (planned)` to the plain
  released install line.
- Add the coverage badge (from the CI `pytest-cov` run) and the CI status badge to
  `README.md`; optionally state the coverage percentage in the manuscript.

Then clear every placeholder and confirm integrity:

```bash
grep -rnE "planned|TBD|at release|minted at release|REF-|TODO|0\.9\.0" manuscript.md README.md   # expect no hits
```

Confirm all references resolve, remove any remaining working comments, and the
submission package is final.
