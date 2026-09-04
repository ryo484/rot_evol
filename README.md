# rot-evol

`rot-evol` is a research codebase for hierarchical inference of stellar
rotation evolution from stellar ages, projected rotation velocities
(`v sin i`), spectroscopy, Gaia photometry, and parallax.

The package contains power-law, broken-power-law, Gaussian-process, and
optional JAXSpin rotation relations. It supports both fits based on existing
isochrone posterior samples and a joint MIST isochrone + rotation fit.

> **Status:** research software. Check convergence diagnostics and validate
> the scientific assumptions before using a result in a publication.

## Installation

Python 3.10 or newer is required. A CPU-only development installation is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

JAX accelerator installation is platform-specific. Install the appropriate
JAX build before installing this project when using a GPU.

The joint isochrone model additionally requires the external `jaxstar`
package and its MIST grids. The `rotation_law="jaxspin"` option additionally
requires `jaxspin`; other rotation laws do not import it.

## Data configuration

Observational data and posterior products are deliberately not stored in Git.
Configure their locations with environment variables:

```bash
export ROT_EVOL_ANALYSIS_ROOT=/path/to/analysis_results
export ROT_EVOL_RESULT_ROOT=result
export ROT_EVOL_JAXSPIN_DATA_DIR=/path/to/jaxspin/data  # JAXSpin only
```

An analysis tree is expected to contain one directory per frame, with files
such as:

```text
analysis_results/
└── .../<frame-id>/
    ├── orders03-08/mcmc.nc
    └── mistfit_g_st38/
        ├── mcmc.nc
        └── job_metadata.json
```

The root-level `observation_log.csv` must contain at least `Frame`, `Object`,
`Tags`, and `Count (e-)`. Joint fits read extinction-corrected Gaia G
magnitudes and parallax measurements from `job_metadata.json`.

See [.env.example](.env.example) for the available path settings.

## Running the analyses

The existing-isochrone-posterior analysis is configured near the top of
`run_rot_evol.py`:

```bash
python run_rot_evol.py
```

The joint MIST isochrone + power-law rotation analysis is run with:

```bash
python run_isochrone_fit.py
```

It currently evaluates four mass/metallicity selections. Run a single group
with `--selection-index`:

| Index | Mass range | Metallicity range |
|---:|---:|---:|
| 1 | 0.9–1.1 M☉ | [Fe/H] > −0.1 |
| 2 | 0.9–1.1 M☉ | [Fe/H] < −0.1 |
| 3 | 1.1–1.3 M☉ | [Fe/H] > −0.1 |
| 4 | 1.1–1.3 M☉ | [Fe/H] < −0.1 |

```bash
python run_isochrone_fit.py --selection-index 1 --overwrite
```

Four NumPyro chains run in parallel on four host CPU devices. Independent
selection indices can also be launched as separate processes when sufficient
CPU and memory are available. Do not run two writers for the same selection.

Regenerate figures without rerunning MCMC:

```bash
python run_isochrone_fit.py --plot-only
```

## Outputs

Each fit is written below `ROT_EVOL_RESULT_ROOT/<selection-label>/`. Outputs
include:

- `mcmc.nc`: ArviZ `InferenceData` with run metadata;
- `corner.png` and `trace.png`: hyperparameter diagnostics;
- `rotation.png` and `rotation_linear.png`: age–rotation relation;
- `age_post_vs_v_post.png`: paired stellar age/velocity posteriors with the
  inferred rotation relation;
- `logage_post_hist.png`: per-star posterior-median log ages with the inferred
  population distribution;
- observed-versus-posterior diagnostic plots.

Generated results, caches, light curves, FITS files, and local CSV catalogs are
ignored by Git. Keep irreplaceable data in separately backed-up storage.

## Repository layout

```text
rot_evol/                 Python package
  data.py                 data containers and posterior loaders
  models.py               NumPyro models and rotation laws
  plotting.py             serialization and diagnostic figures
run_rot_evol.py           posterior-based inference runner
run_isochrone_fit.py      joint isochrone + rotation runner
simulation.py             simulation experiment driver
sim_stars_given_age.py    synthetic-star generation helpers
lightcurves/              TESS analysis utilities and notebook
tests/                    unit tests without external science data
```

## Development

Run the data-independent test suite with:

```bash
python -m pytest
```

GitHub Actions runs the same tests for pushes and pull requests. External
science-data integration tests are intentionally separate because their input
archives and MIST/JAXSpin grids are not distributable with this repository.

## Citation and license

No open-source license has been selected yet. Until a license is added, the
repository is source-available but does not grant reuse rights. Add a license
and citation metadata before a public release.
