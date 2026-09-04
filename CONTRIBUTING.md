# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Do not commit observational catalogs, MCMC output, downloaded light curves,
MIST grids, or other large/generated files. Tests added to the default suite
must run without private science data.

For model changes, include a focused test and report NumPyro diagnostics
(divergences, R-hat, and effective sample size) when the change can affect the
posterior. Keep unrelated formatting or notebook-output changes out of the
same commit.
