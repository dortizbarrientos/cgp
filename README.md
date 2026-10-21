# Gamete persistence and conspecific gametic precedence

Computational work for **Egg availability and the evolution of reproductive
discrimination**, by Daniel Ortiz-Barrientos, Maria R. Servedio and
Loren H. Rieseberg. Daniel is the corresponding author.

This repository contains the current finite-egg model, numerical checks,
all seven manuscript figures and their underlying data, and two earlier model
versions. Numerical values are illustrative model outputs, not fitted data.
The proposed extension to secondary contact and reinforcement **has not yet
been implemented or simulated**.

## Reproduce the current results

The release was tested with Python 3.12.14. Use a Python 3.12 environment:

```bash
git clone https://github.com/dortizbarrientos/cgp.git
cd cgp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/validate.py
python src/analyse.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.
The pinned dependencies are NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8.
With GNU Make installed, `make reproduce` runs validation and analysis.
`bash reproduce.sh` provides the same two steps without Make.

The scripts overwrite generated files in `results/` and `figures/`. The
analysis saves every figure as PDF, SVG and PNG. PDF and SVG are scalable;
CSV files contain the values plotted. Runtime and figure metadata can differ
between reruns. The release verification report compares the scientific data.

## Files

| Path | Purpose |
| --- | --- |
| `src/model.py` | Finite eggs and sperm, functional loss, fertilisation, later receipt, fitness and quantitative-genetic response |
| `src/validate.py` | Independent numerical and analytical checks |
| `src/analyse.py` | All seven current figures, figure CSVs, trajectories, equilibria and sensitivity comparisons |
| `results/parameters.json` | Parameters recorded for the supplied analysis |
| `results/summary.json` | Main numerical results and software versions |
| `results/validation.json` | Validation errors, stochastic seeds and check status |
| `figures/` | Four main and three supplementary figures |
| `legacy/v1/` | Original scaled-availability model, code, outputs and guide |
| `legacy/v2/` | Earlier delayed-fertilisation model, code, outputs and theory note |
| `docs/` | Model history, research scope and release verification |

Change biological defaults in the `Ecology` dataclass in `src/model.py`, or
use `dataclasses.replace` in a separate analysis. The parameters JSON records
what was run; it is not an input configuration file.

## Current model and checks

The physiological model tracks remaining eggs, conspecific and heterospecific
functional sperm, successful fertilisations, and loss of sperm function. A
successful fertilisation consumes one egg and one functional sperm. Possible
later conspecific receipt is evaluated in two branches, then averaged over
its probability. Offspring value enters the reproductive payoff separately
from sperm function and fertilisation success.

Validation covers conservation, nonnegative states, an independently written
adaptive solver, exact limits, time-step sensitivity, an adjoint calculation,
Gaussian-selection calculations and exact stochastic event simulations.
The default stochastic checks use 5,000 replicates in each of four cases.
These are implementation checks and model comparisons, not empirical tests.

The quantitative-genetic recursion holds genetic and phenotypic covariance
fixed. It follows recipient-trait responses but does not track ancestry,
introgression or stable species boundaries. See `docs/research-roadmap.md` for
the proposed next model and the distinction between current results and
reinforcement hypotheses.

## Earlier versions

The two historical versions implement different assumptions. They do not
generate the current manuscript figures. Consult their original READMEs
before rerunning them; version 1 additionally needs ReportLab and pypdf.
The historical documentation and author lists are retained as research
history. Current authorship is listed above and in `CITATION.cff`.

The complete current manuscript, supplement, TeX sources, references and
annotated discussion documents are distributed in a separate Overleaf ZIP.
Unsent correspondence and internal discussion copies are not part of this
public computational repository.

## Research status

This is working research software. Independent scientific review and
journal preparation remain pending. ChatGPT/Codex assisted with formulation,
derivation, drafting, code and numerical figures. No software licence or
public archival DOI has been assigned.
