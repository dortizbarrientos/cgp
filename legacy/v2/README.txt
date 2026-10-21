CGP MODEL VERSION 2
Prepared for Daniel Ortiz-Barrientos, 20 October 2026.

OVERLEAF
Upload CGP_Model_V2_Overleaf.zip as a new project. Set CGP_Model_V2.tex as
the main document and pdfLaTeX as compiler. Compile twice. References are
embedded; no BibTeX, shell escape, network access or Python is needed.

WHAT IS NEW
The existing one-bout payoff is extended with a delay to fertilisation, a
possible later compatible contribution, recipient traits for acceptance,
postmating efficacy and conspecific persistence, and a quantitative-genetic
mean recursion. Selection through mothers is separated from incoming paternal
breeding values. Three illustrative figures are included.

IMPORTANT SCOPE
This is a working theory note, not a complete speciation model or an empirical
prediction. Fixed P and G and a restored Gaussian distribution are approximation
assumptions. Hybrid ancestry, linkage disequilibrium, male trait evolution,
reciprocal population evolution, repeated choices and finite ovule depletion
are not simulated. The document explains their consequences.

REPRODUCE CALCULATIONS
Install dependencies from requirements.txt, then run:
    python3 src/model_v2.py
The script regenerates the figures, CSV trajectories, parameters.json,
results.json and checks.json in this directory. It uses deterministic numerical
quadrature rather than stochastic evolutionary simulations.

For a different trajectory length:
    python3 src/model_v2.py --generations 600
For an isolated output directory:
    python3 src/model_v2.py --output /absolute/path/to/new-output
Edit the Parameters dataclass to explore biological assumptions. P and G are
specified in main(). P-G must remain positive semidefinite. Parameter changes
require updating the numerical table and captions in the TeX document.

CONTROLS
data/future_C_0p1.csv and data/future_C_0p8.csv: primary examples.
data/negative_genetic_correlation.csv: r-b genetic correlation of -0.8.
data/selection_only.csv: incoming-mean term removed, not hybrids removed.
In this last counterfactual, heterospecific fathers effectively match the
current resident mean for the recipient traits in the mean-inheritance step.

Figures are provided in vector PDF/SVG and raster PNG formats. All values are
illustrative. Figure 3 pools fertilisation counts in the standardised assay;
its zero-mean, variable-b starting population is not a homogeneous b=0 null.

VERSIONS
Version 1 files are retained separately and unchanged. This package contains
only version 2 source, figures, numerical outputs and annotated Python code.
The compiled model PDF is supplied separately from the source ZIP.
