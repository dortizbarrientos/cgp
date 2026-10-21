# Model history

The repository preserves three stages of the theory. The current manuscript
uses `src/`; the historical versions are not equivalent implementations.

| Version | Location | Main assumptions |
| --- | --- | --- |
| Original model | `legacy/v1/` | Scaled gamete availability, exponential loss, saturating fertilisation and a one-bout acceptance payoff. No finite egg depletion. |
| Earlier extension | `legacy/v2/` | Delayed fertilisation, a possible later compatible contribution and a quantitative-genetic mean recursion with an incoming paternal mean term. No finite egg depletion or ancestry dynamics. |
| Current manuscript | `src/` | Finite eggs and sperm, functional loss, consumption during fertilisation, later receipt, acceptance and access-restraint traits, and a recipient-limited Gaussian selection approximation. |
| Proposed follow-up | `docs/research-roadmap.md` | Secondary contact, explicit inheritance, migration, ecological selection and ancestry. Not implemented. |

The original model was distributed both as a figure package and in an
Overleaf package. Its duplicate computational files are included once here.
The current paper and its later biological-examples revision have identical
computational source, figures and numerical results; one copy is retained.

Historical documentation preserves its original author lists and terminology.
A historical figure named “isolation” does not imply that hybrid ancestry or
species boundaries were simulated. Read the scope statements for each version.
