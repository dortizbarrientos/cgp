CGP MODEL FIGURES AND MATHEMATICAL ARGUMENTS
Prepared for Daniel Ortiz-Barrientos, 20 October 2026.

Purpose
-------
Four figures explain the structure of a deliberately simple model. All
numerical choices are illustrative. No values were estimated from data.

Files
-----
CGP_Model_Figures_and_Arguments.pdf: four-page guide with vector artwork,
    captions, mathematical interpretations and limits.
figures/: each figure as 300-dpi PNG and vector PDF/SVG. SVG retains editable
    text. A vector editor may require DejaVu Sans for exact font matching.
data/: numerical values used in each panel, as CSV files.
src/make_figures.py: annotated Python source that regenerates all outputs.
model_checks.json: numerical consistency checks.

Reproduce
---------
Install numpy, scipy, matplotlib, reportlab and pypdf, then run:
    python3 src/make_figures.py
Run from any working directory. Paths are resolved relative to the script.

Variables and units
-------------------
x0: initial functional conspecific store, in scaled availability units.
x(t): functional conspecific availability at storage age t.
y: functional heterospecific availability delivered at the second mating.
h_C, h_H: half-lives of functional availability, in days in these examples.
w: expected fitness contribution per hybrid offspring relative to one
   conspecific offspring; restricted here to [0,1]. It can incorporate
   survival and subsequent reproduction, but must use the same definition
   in both offspring classes.
c: extra mating cost, expressed as an absolute fitness deduction per
   potential egg over the evaluated bout. It is not a survival percentage.

Model and derivation
--------------------
Functional availability: x(t) = x0 * 2^(-t/h_C).
Assumed probability of fertilisation with availability z: F(z)=z/(1+z).
Thus z=1 is the availability giving 50% fertilisation, not one sperm.
Conditional on fertilisation, conspecific share=x/(x+y), hybrid share=y/(x+y).
The resulting unconditional probabilities are x/(1+x+y), y/(1+x+y),
and 1/(1+x+y) for conspecific, hybrid and unfertilised outcomes.

W_reject = x/(1+x).
W_accept = (x+w*y)/(1+x+y) - c.

Subtract the two expressions using the common denominator
(1+x+y)*(1+x). The numerator is:
(x+w*y)*(1+x) - x*(1+x+y) = y*[w*(1+x)-x].
Therefore:
Delta W = y/(1+x+y) * [w-x/(1+x)] - c.

For y>0, acceptance has the higher payoff exactly when:
w > w_crit(t) = x(t)/(1+x(t)) + c*(1+x(t)+y)/y.
This identity, rather than a simulation of behaviour, defines Figure 3A.

When c=0 and 0<w<1, the inequality is equivalent to x < w/(1-w).
Substitution of x=x0*2^(-t/h_C) and taking base-2 logarithms gives:
t > h_C * log2[x0*(1-w)/w].
The earliest nonnegative threshold is the maximum of this expression and
zero. At equality the two actions have equal payoff. At w=0 and c>=0,
acceptance cannot improve the payoff; at w=1 and c=0, any y>0 improves
fertilisation whenever the original store is finite. With positive cost,
acceptance never becomes strictly beneficial if c >= w*y/(1+y).

For Figure 4, time u begins when both ejaculates are present. Starting
availability is X(0)=Y(0)=1, h_C=2, and h_H is 1, 2 or 4 days.
X(u)=2^(-u/h_C), Y(u)=2^(-u/h_H).
P_C(u)=X/(X+Y), while total fertilisation is F(u)=(X+Y)/(1+X+Y).
The odds P_C/(1-P_C)=X/Y are proportional to
2^[u*(1/h_H-1/h_C)]. Consequently conspecific share increases when h_H<h_C.
The total store decreases in every scenario, so total fertilisation falls.
These are snapshots at different ages, not accumulated egg production.

Assumptions and limits
----------------------
1. One recipient background; a fixed set of potential eggs in a short bout.
2. Old sperm remain available after a new ejaculate arrives. No complete
   replacement is imposed and there is no explicit storage capacity limit.
3. Constant proportional loss of functional availability; no explicit sperm
   use, ejection, replenishment or changes induced by a competitor.
4. The saturating fertility curve and proportional sperm use are assumptions.
   The exact numerical thresholds depend on these functions.
5. Offspring fitness affects the recipient payoff, not paternity directly.
6. The payoff comparison does not establish that recipients perceive their
   state or implement the favoured action. There is no future mating option
   within the evaluated bout and no dynamic optimisation of waiting.
7. There are no specified genetic effects or costs of production, longevity
   or discrimination. These figures identify conditional fitness advantages;
   they do not establish evolutionary optima or changes in allele frequency.
8. Divergence matters here only through a separately justified effect on w
   or another parameter. No monotonic divergence relationship is assumed.
9. Differential persistence can generate conspecific bias without an induced
   competitive interaction. Single-donor controls are needed to distinguish
   baseline incompatibility from an effect of another ejaculate.
10. A pollen extension needs explicit deposition, germination and ovule
    availability. These figures depict the sperm-storage model.

Relation to established work
---------------------------
Alonzo SH and Pizzari T (2013). Selection on female remating interval is
influenced by male sperm competition strategies and ejaculate characteristics.
Philosophical Transactions of the Royal Society B 368:20120044.
https://doi.org/10.1098/rstb.2012.0044
This provides theoretical context for fertility, depletion, costs and remating.

Matute DR (2010). Reinforcement of Gametic Isolation in Drosophila.
PLoS Biology 8:e1000341. https://doi.org/10.1371/journal.pbio.1000341
This provides empirical context linking changes in the use or persistence
of heterospecific sperm with remating benefits; it did not isolate mortality
from depletion or reduced retention.

The equations and visualisations here implement the toy model developed in
this conversation. They are not figures or fitted results from those studies.

Figure captions
---------------

Figure_1_Persistence_and_Initial_Amount
Argument: measure lifespan and initial amount separately.
Panel A holds initial availability constant and varies the functional half-
life. The open circles mark the age at which half of each initial cohort
remains functional and available. Panel B holds the half-life at two days and
varies the initial store. Increasing the starting amount raises the entire
availability curve without changing proportional decay. In this model,
producing more functional sperm and preserving existing sperm are distinct
ways to maintain availability. A larger store is a possible consequence of
greater production, not a direct measurement of production.

Measurement: S(t) is the surviving functional fraction of the original store,
not the fraction alive among sperm retained at a later sample. Initial
transfer, entry into storage, death, loss of function, use and ejection
require separate measurements. These trajectories do not identify the
biological cause of loss.

Figure_2_Remating_Tradeoff
Argument: fertilisation assurance and offspring fitness can favour different actions.
Panel A partitions potential eggs into conspecific fertilisations, hybrid
fertilisations and fertilisation failures. At either age, accepting the new
ejaculate reduces fertilisation failure but also reduces the probability of
conspecific fertilisation. Panel B weights conspecific offspring by 1 and
hybrid offspring by 0.5, then subtracts a mating cost of 0.05 from acceptance.
Rejection gives the higher payoff early; acceptance gives the higher payoff
after 4.84 days. The original conspecific sperm remain available after
remating. The change in the preferred action is caused by declining
conspecific availability, not a change in hybrid fitness.

Scope: payoffs describe the next short reproductive bout, with a fixed pool of
potential eggs and no further mating opportunity. Cost is an absolute payoff
deduction, not a 5% survival reduction. The comparison assumes access to the
relevant state information; it does not demonstrate that females assess sperm
condition.

Figure_3_Acceptance_Thresholds
Argument: lower hybrid fitness and longer sperm persistence delay beneficial heterospecific mating.
Panel A rearranges the payoff inequality into a critical hybrid fitness. The
solid boundary includes mating cost 0.05; the dashed boundary sets the cost to
zero. Shaded regions refer to the solid boundary. At a fixed age, acceptance
has the higher payoff above that boundary. Lower hybrid fitness requires a
more depleted conspecific store before acceptance becomes beneficial. Panel B
isolates the effect of conspecific half-life when mating is cost-free. Open
circles show thresholds at hybrid fitness 0.5. The formula applies for 0 < w <
1 and a positive new ejaculate; negative calculated times are set to zero.

Inference: divergence predicts a later acceptance threshold only if it lowers
hybrid fitness, holding the other model components fixed. The horizontal
variable in B is hybrid fitness, not divergence time. Initial store and
longevity have no specified evolutionary costs here, so this figure does not
establish their evolutionary optima.

Figure_4_CGP_and_Total_Fertilisation
Argument: a rising conspecific fertilisation share does not imply increasing reproductive output.
Panel A shows conspecific share conditional on an egg being fertilised. The
share increases when heterospecific sperm lose function faster, remains
constant when half-lives are equal, and decreases when heterospecific sperm
persist longer. Panel B shows the probability that a potential egg is
fertilised at all. That probability declines in every scenario because both
functional stores decline. For the one-day heterospecific half-life,
conspecific share reaches about 94% at day eight, while total fertilisation
falls to about 6%. Both quantities are required to interpret the biological
consequence of a paternity bias.

Scope: equal initial stores isolate persistence and are a separate example
from Figures 2 and 3. Half-lives are fixed; there is no new input, explicit
sperm use or induced response to a competitor. These are snapshots across
storage ages, not cumulative egg production. Differential persistence alone
can generate this bias; it does not establish a competition-specific
mechanism.
