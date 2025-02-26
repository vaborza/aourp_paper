### By Victor A. Borza
# Instructions for Use
## How to access datasets
To reproduce analyses with data, scripts need to be uploaded into the _All of Us_ Research Program's Researcher Workbench.
Instructions for obtaining an account are available on [their website](https://workbench.researchallofus.org/).
Note that all analyses using geographic data (i.e., 3-digit ZIP codes) require access to the Controlled Tier.

## Creating an analysis environment
Analyses can be run in the Jupyter environment available in the Researcher Workbench.
Before running code, create directory `data/` and subdirectory `data/ehr_sites/` in the `home/jupyter/` directory.
Ensure that the following Python packages are installed in the environment: `pandas`, `numpy`, `scipy`, and `pathos` (if multiprocessing).

## Replicating Analysis
1. If analysis is performed in the Registered Tier, please set the variable `registered_tier` to `True` in script `scripts/process_data.py`.
2. Run `scripts/process_data.py`, which may also be copied into Jupyter code blocks.
3. To replicate the main results of simulations, run `scripts/simulate_recruitment_ehr_sites.py`. 
To reliably run code in a Jupyter code block (though greater time may be needed), set variable `multiprocessing` to `False`
4. To replicate the Pareto frontier results (Figure 5), run `scripts/simulate_recruitment_ehr_sites_pareto.py`. 
To reliably run code in a Jupyter code block (though greater time may be needed), set variable `multiprocessing` to `False`
5. Appendix results may be replicated via the other scripts available in `scripts/`
6. To plot the main results of the paper (Figures 1-8), run `scripts/plot_results.py`.

Please note that the experimental analyses included in this paper are stochastic and thus may differ from the exact results reported in the paper.
The imputation of race values for individuals who identify as Hispanic/Latino and a non-defined race value is a randomized process mirroring U.S. Census procedures.
While the total participant counts remain the same, the distribution of race values within this subgroup may differ (though it will converge in expectation).
Secondly, simulated recruitment is a stochastic process, necessitating the 40 experimental replicates to determine simulation efficacy.