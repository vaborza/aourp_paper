# Simulates recruitment using EHR sites and adds back the individuals with no EHR site to the cohort

import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.special import softmax
from scipy.optimize import minimize
from itertools import product
from pathos.pools import ProcessPool
import pickle

## Load in datasets and related functions

# Census data by generalized (i.e., modified) ZIP3 values
aou_census_imputed = pd.read_pickle('~/data/census_generalized.pkl')
census_tot = aou_census_imputed.sum()

# Superellipse boundaries, the KLD from Census to Uniform and vice versa
KLD_CU = entropy(census_tot, np.ones_like(census_tot))
KLD_UC = entropy(np.ones_like(census_tot), census_tot)

# Cohorts filtered by when participants joined the program
quarterly_cumu_cohorts = []
quarterly_indiv_cohorts = []
quarterly_indiv_nosite_cohorts = []
for i in range(21):
    quarterly_cumu_cohorts.append(pd.read_pickle(f'~/data/ehr_sites/aou_quarterly_cumu_cohorts_{i}.pkl'))
    quarterly_indiv_cohorts.append(pd.read_pickle(f'~/data/ehr_sites/aou_quarterly_indiv_cohorts_{i}.pkl'))
    quarterly_indiv_nosite_cohorts.append(
        pd.read_pickle(f'~/data/ehr_sites/aou_quarterly_indiv_cohorts_no_site_{i}.pkl'))


def se_y(x, se_radius):
    n = 0.522
    return (KLD_CU * se_radius) * (1 - (x / (KLD_UC * se_radius)) ** n) ** (1 / n)


def objective_from_strategy(resource_vector, current_cohort, site_expected_demographics, objective_type, target_point):
    ### Find the value of the objective function for given inputs
    # INPUTS: resource vector (to test), recruited cohort, expected site demographics, type of objective, and target point if using an L2 objective
    # OUTPUTS: scalar objective function value

    resource_vector_prime = softmax(resource_vector)  # Transform resource vector into a probability distribution
    expected_recruitments = np.matmul(resource_vector_prime,
                                      site_expected_demographics)  # Expected recruitments for the given resource vector
    new_cohort = current_cohort + expected_recruitments  # Expected total cohort is current cohort with expected new cohort

    if objective_type == 'kld_c':  # KLD to census objective function
        kld_c = entropy(new_cohort, census_tot)
        return kld_c
    elif objective_type == 'kld_u':  # KLD to uniform objective function
        kld_u = entropy(new_cohort, np.ones_like(new_cohort))
        return kld_u
    elif objective_type == 'dist_opt':  # Distance to Optima objective function
        kld_c = entropy(new_cohort, census_tot)
        kld_u = entropy(new_cohort, np.ones_like(new_cohort))
        ser = ((kld_c / KLD_UC) ** 0.522 + (kld_u / KLD_CU) ** 0.522) ** (1 / 0.522)
        return ser
    elif objective_type == 'l2':  # L2 objective function to a specific point (could be on the Pareto frontier, 0, etc.)
        kld_c = entropy(new_cohort, census_tot)
        kld_u = entropy(new_cohort, np.ones_like(new_cohort))
        delta_x = kld_c - target_point[0]
        delta_y = kld_u - target_point[1]
        return ((delta_x) ** 2 + (delta_y) ** 2) ** 0.5


def optimize_recruitment_strategy(num_recruitments, current_cohort, site_categoricals, objective='dist_opt',
                                  upper_bound_factor=5, target_point=None):
    site_expected_demographics = site_categoricals * num_recruitments  # What each site would look like if all recruitment resources were allocated to it
    optimal_strat = minimize(fun=objective_from_strategy,
                             x0=np.ones(shape=len(site_categoricals)),
                             # Start with an even distribution of resources to all sites
                             args=(current_cohort, site_expected_demographics, objective, target_point),
                             # Other inputs to find the objective function
                             method='Powell',  # Powell's method is a fast an effective optimizer for what we need.
                             options={'xtol': 1 / num_recruitments, 'disp': False},
                             # Limit the resource vector search space to no less than 1 whole recruitment because fractional recruitments would not be possible
                             bounds=[(0, upper_bound_factor)] * len(site_categoricals),
                             # Bound the softmax allocation factors between 0 and a hyperparameter, to prevent full resource allocation to 1 site (which may not be realistic)
                             ).x
    attempted_recruitments = softmax(
        optimal_strat) * num_recruitments  # Turn the optimized resource vector into an actual recruitment number
    actual_recruitments = np.floor(attempted_recruitments).astype(
        int)  # Round down recruitments to nearest whole numbers
    remainder_recruitments = np.round(np.sum(attempted_recruitments - actual_recruitments)).astype(
        int)  # Find the number of leftover recruitments

    #  Assign remainder recruitments to the site with highest density
    actual_recruitments[np.argmax(actual_recruitments)] += remainder_recruitments
    return actual_recruitments


def simulate_recruitment(args):
    site_prior, prior_update, objective, rng_seed, upper_bound_factor, target_point, starting_cohort, starting_steps = args
    rng = np.random.default_rng(seed=rng_seed)

    #  Initialize the site dirichlets to zero
    site_dirichlet_dict = {}
    for mz3 in quarterly_cumu_cohorts[-1]['SITE']:
        site_dirichlet_dict[mz3] = np.zeros(60)  # There are 60 demographic groups
    site_dirichlets = pd.DataFrame.from_dict(
        {'SITE': site_dirichlet_dict.keys(), 'DIRICHLET': site_dirichlet_dict.values()})
    site_knowledge_dirichlets = site_dirichlets.copy(deep=True)

    # Initialize the KLDs, as well as the recruitment vector history
    kld_cs = []
    kld_us = []
    recruitment_history = []

    if starting_cohort is None:
        # Starting from scratch - cohort is 0
        cohort = np.zeros(60)
    else:
        # Starting somewhere intermediate in the recruitment process
        cohort = starting_cohort
        kld_cs.append(entropy(cohort, census_tot))
        kld_us.append(entropy(cohort, np.ones_like(cohort)))

    for i in range(len(quarterly_indiv_cohorts) - starting_steps):
        # i refers to the step that is currently about to be recruited, i-1 refers to the recruitment step that just occurred (if i != 0)

        #  Identify sites that have had some recruitments as of this prospective time tick, i.e., available sites
        sites_available_to_sample = quarterly_cumu_cohorts[i]['SITE'].to_numpy()
        dirichlets_available_sites = site_dirichlets['SITE'].isin(sites_available_to_sample)

        #  Identify sites that do not have any samples yet (e.g., this is the first time they show up as a site)
        sites_without_samples = site_dirichlets['DIRICHLET'][dirichlets_available_sites].apply(
            lambda row: row.sum()) == 0

        # Handle the initialization of never-before-seen sites
        if site_prior == 'census':
            #  Initialize new sites to a Dirichlet based on their Census values
            for new_mz3 in site_dirichlets['SITE'][dirichlets_available_sites][sites_without_samples]:
                site_dirichlets['DIRICHLET'].loc[site_dirichlets['SITE'] == new_mz3] = [aou_census_imputed.loc[new_mz3]]
        elif site_prior == 'noise':
            #  Initialize new sites to a noisy Dirichlet off their original values
            for new_mz3 in site_dirichlets['SITE'][dirichlets_available_sites][sites_without_samples]:
                mz3_orig_demo = quarterly_indiv_cohorts[i]['DEMOGRAPHICS'][
                    quarterly_indiv_cohorts[i]['SITE'] == new_mz3].to_numpy()
                mod_factor_magnitudes = 1 + 0.5 * rng.random(size=60)
                mod_factor_signs = rng.random(size=60) < 0.5
                mod_factor_magnitudes[mod_factor_signs] = 1 / mod_factor_magnitudes[mod_factor_signs]
                mz3_demo = mod_factor_magnitudes * mz3_orig_demo
                site_dirichlets.loc[site_dirichlets['SITE'] == new_mz3, 'DIRICHLET'] = [mz3_demo[0]]
        elif site_prior == 'no_noise':
            for new_mz3 in site_dirichlets['SITE'][dirichlets_available_sites][sites_without_samples]:
                mz3_orig_demo = quarterly_indiv_cohorts[i]['DEMOGRAPHICS'][
                    quarterly_indiv_cohorts[i]['SITE'] == new_mz3].to_numpy()
                mz3_demo = mz3_orig_demo
                site_dirichlets.loc[site_dirichlets['SITE'] == new_mz3, 'DIRICHLET'] = [mz3_demo[0]]
        elif site_prior == 'jeffreys':
            # Initialize new sites to an uninformative Jeffreys prior
            for new_mz3 in site_dirichlets['SITE'][dirichlets_available_sites][sites_without_samples]:
                site_dirichlets.loc[site_dirichlets['SITE'] == new_mz3, 'DIRICHLET'] = [0.5 * np.ones(60)]

        if prior_update == 'sim':
            # Update the priors using recruited individuals in simulation
            for existing_mz3 in site_dirichlets['SITE'][dirichlets_available_sites][~sites_without_samples]:
                site_dirichlets.loc[site_dirichlets['SITE'] == existing_mz3, 'DIRICHLET'] = [
                    site_knowledge_dirichlets['DIRICHLET'].loc[
                        site_knowledge_dirichlets['SITE'] == existing_mz3].item()]
        elif prior_update == 'actual':
            #  Update site priors with the actual AoURP recruitment values
            for existing_mz3 in site_dirichlets['SITE'][dirichlets_available_sites][~sites_without_samples]:
                mz3_recruited = quarterly_cumu_cohorts[i - 1]['DEMOGRAPHICS'][
                    quarterly_cumu_cohorts[i - 1]['SITE'] == existing_mz3].to_numpy()
                site_dirichlets.loc[site_dirichlets['SITE'] == existing_mz3, 'DIRICHLET'] = [mz3_recruited[0]]

        # Estimate demographics at each site by drawing from the Dirichlet prior distributions, a small float factor is added to keep all parameters above 0
        estimated_demographics = site_dirichlets['DIRICHLET'][dirichlets_available_sites].apply(
            lambda row: rng.dirichlet(row + 1e-10))
        # Identify the number of recruitments that actually occurred during this step so they can be re-allocated
        num_recruitments = quarterly_indiv_cohorts[i]['DEMOGRAPHICS'].sum().sum().astype(int)
        # Merging the available sites with the actual underlying distributions at these sites
        dirichlets_to_cumulative_mapping = site_dirichlets[dirichlets_available_sites].merge(quarterly_cumu_cohorts[i],
                                                                                             on='SITE', how='left')
        # Find the optimal recruitment strategy
        recruits = optimize_recruitment_strategy(num_recruitments, cohort, estimated_demographics, objective,
                                                 upper_bound_factor, target_point)

        recruitments_record = {}  # Keep a record of recruitments from each site

        #  Simulate the recruitment via random draw from the site's cumulative demographics distribution
        for site_id, num_recruits in enumerate(recruits):
            site_demographics = dirichlets_to_cumulative_mapping['DEMOGRAPHICS'].iloc[
                site_id]  # For each available site, find its demographics
            site_mz3 = dirichlets_to_cumulative_mapping['SITE'].iloc[
                site_id]  # Determine the matching location for that site ID
            recruitments_record[
                site_mz3] = num_recruits  # Add to the recruitment record the number of desired recruitments
            site_distribution = site_demographics / np.sum(
                site_demographics)  # Generate a categorical distribution for the site response distribution
            draws = rng.choice(60, size=num_recruits, replace=True,
                               p=site_distribution)  # Recruit from this respone distribution
            np.add.at(cohort, draws, 1)  # Add recruits to the cohort

            # If using in-simulation prior updates, update this with knowledge determined through recruitment
            if prior_update == 'sim':
                site_recruited_cohort = np.zeros(60)
                np.add.at(site_recruited_cohort, draws, 1)
                site_knowledge_dirichlets.loc[site_knowledge_dirichlets['SITE'] == site_mz3, 'DIRICHLET'].iat[
                    0] += site_recruited_cohort
        # Add the individuals not affiliated w/ any EHR site
        cohort += np.array(quarterly_indiv_nosite_cohorts[i])

        # Add the history of recruitments and evaluate the KLDs
        recruitment_history.append(recruitments_record)
        kld_cs.append(entropy(cohort, census_tot))
        kld_us.append(entropy(cohort, np.ones_like(cohort)))
        print(f'Step {i + 1} of {len(quarterly_indiv_cohorts) - starting_steps} done!')

    return {'site_prior': site_prior,
            'prior_update': prior_update,
            'obj': objective,
            'target_point': target_point,
            'kld_cs': kld_cs,
            'kld_us': kld_us,
            'policy': recruitment_history,
            'site_knowledge_dirichlets': site_knowledge_dirichlets,
            'final_cohort': cohort}


num_trials = 40
multiprocessing = True  # Set to False if multiprocessing is not desired or supported
prior = 'jeffreys'  # 'census' (requires Census conversion), 'noise', 'no_noise', 'jeffreys'
update_method = 'sim'  # 'sim', 'actual'
objective = 'l2'  # 'kld_c', 'kld_u', 'dist_opt', 'l2' (requires target point)
upper_bound_factor = 5
target_point = (0.04, 0.909)  # goal in (KLD_C, KLD_U) form
starting_cohort = None
starting_steps = 0

parameters = list(
    product([prior], [update_method], [objective], range(num_trials), [upper_bound_factor], [target_point],
            [starting_cohort], [starting_steps]))

if multiprocessing:
    with ProcessPool(num_trials) as pool:
        results = pool.map(simulate_recruitment, parameters)
else:
    results = []
    for parameter_set in parameters:
        results.append(simulate_recruitment(parameter_set))

with open(
        f'~/paper_results/ehr_site_{prior}_prior_{update_method}_update_{objective}_objective_{upper_bound_factor}_upper_bound_factor_004_0909_target_add_nosites_back.pickle',
        'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
