# Plots results once they are generated and saved
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pickle
import copy
import datetime

### Load in datasets as needed
aou_census_imputed = pd.read_pickle('~/data/census_generalized.pkl')
census_tot = aou_census_imputed.sum()
aou_row_level_data = pd.read_pickle('~/data/aourp_row_level_data.pkl')
quarterly_cumu_cohorts = []
quarterly_indiv_cohorts = []
for i in range(21):
    quarterly_cumu_cohorts.append(pd.read_pickle(f'~/data/ehr_sites/aou_quarterly_cumu_cohorts_{i}.pkl'))
    quarterly_indiv_cohorts.append(pd.read_pickle(f'~/data/ehr_sites/aou_quarterly_indiv_cohorts_{i}.pkl'))

pu = entropy(census_tot, np.ones_like(census_tot))
up = entropy(np.ones_like(census_tot), census_tot)

def cu_from_cp(cp):
    # Return KLD(C||U) on the superellipse given a KLD(C||P)
    n = 0.522
    return pu * (1 - (cp / up) ** n) ** (1 / n)


def cp_from_cu(cu):
    # Return KLD(C||U) on the superellipse given a KLD(C||P)
    n = 0.522
    return up * (1 - (cu / pu) ** n) ** (1 / n)

# Filter participant set to those with an EHR-based site
chronologic_aou = aou_row_level_data.loc[~aou_row_level_data['SITE'].isna()].sort_values(by='SURVEY_TIME')
datetimes = chronologic_aou['SURVEY_TIME']

# Generate per-participant cohorts of AoURP, sorted by time when participant joined
cohorts = []
cohorts.append(np.zeros(60))
i = 0

for index, row in chronologic_aou.iterrows():
    cohort = copy.deepcopy(cohorts[i])
    temp_cohort = np.zeros(shape=(3, 2, 5, 2))

    if row['AGE'] == '20-44':
        age_idx = 0
    elif row['AGE'] == '45-64':
        age_idx = 1
    else:
        age_idx = 2

    if row['GENDER'] == 'WOMAN':
        gender_idx = 0
    else:
        gender_idx = 1

    if row['RACE'] == 'ASIAN':
        race_idx = 0
    elif row['RACE'] == 'BLACK':
        race_idx = 1
    elif row['RACE'] == 'NHPI':
        race_idx = 2
    elif row['RACE'] == 'MIXED':
        race_idx = 3
    else:
        race_idx = 4

    if row['ETH'] == 'HL':
        eth_idx = 0
    else:
        eth_idx = 1

    temp_cohort[age_idx, gender_idx, race_idx, eth_idx] += 1
    cohort += temp_cohort.reshape(-1)
    cohorts.append(cohort)
    i += 1
cohorts = cohorts[1:]

total_count = 269862 # Participant count with defined EHR site
cohorts = np.array(cohorts)
census_array = np.array(census_tot)
repeated_census = np.repeat(census_array[:, np.newaxis], cohorts.shape[0], axis=1).T
kld_cs = entropy(cohorts, repeated_census, axis=1)
kld_us = entropy(cohorts, np.ones_like(cohorts), axis=1)

raveled_cohorts = cohorts.reshape(total_count, 3, 2, 5, 2)
raveled_census = census_tot.reshape(3, 2, 5, 2)

ur_proportions = []
for cohort in raveled_cohorts:
    total_cohort = cohort.sum()
    ur_by_age = cohort[2, :, :, :].sum() # 65+
    ur_by_race = cohort[0:2, :, 0:4, :].sum() # Non-white
    ur_by_eth = cohort[0:2, :, 4, 0].sum() # White & H/L
    ur_proportions.append((ur_by_age + ur_by_race + ur_by_eth) / total_cohort)

aou_row_level_sites_only = aou_row_level_data.loc[~aou_row_level_data['SITE'].isna()]
recruits_by_month = aou_row_level_sites_only.groupby(pd.Grouper(key='SURVEY_TIME', freq='4W')).count().person_id

age_proportions = raveled_cohorts.sum(axis=(2, 3, 4)).T / (np.arange(total_count) + 1)
gender_proportions = raveled_cohorts.sum(axis=(1, 3, 4)).T / (np.arange(total_count) + 1)
race_proportions = raveled_cohorts.sum(axis=(1, 2, 4)).T / (np.arange(total_count) + 1)
eth_proportions = raveled_cohorts.sum(axis=(1, 2, 3)).T / (np.arange(total_count) + 1)

age_census = raveled_census.sum(axis=(1, 2, 3)) / raveled_census.sum()
gender_census = raveled_census.sum(axis=(0, 2, 3)) / raveled_census.sum()
race_census = raveled_census.sum(axis=(0, 1, 3)) / raveled_census.sum()
eth_census = raveled_census.sum(axis=(0, 1, 2)) / raveled_census.sum()

### Figure 1
with matplotlib.rc_context({'font.family': 'sans', 'font.size': 8, 'figure.dpi': 300}):
    fig, axs = plt.subplots(figsize=[7, 4], nrows=5, ncols=2, sharex=True)

    ax = axs[0, 0]
    ax.set_xlim([datetime.date(2017, 6, 1), datetime.date(2022, 7, 1)])
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.plot(datetimes[100:], kld_cs[100:], '-', linewidth=2, color='tab:red', label='Representativeness (lower better)')
    ax.legend()
    ax.set_ylabel('KLD(C||P)')

    ax = axs[1, 0]
    ax.set_xlim([datetime.date(2017, 6, 1), datetime.date(2022, 7, 1)])
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.plot(datetimes[100:], kld_us[100:], '-', linewidth=2, color='tab:blue', label='Coverage (lower better)')
    ax.legend()
    ax.set_ylabel('KLD(C||U)')

    ax = axs[2, 0]
    ax.set_xlim([datetime.date(2017, 6, 1), datetime.date(2022, 7, 1)])
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.axhline(y=75, color='#B0B0B0', label='75% Target', lw=1)
    ax.plot(datetimes[100:], 100 * np.array(ur_proportions[100:]), '-', linewidth=2, color='tab:gray',
            label='% Historically Underrepresented')
    ax.legend(loc=4)
    ax.set_ylabel('%')

    ax = axs[3, 0]
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 1))
    ax.bar(recruits_by_month.index, recruits_by_month, width=pd.Timedelta(weeks=4), color='tab:olive')
    ax.set_ylabel('Recruitments')
    ax.set_yticks([0, 2500, 5000, 7500, 10000])
    ax.set_ylim([0, 10000])

    ax = axs[4, 0]
    ax.set_xlim([datetime.date(2017, 6, 1), datetime.date(2022, 7, 1)])
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.axhline(y=100 * age_census[2], color='#CC6600', label='Census Age 65+', lw=1)
    ax.plot(datetimes[100:], 100 * age_proportions[2, 100:], '-', linewidth=2, color='tab:brown', label='AoURP Age 65+')
    ax.legend(loc=4)
    ax.set_ylabel('%')

    ax = axs[0, 1]
    ax.set_xlim([datetime.date(2017, 6, 1), datetime.date(2022, 7, 1)])
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.axhline(y=100 * race_census[0], color='#FF9933', label='Census Race Asian', lw=1)
    ax.plot(datetimes[100:], 100 * race_proportions[0, 100:], '-', linewidth=2, color='tab:orange',
            label='AoURP Race Asian')
    ax.legend()
    ax.set_ylabel('%')

    ax = axs[1, 1]
    ax.set_xlim([datetime.date(2017, 6, 1), datetime.date(2022, 7, 1)])
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.axhline(y=100 * race_census[1], color='#00CC00', label='Census Race Black', lw=1)
    ax.plot(datetimes[100:], 100 * race_proportions[1, 100:], '-', linewidth=2, color='tab:green',
            label='AoURP Race Black')
    ax.legend()
    ax.set_ylabel('%')

    ax = axs[2, 1]
    ax.set_xlim([datetime.date(2017, 6, 1), datetime.date(2022, 7, 1)])
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.axhline(y=100 * race_census[2], color='#BF80FF', label='Census Race NH/PI', lw=1)
    ax.plot(datetimes[100:], 100 * race_proportions[2, 100:], '-', linewidth=2, color='tab:purple',
            label='AoURP Race NH/PI')
    ax.legend(loc=4)
    ax.set_ylabel('%')

    ax = axs[3, 1]
    ax.set_xlim([datetime.date(2017, 6, 1), datetime.date(2022, 7, 1)])
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.axhline(y=100 * race_census[3], color='#FFB3FF', label='Census Race Two or More', lw=1)
    ax.plot(datetimes[100:], 100 * race_proportions[3, 100:], '-', linewidth=2, color='tab:pink',
            label='AoURP Race Two or More')
    ax.legend()
    ax.set_ylabel('%')

    ax = axs[4, 1]
    ax.set_xlim([datetime.date(2017, 6, 1), datetime.date(2022, 7, 1)])
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.axhline(y=100 * eth_census[0], color='#66FFFF', label='Census Ethnicity H/L', lw=1)
    ax.plot(datetimes[100:], 100 * eth_proportions[0, 100:], '-', linewidth=2, color='tab:cyan',
            label='AoURP Ethnicity H/L')
    ax.legend(loc=4)
    ax.set_ylabel('%')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.28, wspace=0.18)
    plt.show()

# AoURP final cohort demographics compared to Census and uniform
ll = 1e-5
ul = 3e-1

census_distribution = (census_tot / census_tot.sum()).reshape([3, 2, 5, 2])
uniform_distribution = 1 / 60 * np.ones(60)
aourp_recruit_distribution = quarterly_cumu_cohorts[-1]['DEMOGRAPHICS'].sum(axis=0) / quarterly_cumu_cohorts[-1][
    'DEMOGRAPHICS'].sum().sum()

site_specific_demos_1 = np.array(quarterly_cumu_cohorts[-1]['DEMOGRAPHICS'].sum()).reshape([3, 2, 5, 2])

race_labels = ['Asian', 'Black', 'NH/PI', 'T o M', 'White']
eth_labels = ['H/L', 'NH/L']
gender_labels = ['F', 'M']
age_labels = ['20-44', '45-64', '65+']

gen_demos_1 = []
gen_demos_2 = []
census_gen_demos = []
uniform_demos = []
labels = []

# Print whether each group is within the proportions set by the Census values and uniform values
print(((aourp_recruit_distribution < census_distribution.reshape(-1)) & (
            aourp_recruit_distribution > uniform_distribution)) |
      ((aourp_recruit_distribution > census_distribution.reshape(-1)) & (
                  aourp_recruit_distribution < uniform_distribution)))

for race_idx in range(5):
    for eth_idx in range(2):
        for gender_idx in range(2):
            for age_idx in range(3):
                if site_specific_demos_1[age_idx, gender_idx, race_idx, eth_idx] < 20:
                    age_summed_demos_1 = site_specific_demos_1[:, gender_idx, race_idx, eth_idx].sum()
                    if age_summed_demos_1 < 20:
                        age_gender_summed_demos_1 = site_specific_demos_1[:, :, race_idx, eth_idx].sum()
                        if age_gender_summed_demos_1 < 20:
                            age_gender_eth_summed_demos_1 = site_specific_demos_1[:, :, race_idx, :].sum()
                            if age_gender_eth_summed_demos_1 < 20:
                                pass
                            else:
                                gen_demos_1.append(site_specific_demos_1[:, :, race_idx, :].sum())
                                census_gen_demos.append(census_distribution[:, :, race_idx, :].sum())
                                labels.append(f'{race_labels[race_idx]}')
                                uniform_demos.append(12)
                        else:
                            gen_demos_1.append(site_specific_demos_1[:, :, race_idx, eth_idx].sum())
                            census_gen_demos.append(census_distribution[:, :, race_idx, eth_idx].sum())
                            labels.append(f'{race_labels[race_idx]} {eth_labels[eth_idx]}')
                            uniform_demos.append(6)
                    else:
                        gen_demos_1.append(site_specific_demos_1[:, gender_idx, race_idx, eth_idx].sum())
                        census_gen_demos.append(census_distribution[:, gender_idx, race_idx, eth_idx].sum())
                        labels.append(f'{race_labels[race_idx]} {eth_labels[eth_idx]} {gender_labels[gender_idx]}')
                        uniform_demos.append(3)
                else:
                    gen_demos_1.append(site_specific_demos_1[age_idx, gender_idx, race_idx, eth_idx])
                    census_gen_demos.append(census_distribution[age_idx, gender_idx, race_idx, eth_idx])
                    labels.append(
                        f'{race_labels[race_idx]} {eth_labels[eth_idx]} {gender_labels[gender_idx]} {age_labels[age_idx]}')
                    uniform_demos.append(1)

# Identify duplicate aggregations
duplicate_idxs = []
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        if labels[i] == labels[j]:
            duplicate_idxs.append(j)

# Remove duplicate aggregations from all lists
for pops, dup_idx in enumerate(np.unique(duplicate_idxs)):
    gen_demos_1.pop(dup_idx - pops)
    census_gen_demos.pop(dup_idx - pops)
    uniform_demos.pop(dup_idx - pops)
    labels.pop(dup_idx - pops)

census_sort = np.flip(np.argsort(census_gen_demos))

site_specific_demos_1 = site_specific_demos_1 / site_specific_demos_1.sum()
site_1_kld_p = entropy(site_specific_demos_1.reshape(-1), census_distribution.reshape(-1))
site_1_kld_u = entropy(site_specific_demos_1.reshape(-1), uniform_distribution)
higher_line = np.maximum(np.array(census_gen_demos), np.array(uniform_demos) / 60)
lower_line = np.minimum(np.array(census_gen_demos), np.array(uniform_demos) / 60)

### Figure 2
with matplotlib.rc_context({'font.family': 'sans', 'font.size': 8, 'figure.dpi': 300}):
    fig, axs = plt.subplots(nrows=2, figsize=[6, 3.7], height_ratios=[3.5, 1])

    ax = axs[0]
    ax.step(np.arange(len(census_gen_demos)) + 0.5, np.array(uniform_demos) / 60, 'k--', lw=1, where='post',
            label='Uniform Group Proportion')
    ax.hlines(np.array(uniform_demos)[-1] / 60, len(census_gen_demos) - 0.5, len(census_gen_demos) + 0.5, 'k', lw=1,
              linestyles='--')
    ax.step(np.arange(len(census_gen_demos)) + 0.5, np.array(census_gen_demos), 'k', lw=1, where='post',
            label='Census Group Proportion')
    ax.hlines(np.array(census_gen_demos)[-1], len(census_gen_demos) - 0.5, len(census_gen_demos) + 0.5, 'k', lw=1)

    ax.plot(np.arange(len(census_gen_demos)) + 1,
            np.array(gen_demos_1) / np.array(gen_demos_1).sum(),
            'o', markersize=2.5, color='k',
            label=f'AoURP Cohort')

    ax.fill_between(np.arange(len(census_gen_demos)) + 0.5,
                    np.array(uniform_demos) / 60,
                    np.array(census_gen_demos),
                    color='tab:purple', alpha=0.15, step='post',
                    label='Target Zone')
    ax.fill_between([len(census_gen_demos) - 0.5, len(census_gen_demos) + 0.5],
                    [np.array(uniform_demos)[-1] / 60, np.array(uniform_demos)[-1] / 60],
                    [np.array(census_gen_demos)[-1], np.array(census_gen_demos)[-1]],
                    color='tab:purple', alpha=0.15, step='post')

    ax.fill_between(np.arange(len(census_gen_demos)) + 0.5,
                    lower_line,
                    ll,
                    color='tab:red', alpha=0.15, step='post')
    ax.fill_between([len(census_gen_demos) - 0.5, len(census_gen_demos) + 0.5],
                    lower_line[-1],
                    ll,
                    color='tab:red', alpha=0.15, step='post')
    ax.fill_between(np.arange(len(census_gen_demos)) + 0.5,
                    higher_line,
                    ul,
                    color='tab:blue', alpha=0.15, step='post')
    ax.fill_between([len(census_gen_demos) - 0.5, len(census_gen_demos) + 0.5],
                    higher_line[-1],
                    ul,
                    color='tab:blue', alpha=0.15, step='post')

    ax.set_yscale('log')
    ax.set_xlim([0.5, len(census_gen_demos) + 0.5])
    ax.set_ylim([ll, ul])
    ax.set_xticks(ticks=np.arange(len(census_gen_demos) + 1) + 0.5, labels=None, minor=True)
    ax.set_xticks(ticks=np.arange(len(census_gen_demos)) + 1, labels=np.array([label[12:].strip() for label in labels]),
                  minor=False, rotation=90, fontsize=6)
    ax.grid(axis='x', which='minor', color='0.8')
    ax.set_ylabel('Group Proportion (log scale)')
    ax.legend(loc=4, fontsize=7)
    ax.annotate('Underrepresented', xy=(0.5, ll), xytext=(1, 2), textcoords='offset points', fontsize=7)
    ax.annotate('Overrepresented', xy=(0.5, ul), xytext=(1, -8), textcoords='offset points', fontsize=7)

    ax = axs[1]
    color_dict = {'Asian': 0, 'Black': 1, 'NH/PI': 2, 'T o M': 3, 'White': 4,
                  'H/L': 5, 'NH/L': 6,
                  'F': 7, 'M': 8,
                  '20-44': 9, '45-64': 10, '65+': 11
                  }
    parsed_labels = np.flip(np.array([(color_dict[label[:5].strip()], color_dict[label[6:10].strip()],
                                       color_dict[label[10:12].strip()], color_dict[label[12:].strip()]) for label in
                                      labels]).T)
    ax.imshow(parsed_labels[1:, :], cmap='tab20', aspect=1.5)
    ax.axis('off')
    for i in range(10):
        ax.annotate('F', xy=(6 * i, 0), xytext=(4, -2), textcoords='offset points', fontsize=6)
        ax.annotate('M', xy=(6 * i + 3, 0), xytext=(3, -2), textcoords='offset points', fontsize=6)
    for i in range(5):
        ax.annotate('H/L', xy=(12 * i, 1), xytext=(10, -2), textcoords='offset points', fontsize=6)
        ax.annotate('NH/L', xy=(12 * i + 6, 1), xytext=(8, -2), textcoords='offset points', fontsize=6)
    ax.annotate('Asian', xy=(0, 2), xytext=(24, -2), textcoords='offset points', fontsize=6, color='white')
    ax.annotate('Black', xy=(12, 2), xytext=(24, -2), textcoords='offset points', fontsize=6)
    ax.annotate('NH/PI', xy=(24, 2), xytext=(24, -2), textcoords='offset points', fontsize=6)
    ax.annotate('Two or More', xy=(36, 2), xytext=(14, -2), textcoords='offset points', fontsize=6, color='white')
    ax.annotate('White', xy=(48, 2), xytext=(24, -2), textcoords='offset points', fontsize=6, color='white')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    plt.show()

# Top 6 sites:
n_sites = 6
sites_of_interest = quarterly_cumu_cohorts[-1].loc[(quarterly_cumu_cohorts[-1]['SITE'] == '783') | (quarterly_cumu_cohorts[-1]['SITE'] == '195') | (quarterly_cumu_cohorts[-1]['SITE'] == '481') | (quarterly_cumu_cohorts[-1]['SITE'] == '267') | (quarterly_cumu_cohorts[-1]['SITE'] == '321') | (quarterly_cumu_cohorts[-1]['SITE'] == '199')].reset_index()
site_cohorts = np.vstack(sites_of_interest['DEMOGRAPHICS'])
site_sums = site_cohorts.sum(axis=1, keepdims=True)
site_distributions = (site_cohorts / site_sums).reshape(n_sites, 3, 2, 5, 2)
census_distribution = (census_tot / census_tot.sum()).reshape(3, 2, 5, 2)

reordered_site_distributions = []
reordered_census = []
labels = []

site_size_sort = np.flip(site_sums.squeeze().argsort())
site_distributions = site_distributions[site_size_sort, :, :, :, :]

race_labels = ['Asian', 'Black', 'NH/PI', 'T o M', 'White']
eth_labels = ['H/L', 'NH/L']
gender_labels = ['F', 'M']
age_labels = ['20-44', '45-64', '65+']

for race_idx in range(5):
    for eth_idx in range(2):
        for gender_idx in range(2):
            for age_idx in range(3):
                reordered_site_distributions.append(site_distributions[:, age_idx, gender_idx, race_idx, eth_idx])
                reordered_census.append(census_distribution[age_idx, gender_idx, race_idx, eth_idx])
                labels.append(
                    f'{race_labels[race_idx]} {eth_labels[eth_idx]} {gender_labels[gender_idx]} {age_labels[age_idx]}')

site_ratios = np.array(reordered_site_distributions).T / np.array(reordered_census)[None, :]

### Figure 3a
with matplotlib.rc_context({'font.family': 'sans', 'font.size': 7, 'figure.dpi': 300}):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[6, 3], height_ratios=[3.5, 1], width_ratios=[10, 1])
    axs[0, 1].axis('off')
    axs[1, 1].axis('off')

    ax = axs[0, 0]
    cmap = matplotlib.cm.RdBu
    cmap.set_bad(color='#500000')
    im = ax.imshow(np.log10(site_ratios), aspect='auto', cmap=cmap, vmin=-2, vmax=2)
    ax.set_ylabel('Sites (ordered by descending size)')
    ax.set_xticks(ticks=np.arange(60), labels=np.array([label[12:].strip() for label in labels]), minor=False,
                  rotation=90, fontsize=6)
    ax.set_yticks(ticks=np.arange(n_sites), labels=sites_of_interest['SITE'][site_size_sort].to_numpy(), fontsize=7,
                  rotation=90, va='center')
    clb = fig.colorbar(im, ax=axs[0, 1], aspect=15, fraction=1, ticks=[-2., -1., 0., 1., 2.],
                       format=lambda x, _: f"{np.power(10, x)}x")
    clb.set_label(label='Subgroup Representation (relative to Census)', size=7)
    clb.ax.tick_params(labelsize=7)

    ax = axs[1, 0]
    color_dict = {'Asian': 0, 'Black': 1, 'NH/PI': 2, 'T o M': 3, 'White': 4,
                  'H/L': 5, 'NH/L': 6,
                  'F': 7, 'M': 8,
                  '20-44': 9, '45-64': 10, '65+': 11
                  }
    parsed_labels = np.flip(np.array([(color_dict[label[:5].strip()], color_dict[label[6:10].strip()],
                                       color_dict[label[10:12].strip()], color_dict[label[12:].strip()]) for label in
                                      labels]).T)
    ax.imshow(parsed_labels[1:, :], cmap='tab20', aspect=1.5)
    ax.axis('off')
    for i in range(10):
        ax.annotate('F', xy=(6 * i, 0), xytext=(3, -2), textcoords='offset points', fontsize=6)
        ax.annotate('M', xy=(6 * i + 3, 0), xytext=(2.5, -2), textcoords='offset points', fontsize=6)
    for i in range(5):
        ax.annotate('H/L', xy=(12 * i, 1), xytext=(10, -2), textcoords='offset points', fontsize=6)
        ax.annotate('NH/L', xy=(12 * i + 6, 1), xytext=(7, -2), textcoords='offset points', fontsize=6)
    ax.annotate('Asian', xy=(0, 2), xytext=(20, -2), textcoords='offset points', fontsize=6, color='white')
    ax.annotate('Black', xy=(12, 2), xytext=(20, -2), textcoords='offset points', fontsize=6)
    ax.annotate('NH/PI', xy=(24, 2), xytext=(20, -2), textcoords='offset points', fontsize=6)
    ax.annotate('Two or More', xy=(36, 2), xytext=(10, -2), textcoords='offset points', fontsize=6, color='white')
    ax.annotate('White', xy=(48, 2), xytext=(20, -2), textcoords='offset points', fontsize=6, color='white')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.22)
    plt.show()

site_cohorts = np.vstack(quarterly_cumu_cohorts[-1]['DEMOGRAPHICS'])
site_sums = site_cohorts.sum(axis=1, keepdims=True)
site_distributions = (site_cohorts / site_sums).reshape(50, 3, 2, 5, 2)
census_distribution = (census_tot / census_tot.sum()).reshape(3, 2, 5, 2)

reordered_site_distributions = []
reordered_census = []
labels = []

site_size_sort = np.flip(site_sums.squeeze().argsort())
site_distributions = site_distributions[site_size_sort, :, :, :, :]

race_labels = ['Asian', 'Black', 'NH/PI', 'T o M', 'White']
eth_labels = ['H/L', 'NH/L']
gender_labels = ['F', 'M']
age_labels = ['20-44', '45-64', '65+']

for race_idx in range(5):
    for eth_idx in range(2):
        for gender_idx in range(2):
            for age_idx in range(3):
                reordered_site_distributions.append(site_distributions[:, age_idx, gender_idx, race_idx, eth_idx])
                reordered_census.append(census_distribution[age_idx, gender_idx, race_idx, eth_idx])
                labels.append(
                    f'{race_labels[race_idx]} {eth_labels[eth_idx]} {gender_labels[gender_idx]} {age_labels[age_idx]}')

site_ratios = np.array(reordered_site_distributions).T / np.array(reordered_census)[None, :]

### Figure 3b
with matplotlib.rc_context({'font.family': 'sans', 'font.size': 7, 'figure.dpi': 300}):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[6, 4.6], height_ratios=[3.5, 1], width_ratios=[10, 1])
    axs[0, 1].axis('off')
    axs[1, 1].axis('off')

    ax = axs[0, 0]
    cmap = matplotlib.cm.RdBu
    cmap.set_bad(color='#500000')
    im = ax.imshow(np.log10(site_ratios), aspect='auto', cmap=cmap, vmin=-2, vmax=2)
    ax.set_ylabel('Sites (ordered by descending size)')
    ax.set_xticks(ticks=np.arange(60), labels=np.array([label[12:].strip() for label in labels]), minor=False,
                  rotation=90, fontsize=6)
    ax.set_yticks(ticks=np.arange(50), labels=quarterly_cumu_cohorts[-1]['SITE'][site_size_sort].to_numpy(), fontsize=5)
    clb = fig.colorbar(im, ax=axs[0, 1], aspect=15, fraction=1, ticks=[-2., -1., 0., 1., 2.],
                       format=lambda x, _: f"{np.power(10, x)}x")
    clb.set_label(label='Subgroup Representation (relative to Census)', size=7)
    clb.ax.tick_params(labelsize=7)

    ax = axs[1, 0]
    color_dict = {'Asian': 0, 'Black': 1, 'NH/PI': 2, 'T o M': 3, 'White': 4,
                  'H/L': 5, 'NH/L': 6,
                  'F': 7, 'M': 8,
                  '20-44': 9, '45-64': 10, '65+': 11
                  }
    parsed_labels = np.flip(np.array([(color_dict[label[:5].strip()], color_dict[label[6:10].strip()],
                                       color_dict[label[10:12].strip()], color_dict[label[12:].strip()]) for label in
                                      labels]).T)
    ax.imshow(parsed_labels[1:, :], cmap='tab20', aspect=1.5)
    ax.axis('off')
    for i in range(10):
        ax.annotate('F', xy=(6 * i, 0), xytext=(3, -2), textcoords='offset points', fontsize=6)
        ax.annotate('M', xy=(6 * i + 3, 0), xytext=(2.5, -2), textcoords='offset points', fontsize=6)
    for i in range(5):
        ax.annotate('H/L', xy=(12 * i, 1), xytext=(10, -2), textcoords='offset points', fontsize=6)
        ax.annotate('NH/L', xy=(12 * i + 6, 1), xytext=(7, -2), textcoords='offset points', fontsize=6)
    ax.annotate('Asian', xy=(0, 2), xytext=(20, -2), textcoords='offset points', fontsize=6, color='white')
    ax.annotate('Black', xy=(12, 2), xytext=(20, -2), textcoords='offset points', fontsize=6)
    ax.annotate('NH/PI', xy=(24, 2), xytext=(20, -2), textcoords='offset points', fontsize=6)
    ax.annotate('Two or More', xy=(36, 2), xytext=(10, -2), textcoords='offset points', fontsize=6, color='white')
    ax.annotate('White', xy=(48, 2), xytext=(20, -2), textcoords='offset points', fontsize=6, color='white')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.03)
    plt.show()

# NY 3 sites:
n_sites = 3
sites_of_interest = quarterly_cumu_cohorts[-1].loc[
    (quarterly_cumu_cohorts[-1]['SITE'] == '783') | (quarterly_cumu_cohorts[-1]['SITE'] == '752') | (
                quarterly_cumu_cohorts[-1]['SITE'] == '689')].reset_index()
site_cohorts = np.vstack(sites_of_interest['DEMOGRAPHICS'])
site_sums = site_cohorts.sum(axis=1, keepdims=True)
site_distributions = (site_cohorts / site_sums).reshape(n_sites, 3, 2, 5, 2)
census_distribution = (census_tot / census_tot.sum()).reshape(3, 2, 5, 2)

reordered_site_distributions = []
reordered_census = []
labels = []

site_size_sort = np.flip(site_sums.squeeze().argsort())
site_distributions = site_distributions[site_size_sort, :, :, :, :]

race_labels = ['Asian', 'Black', 'NH/PI', 'T o M', 'White']
eth_labels = ['H/L', 'NH/L']
gender_labels = ['F', 'M']
age_labels = ['20-44', '45-64', '65+']

for race_idx in range(5):
    for eth_idx in range(2):
        for gender_idx in range(2):
            for age_idx in range(3):
                reordered_site_distributions.append(site_distributions[:, age_idx, gender_idx, race_idx, eth_idx])
                reordered_census.append(census_distribution[age_idx, gender_idx, race_idx, eth_idx])
                labels.append(
                    f'{race_labels[race_idx]} {eth_labels[eth_idx]} {gender_labels[gender_idx]} {age_labels[age_idx]}')

site_ratios = np.array(reordered_site_distributions).T / np.array(reordered_census)[None, :]

### Figure 4
with matplotlib.rc_context({'font.family': 'sans', 'font.size': 7, 'figure.dpi': 300}):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[6, 3], height_ratios=[3.5, 1], width_ratios=[10, 1])
    axs[0, 1].axis('off')
    axs[1, 1].axis('off')

    ax = axs[0, 0]
    cmap = matplotlib.cm.RdBu
    cmap.set_bad(color='#500000')
    im = ax.imshow(np.log10(site_ratios), aspect='auto', cmap=cmap, vmin=-2, vmax=2)
    ax.set_ylabel('Sites (ordered by descending size)')
    ax.set_xticks(ticks=np.arange(60), labels=np.array([label[12:].strip() for label in labels]), minor=False,
                  rotation=90, fontsize=6)
    ax.set_yticks(ticks=np.arange(n_sites), labels=sites_of_interest['SITE'][site_size_sort].to_numpy(), fontsize=7,
                  rotation=90, va='center')
    clb = fig.colorbar(im, ax=axs[0, 1], aspect=15, fraction=1, ticks=[-2., -1., 0., 1., 2.],
                       format=lambda x, _: f"{np.power(10, x)}x")
    clb.set_label(label='Subgroup Representation (relative to Census)', size=7)
    clb.ax.tick_params(labelsize=7)

    ax = axs[1, 0]
    color_dict = {'Asian': 0, 'Black': 1, 'NH/PI': 2, 'T o M': 3, 'White': 4,
                  'H/L': 5, 'NH/L': 6,
                  'F': 7, 'M': 8,
                  '20-44': 9, '45-64': 10, '65+': 11
                  }
    parsed_labels = np.flip(np.array([(color_dict[label[:5].strip()], color_dict[label[6:10].strip()],
                                       color_dict[label[10:12].strip()], color_dict[label[12:].strip()]) for label in
                                      labels]).T)
    ax.imshow(parsed_labels[1:, :], cmap='tab20', aspect=1.5)
    ax.axis('off')
    for i in range(10):
        ax.annotate('F', xy=(6 * i, 0), xytext=(3, -2), textcoords='offset points', fontsize=6)
        ax.annotate('M', xy=(6 * i + 3, 0), xytext=(2.5, -2), textcoords='offset points', fontsize=6)
    for i in range(5):
        ax.annotate('H/L', xy=(12 * i, 1), xytext=(10, -2), textcoords='offset points', fontsize=6)
        ax.annotate('NH/L', xy=(12 * i + 6, 1), xytext=(7, -2), textcoords='offset points', fontsize=6)
    ax.annotate('Asian', xy=(0, 2), xytext=(20, -2), textcoords='offset points', fontsize=6, color='white')
    ax.annotate('Black', xy=(12, 2), xytext=(20, -2), textcoords='offset points', fontsize=6)
    ax.annotate('NH/PI', xy=(24, 2), xytext=(20, -2), textcoords='offset points', fontsize=6)
    ax.annotate('Two or More', xy=(36, 2), xytext=(10, -2), textcoords='offset points', fontsize=6, color='white')
    ax.annotate('White', xy=(48, 2), xytext=(20, -2), textcoords='offset points', fontsize=6, color='white')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.22)
    plt.show()


# Load in results from saved experiments
prior = 'jeffreys'  # 'census' (requires Census conversion), 'noise', 'jeffreys'
update_method = 'sim'  # 'sim', 'actual'
objective = 'l2'  # 'kld_c', 'kld_u', 'dist_opt', 'l2' (requires target point)
upper_bound_factor = 5
target_point = None
starting_cohort = None
starting_steps = 0

quarterly_cumu_cohorts = []
quarterly_indiv_cohorts = []
for i in range(21):
    quarterly_cumu_cohorts.append(pd.read_pickle(f'~/data/ehr_sites/aou_quarterly_cumu_cohorts_{i}.pkl'))
with open(
        f'~/paper_results/ehr_site_{prior}_prior_{update_method}_update_{objective}_objective_{upper_bound_factor}_upper_bound_factor_pareto.pickle',
        'rb') as handle:
    results = pickle.load(handle)
results_df = pd.DataFrame(results)

actual_kld_cs = np.array([entropy(cohort['DEMOGRAPHICS'].sum(axis=0), census_tot) for cohort in quarterly_cumu_cohorts])
actual_kld_us = np.array(
    [entropy(cohort['DEMOGRAPHICS'].sum(axis=0), np.ones_like(census_tot)) for cohort in quarterly_cumu_cohorts])

final_kld_cs = np.vstack(results_df['kld_cs'])[:, -1]
final_kld_us = np.vstack(results_df['kld_us'])[:, -1]

### Figure 5
with matplotlib.rc_context({'font.family': 'sans', 'font.size': 8, 'figure.dpi': 300}):
    fig, axs = plt.subplots(ncols=1, figsize=[4, 3])
    ax = axs
    ax.plot(final_kld_cs, final_kld_us, 'o-', markersize=3, label='Simulation Final Cohorts', zorder=200)
    ax.plot(final_kld_cs[8], final_kld_us[8], 'o-', markersize=4.5, zorder=100, color='tab:pink',
            label='Selected Point')
    ax.plot(actual_kld_cs[-1], actual_kld_us[-1], 's', markersize=3, color='black', label='AoURP Final Cohort')
    ax.plot(np.arange(0, 0.35, 0.001), cu_from_cp(np.arange(0, 0.35, 0.001)), color='tab:red', label='Theoretic Optima',
            zorder=200)
    ax.set_xlabel('KLD(C||P)')
    ax.set_ylabel('KLD(C||U)')
    for i in range(len(final_kld_cs)):
        if i == 0:
            ax.plot([results_df['target_point'].iloc[i][0], final_kld_cs[i]],
                    [results_df['target_point'].iloc[i][1], final_kld_us[i]], color='tab:cyan', lw=0.7, zorder=100,
                    label='Simulated Cohort to Target')
        elif i == 8:
            ax.plot([results_df['target_point'].iloc[i][0], final_kld_cs[i]],
                    [results_df['target_point'].iloc[i][1], final_kld_us[i]], color='tab:pink', lw=1, zorder=100)
        else:
            ax.plot([results_df['target_point'].iloc[i][0], final_kld_cs[i]],
                    [results_df['target_point'].iloc[i][1], final_kld_us[i]], color='tab:cyan', lw=0.7, zorder=100)
    ax.set_xlim([0, 0.32])
    ax.legend(loc=3).set_zorder(201)
    ax.hlines(actual_kld_us[-1], 0.114, actual_kld_cs[-1], color='k', linestyles='--', zorder=100)
    ax.vlines(actual_kld_cs[-1], 0.902, actual_kld_us[-1], color='k', linestyles='--', zorder=100)
    ax.set_xticks(ticks=np.arange(0, 0.35, 0.05), minor=False)
    ax.set_xticks(ticks=np.arange(0, 0.33, 0.01), minor=True)
    ax.set_yticks(ticks=np.arange(0.4, 1.4, 0.1), minor=False)
    ax.set_yticks(ticks=np.arange(0.4, 1.32, 0.02), minor=True)
    plt.tight_layout()
    plt.show()

### Load in main paper paper_results
prior = 'jeffreys'  # 'census' (requires Census conversion), 'noise', 'jeffreys'
update_method = 'sim'  # 'sim', 'actual'
objective = 'l2'  # 'kld_c', 'kld_u', 'dist_opt', 'l2' (requires target point)
upper_bound_factor = 5
target_point = None
starting_cohort = None
starting_steps = 0

# EHR site cohorts
quarterly_cumu_cohorts = []
# quarterly_indiv_cohorts = []
for i in range(21):
    # For ZIP3 site cohorts
    # quarterly_cumu_cohorts.append(pd.read_pickle(f'~/data/aou_quarterly_cumu_cohorts_{i}.pkl'))
    # For EHR site cohorts, not including non-site participants
    quarterly_cumu_cohorts.append(pd.read_pickle(f'~/data/ehr_sites/aou_quarterly_cumu_cohorts_{i}.pkl'))
    # For EHR site cohorts, including non-site participants
    # quarterly_cumu_cohorts.append(pd.read_pickle(f'~/data/ehr_sites/aou_quarterly_cumu_cohorts_including_nan_{i}.pkl'))

# for i in range(6):
#     if i < 5:
#         quarterly_cumu_cohorts.append(
#             pd.read_pickle(f'~/data/ehr_sites/aou_quarterly_cumu_cohorts_{4 * i + 3}.pkl'))
#     else:
#         quarterly_cumu_cohorts.append(
#             pd.read_pickle(f'~/data/ehr_sites/aou_quarterly_cumu_cohorts_{4 * i}.pkl'))

with open(
        f'~/paper_results/ehr_site_{prior}_prior_{update_method}_update_{objective}_objective_{upper_bound_factor}_upper_bound_factor_004_0909_yearly.pickle',
        'rb') as handle:
    results = pickle.load(handle)

results_df = pd.DataFrame(results)

### Process final cohort KLD_C, KLD_U, and underrepresented % for simulation and historic AoURP cohorts
actual_kld_cs = np.array([entropy(cohort['DEMOGRAPHICS'].sum(axis=0), census_tot) for cohort in quarterly_cumu_cohorts])
actual_kld_us = np.array(
    [entropy(cohort['DEMOGRAPHICS'].sum(axis=0), np.ones_like(census_tot)) for cohort in quarterly_cumu_cohorts])
kld_c_mean = np.array(list(results_df['kld_cs'])).mean(axis=0)
kld_c_sem = np.array(list(results_df['kld_cs'])).std(axis=0) / np.sqrt(len(results_df))
kld_u_mean = np.array(list(results_df['kld_us'])).mean(axis=0)
kld_u_sem = np.array(list(results_df['kld_us'])).std(axis=0) / np.sqrt(len(results_df))

print(f'AoURP: {actual_kld_cs[-1]:.4f}, {actual_kld_us[-1]:.4f}')
print(
    f'Sim KLD(C||P): {kld_c_mean[-1]:.4f} [{(kld_c_mean[-1] - 1.96 * kld_c_sem[-1]):.4f}, {(kld_c_mean[-1] + 1.96 * kld_c_sem[-1]):.4f}]')
print(
    f'Sim KLD(C||U): {kld_u_mean[-1]:.4f} [{(kld_u_mean[-1] - 1.96 * kld_u_sem[-1]):.4f},{(kld_u_mean[-1] + 1.96 * kld_u_sem[-1]):.4f}]')

stacked_sim_cohorts = np.vstack(results_df['final_cohort'])
unwrapped_stacked_sim_cohorts = stacked_sim_cohorts.reshape([40, 3, 2, 5, 2])

# Age, Gender, Race, Ethnicity
# AGE_LABELS = ['20-44', '45-64', '65+']
# GENDER_LABELS = ['F', 'M']
# RACE_LABELS = ['ASIAN', 'BLACK', 'NH/PI', 'TWO+', 'WHITE']
# ETH_LABELS = ['H/L', 'NH/L']

ur_proportions = []
ur_by_race_eth_proportions = []

for i in range(40):
    total_cohort = unwrapped_stacked_sim_cohorts[i].sum()
    ur_by_race = unwrapped_stacked_sim_cohorts[i, :, :, 0:4, :].sum()  # All non-white
    ur_by_eth = unwrapped_stacked_sim_cohorts[i, :, :, 4, 0].sum()  # White & H/L
    ur_by_age = unwrapped_stacked_sim_cohorts[i, 2, :, 4, 1].sum()  # 65 + and White and non-H/L
    ur_proportions.append((ur_by_age + ur_by_race + ur_by_eth) / total_cohort)
    ur_by_race_eth_proportions.append((ur_by_race + ur_by_eth) / total_cohort)

aourp_final = quarterly_cumu_cohorts[-1]['DEMOGRAPHICS'].sum().reshape([3, 2, 5, 2])
aourp_total = aourp_final.sum()
aourp_ur_by_race = aourp_final[:, :, 0:4, :].sum()
aourp_ur_by_eth = aourp_final[:, :, 4, 0].sum()
aourp_ur_by_age = aourp_final[2, :, 4, 1].sum()
aourp_ur_prop = (aourp_ur_by_age + aourp_ur_by_race + aourp_ur_by_eth) / aourp_total
aourp_ur_re_prop = (aourp_ur_by_race + aourp_ur_by_eth) / aourp_total

ur_prop = np.array(ur_proportions)
ur_re_prop = np.array(ur_by_race_eth_proportions)

print('AoURP values:')
print(f'Underrepresented: {aourp_ur_prop:.4f}, just by race/ethnicity {aourp_ur_re_prop:.4f}')
print('Sim values:')
print(
    f'Underrepresented: {ur_prop.mean():.4f} ([{ur_prop.mean() - ur_prop.std() * 1.96 / np.sqrt(40):.4f}, {ur_prop.mean() + ur_prop.std() * 1.96 / np.sqrt(40):.4f}]), just by race/ethnicity {ur_re_prop.mean():.4f} ({ur_re_prop.mean() - ur_re_prop.std() * 1.96 / np.sqrt(40):.4f}, {ur_re_prop.mean() + ur_re_prop.std() * 1.96 / np.sqrt(40):.4f})')

### Show trends in representativeness and coverage over time for simulation compared to historic
# Calculate the means and SEM across experiments
kld_c_mean = np.array(list(results_df['kld_cs'])).mean(axis=0)
kld_c_sem = np.array(list(results_df['kld_cs'])).std(axis=0) / np.sqrt(len(results_df))

kld_u_mean = np.array(list(results_df['kld_us'])).mean(axis=0)
kld_u_sem = np.array(list(results_df['kld_us'])).std(axis=0) / np.sqrt(len(results_df))

### Figure 6
with matplotlib.rc_context({'font.family': 'sans', 'font.size': 8, 'figure.dpi': 300}):
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=[6, 2.3])

    for ax in axs:
        ax.set_xlim([0, 22])
        ax.set_xticks(ticks=[1, 5, 10, 15, 20], minor=False)
        ax.set_xticks(ticks=np.arange(21) + 1, minor=True)
        ax.set_xlabel('Iteration (Quarterly)')

    ax = axs[0]
    ax.plot(np.arange(21) + 1, kld_c_mean, color='tab:red', lw=1.6, label='Simulated KLD(C||P)')
    ax.plot(np.arange(21) + 1, actual_kld_cs, '--', lw=1.6, color='#A01030', label='Historic KLD(C||P)')
    ax.fill_between(np.arange(21) + 1, kld_c_mean - 1.96 * kld_c_sem, kld_c_mean + 1.96 * kld_c_sem, color='tab:red',
                    alpha=0.3, lw=0)
    ax.set_ylabel('Representativeness [KLD(C||P)]')
    ax.legend(loc=1)

    ax2 = axs[1]
    ax2.plot(np.arange(21) + 1, kld_u_mean, color='tab:blue', lw=1.6, label='Simulated KLD(C||U)')
    ax2.fill_between(np.arange(21) + 1, kld_u_mean - 1.96 * kld_u_sem, kld_u_mean + 1.96 * kld_u_sem, color='tab:blue',
                     alpha=0.3, lw=0)
    ax2.plot(np.arange(21) + 1, actual_kld_us, '--', lw=1.6, color='#103080', label='Historic KLD(C||U)')
    ax2.set_ylabel('Coverage [KLD(C||U)]')
    ax2.legend(loc=1)

    fig.tight_layout()
    plt.show()

# Plot final cohort demographics compared to historic results
# y-limits for the graph
ll = 1e-5
ul = 3e-1
census_distribution = (census_tot / census_tot.sum()).reshape([3, 2, 5, 2])
uniform_distribution = 1 / 60 * np.ones(60)
aourp_recruit_distribution = quarterly_cumu_cohorts[-1]['DEMOGRAPHICS'].sum(axis=0) / quarterly_cumu_cohorts[-1][
    'DEMOGRAPHICS'].sum().sum()
sim_distributions = np.vstack(results_df['final_cohort']) / 269862
sim_dist_means = sim_distributions.mean(axis=0)

# Print whether each group is within the proportions set by the Census values and uniform values
print(((sim_dist_means < census_distribution.reshape(-1)) & (sim_dist_means > uniform_distribution)) |
      ((sim_dist_means > census_distribution.reshape(-1)) & (sim_dist_means < uniform_distribution)))

site_specific_demos_1 = (np.array(quarterly_cumu_cohorts[-1]['DEMOGRAPHICS'].sum()) / np.array(
    quarterly_cumu_cohorts[-1]['DEMOGRAPHICS'].sum()).sum()).reshape([3, 2, 5, 2])
site_specific_demos_2 = (np.vstack(results_df['final_cohort']) / np.vstack(results_df['final_cohort']).sum(axis=1,
                                                                                                           keepdims=True)).reshape(
    [40, 3, 2, 5, 2])

race_labels = ['Asian', 'Black', 'NH/PI', 'T o M', 'White']
eth_labels = ['H/L', 'NH/L']
gender_labels = ['F', 'M']
age_labels = ['20-44', '45-64', '65+']

gen_demos_1 = []
gen_demos_2 = []
census_gen_demos = []
uniform_demos = []
labels = []

for race_idx in range(5):
    for eth_idx in range(2):
        for gender_idx in range(2):
            for age_idx in range(3):
                gen_demos_1.append(site_specific_demos_1[age_idx, gender_idx, race_idx, eth_idx])
                gen_demos_2.append(site_specific_demos_2[:, age_idx, gender_idx, race_idx, eth_idx])
                census_gen_demos.append(census_distribution[age_idx, gender_idx, race_idx, eth_idx])
                labels.append(
                    f'{race_labels[race_idx]} {eth_labels[eth_idx]} {gender_labels[gender_idx]} {age_labels[age_idx]}')
                uniform_demos.append(1)

gen_demos_2 = np.array(gen_demos_2).T

higher_line = np.maximum(np.array(census_gen_demos), np.array(uniform_demos) / 60)
lower_line = np.minimum(np.array(census_gen_demos), np.array(uniform_demos) / 60)

### Figure 7
with matplotlib.rc_context({'font.family': 'sans', 'font.size': 8, 'figure.dpi': 300}):
    fig, axs = plt.subplots(nrows=2, figsize=[6, 3.7], height_ratios=[3.5, 1])
    ax = axs[0]
    ax.step(np.arange(len(census_gen_demos)) + 0.5, np.array(uniform_demos) / 60, 'k--', lw=1, where='post',
            label='Uniform Group Proportion')
    ax.hlines(np.array(uniform_demos)[-1] / 60, len(census_gen_demos) - 0.5, len(census_gen_demos) + 0.5, 'k', lw=1,
              linestyles='--')

    ax.step(np.arange(len(census_gen_demos)) + 0.5, np.array(census_gen_demos), 'k', lw=1, where='post',
            label='Census Group Proportion')
    ax.hlines(np.array(census_gen_demos)[-1], len(census_gen_demos) - 0.5, len(census_gen_demos) + 0.5, 'k', lw=1)

    vplot = ax.violinplot(gen_demos_2, widths=1, showmeans=True)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = vplot[partname]
        vp.set_edgecolor('tab:green')
        vp.set_linewidth(0.8)
        vp.set_zorder(1000)

    for vp in vplot['bodies']:
        vp.set_facecolor('tab:green')
        vp.set_edgecolor('tab:green')
        vp.set_linewidth(0)
        vp.set_alpha(0.5)

    ax.plot(np.arange(len(census_gen_demos)) + 1,
            gen_demos_1,
            'o', markersize=2, color='k',
            label=f'AoURP Actual Cohort',
            zorder=100)

    ax.plot(1, 0.1, color='tab:green', alpha=0, label='Simulated Cohorts')

    ax.fill_between(np.arange(len(census_gen_demos)) + 0.5,
                    np.array(uniform_demos) / 60,
                    np.array(census_gen_demos),
                    color='tab:purple', alpha=0.15, step='post',
                    label='Target Zone')
    ax.fill_between([len(census_gen_demos) - 0.5, len(census_gen_demos) + 0.5],
                    [np.array(uniform_demos)[-1] / 60, np.array(uniform_demos)[-1] / 60],
                    [np.array(census_gen_demos)[-1], np.array(census_gen_demos)[-1]],
                    color='tab:purple', alpha=0.15, step='post')

    ax.fill_between(np.arange(len(census_gen_demos)) + 0.5,
                    lower_line,
                    ll,
                    color='tab:red', alpha=0.15, step='post')
    ax.fill_between([len(census_gen_demos) - 0.5, len(census_gen_demos) + 0.5],
                    lower_line[-1],
                    ll,
                    color='tab:red', alpha=0.15, step='post')
    ax.fill_between(np.arange(len(census_gen_demos)) + 0.5,
                    higher_line,
                    ul,
                    color='tab:blue', alpha=0.15, step='post')
    ax.fill_between([len(census_gen_demos) - 0.5, len(census_gen_demos) + 0.5],
                    higher_line[-1],
                    ul,
                    color='tab:blue', alpha=0.15, step='post')

    ax.set_yscale('log')
    ax.set_xlim([0.5, len(census_gen_demos) + 0.5])
    ax.set_ylim([ll, ul])
    ax.set_xticks(ticks=np.arange(len(census_gen_demos) + 1) + 0.5, labels=None, minor=True)
    ax.set_xticks(ticks=np.arange(len(census_gen_demos)) + 1, labels=np.array([label[12:].strip() for label in labels]),
                  minor=False, rotation=90, fontsize=6)
    ax.grid(axis='x', which='minor', color='0.8')
    ax.set_ylabel('Group Proportion (log scale)')
    leg = ax.legend(loc=4, fontsize=7)
    for i, lh in enumerate(leg.legendHandles):
        if i == 3:
            lh.set_alpha(1)
    ax.annotate('Underrepresented', xy=(0.5, ll), xytext=(1, 2), textcoords='offset points', fontsize=7)
    ax.annotate('Overrepresented', xy=(0.5, ul), xytext=(1, -8), textcoords='offset points', fontsize=7)

    ax = axs[1]
    color_dict = {'Asian': 0, 'Black': 1, 'NH/PI': 2, 'T o M': 3, 'White': 4,
                  'H/L': 5, 'NH/L': 6,
                  'F': 7, 'M': 8,
                  '20-44': 9, '45-64': 10, '65+': 11
                  }
    parsed_labels = np.flip(np.array([(color_dict[label[:5].strip()], color_dict[label[6:10].strip()],
                                       color_dict[label[10:12].strip()], color_dict[label[12:].strip()]) for label in
                                      labels]).T)
    ax.imshow(parsed_labels[1:, :], cmap='tab20', aspect=1.5)
    ax.axis('off')
    for i in range(10):
        ax.annotate('F', xy=(6 * i, 0), xytext=(4, -2), textcoords='offset points', fontsize=6)
        ax.annotate('M', xy=(6 * i + 3, 0), xytext=(3, -2), textcoords='offset points', fontsize=6)
    for i in range(5):
        ax.annotate('H/L', xy=(12 * i, 1), xytext=(10, -2), textcoords='offset points', fontsize=6)
        ax.annotate('NH/L', xy=(12 * i + 6, 1), xytext=(8, -2), textcoords='offset points', fontsize=6)
    ax.annotate('Asian', xy=(0, 2), xytext=(24, -2), textcoords='offset points', fontsize=6, color='white')
    ax.annotate('Black', xy=(12, 2), xytext=(24, -2), textcoords='offset points', fontsize=6)
    ax.annotate('NH/PI', xy=(24, 2), xytext=(24, -2), textcoords='offset points', fontsize=6)
    ax.annotate('Two or More', xy=(36, 2), xytext=(14, -2), textcoords='offset points', fontsize=6, color='white')
    ax.annotate('White', xy=(48, 2), xytext=(24, -2), textcoords='offset points', fontsize=6, color='white')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    plt.show()

# Resource distribution to sites
ehr_site_recruits = pd.concat([pd.DataFrame.from_dict(policy).sum(axis=0) for policy in results_df['policy']], axis=1)
actual_demographics = pd.merge(ehr_site_recruits, quarterly_cumu_cohorts[-1][['SITE', 'DEMOGRAPHICS']], left_index=True, right_on='SITE')[['SITE', 'DEMOGRAPHICS']]
actual_demographics['SITE_SUM'] = actual_demographics['DEMOGRAPHICS'].apply(lambda row: row.sum())
sorted_values = np.flip(np.argsort(ehr_site_recruits.median(axis=1)))

### Figure 8
with matplotlib.rc_context({'font.family': 'sans', 'font.size': 9, 'figure.dpi':300}):
    fig, ax = plt.subplots(figsize=[8, 4])
    bplot = ax.violinplot(np.array(ehr_site_recruits.T)[:, sorted_values],
                        widths=0.8,
                        showmedians=True,)
    ax.set_xticks(ticks=np.arange(50)+1, labels=ehr_site_recruits.index.values[sorted_values], minor=False)
    ax.set_xticks(ticks=np.arange(51)+0.5, labels=None, minor=True)
    ax.tick_params(axis='x', labelrotation=90)
    ax.plot(np.arange(50)+1, np.array(actual_demographics['SITE_SUM'])[sorted_values], 'ro', markersize=4, label='AoURP Recruitments')
    ax.plot(1, 1000, color='tab:blue', alpha=0, label='Simulated Recruitments')
    ax.set_yscale('log')
    ax.set_xlim(0.5, 50.5)
    ax.set_xlabel('EHR Site src_id')
    ax.set_ylabel('Participants Recruited (log scale)')
    leg = ax.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    fig.tight_layout()
    ax.grid(axis='x', which='minor')
    plt.show()