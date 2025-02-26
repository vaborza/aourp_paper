# Collected preprocessing steps

## Packages and Default Functions
import pandas as pd
import numpy as np
import os
rng = np.random.default_rng(seed=8080)

registered_tier = False # Change this flag to True if analyses are performed in Registered Tier

# SQL code to query AoURP database and pipe into python, this may need modification to user's environment
demographics_table = pd.read_gbq(
    f'''
    SELECT person_id,
        CASE
            WHEN (2022 - person_t.year_of_birth) < 20 THEN '0-19'
            WHEN ((2022 - person_t.year_of_birth) >= 20) AND ((2022 - person_t.year_of_birth) < 45) THEN '20-44'
            WHEN ((2022 - person_t.year_of_birth) >= 45) AND ((2022 - person_t.year_of_birth) < 65) THEN '45-64'
            ELSE '65+'
            END AS AGE,
        CASE
            WHEN (person_t.gender_source_value = 'GenderIdentity_Man') THEN 'MAN'
            WHEN (person_t.gender_source_value = 'GenderIdentity_Woman') THEN 'WOMAN'
            ELSE 'OTHER'
            END AS GENDER,
        CASE
            WHEN (person_t.race_source_value = 'WhatRaceEthnicity_White') THEN 'WHITE'
            WHEN (person_t.race_source_value = 'WhatRaceEthnicity_Black') THEN 'BLACK'
            WHEN (person_t.race_source_value = 'WhatRaceEthnicity_Asian') THEN 'ASIAN'
            WHEN (person_t.race_source_value = 'WhatRaceEthnicity_GeneralizedMultPopulations') THEN 'MIXED'
            WHEN (person_t.race_source_value = 'WhatRaceEthnicity_NHPI') THEN 'NHPI'
            ELSE 'OTHER'
            END AS RACE,
        CASE
            WHEN (person_t.ethnicity_source_value = 'Not Hispanic') THEN 'NH'
            WHEN (person_t.ethnicity_source_value = 'Hispanic') THEN 'HL'
            ELSE 'OTHER'
            END AS ETH
        FROM {os.environ["WORKSPACE_CDR"]}.person AS person_t
    ''',
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook")

ehr_site_table = pd.read_gbq(
    f'''
    SELECT obs.person_id, RIGHT(obs_ext.src_id, 3) AS SITE, MIN(obs.observation_datetime) AS TIME
    FROM {os.environ["WORKSPACE_CDR"]}.observation AS obs
    INNER JOIN {os.environ["WORKSPACE_CDR"]}.observation_ext AS obs_ext
    ON obs.observation_id = obs_ext.observation_id
    WHERE src_id LIKE "EHR%"
    GROUP BY obs.person_id, obs_ext.src_id
    ''',
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook")

if registered_tier:
    pass
else:
    zip3_table = pd.read_gbq(
        f'''
        SELECT person_id, LEFT(value_as_string, 3) AS ZIP3
        FROM {os.environ["WORKSPACE_CDR"]}.observation
        WHERE observation_source_concept_id = 1585250
            ''',
        dialect="standard",
        use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
        progress_bar_type="tqdm_notebook")

survey_table = pd.read_gbq(
    f'''
    SELECT person_id, MIN(survey_datetime) AS SURVEY_TIME
            FROM {os.environ["WORKSPACE_CDR"]}.ds_survey
            WHERE survey = 'The Basics'
            GROUP BY person_id
    ''',
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook")

# Impute race values for H/L ethnicity, per description in methods section
pop_uncleaned = demographics_table[demographics_table['AGE'] != '0-19'].copy(deep=True)
pop_imputed = pop_uncleaned.copy(deep=True)
lut = pop_uncleaned[(pop_uncleaned['ETH'] == 'HL') &
                    (pop_uncleaned['RACE'] != 'OTHER')].groupby(['AGE', 'GENDER', 'RACE']).size()
for i, row in pop_imputed[(pop_imputed['ETH'] == 'HL') & (pop_imputed['RACE'] == 'OTHER')].iterrows():
    imputed_race = rng.choice(lut[row['AGE'], row['GENDER']].index,
                              p=lut[row['AGE'], row['GENDER']]/lut[row['AGE'], row['GENDER']].sum())
    pop_imputed.at[i, 'RACE'] = imputed_race
pop_cleaned = pop_imputed[(pop_imputed['AGE'] != '0-19') &
                          (pop_imputed['GENDER'] != 'OTHER') &
                          (pop_imputed['RACE'] != 'OTHER') &
                          (pop_imputed['ETH'] != 'OTHER')]

if registered_tier:
    pass
else:
    # Aggregate ZIP3 to regions with >100 individuals. Census data (available in pickle) are aggregated similarly.
    aou_census_geography = zip3_table
    aou_census_geography['MOD_ZIP3'] = aou_census_geography['ZIP3']

    # AK
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '995') | (aou_census_geography['ZIP3'] == '996') | (aou_census_geography['ZIP3'] == '997') | (aou_census_geography['ZIP3'] == '998') | (aou_census_geography['ZIP3'] == '999')] = '99_56789'

    # WA
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '988') | (aou_census_geography['ZIP3'] == '989') | (aou_census_geography['ZIP3'] == '991') | (aou_census_geography['ZIP3'] == '993') | (aou_census_geography['ZIP3'] == '994')] = '98_89_99_134'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '990') | (aou_census_geography['ZIP3'] == '992')] = '99_02'

    # OR
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '970') | (aou_census_geography['ZIP3'] == '971') | (aou_census_geography['ZIP3'] == '972')] = '97_012'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '974') | (aou_census_geography['ZIP3'] == '975') | (aou_census_geography['ZIP3'] == '976') | (aou_census_geography['ZIP3'] == '977') | (aou_census_geography['ZIP3'] == '978') | (aou_census_geography['ZIP3'] == '979')] = '97_456789'

    # HI
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '967') | (aou_census_geography['ZIP3'] == '968')] = '96_78'

    # CA
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '903') | (aou_census_geography['ZIP3'] == '904') | (aou_census_geography['ZIP3'] == '905')] = '90_345'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '923') | (aou_census_geography['ZIP3'] == '924')] = '92_34'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '930') | (aou_census_geography['ZIP3'] == '931')] = '93_01'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '936') | (aou_census_geography['ZIP3'] == '937')] = '93_67'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '940') | (aou_census_geography['ZIP3'] == '944')] = '94_04'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '945') | (aou_census_geography['ZIP3'] == '948')] = '94_58'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '955') | (aou_census_geography['ZIP3'] == '960') | (aou_census_geography['ZIP3'] == '961')] = '95_5_96_01'

    # NV
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '894') | (aou_census_geography['ZIP3'] == '895') | (aou_census_geography['ZIP3'] == '897') | (aou_census_geography['ZIP3'] == '898')] = '89_4578'

    # NM
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '870') | (aou_census_geography['ZIP3'] == '871') | (aou_census_geography['ZIP3'] == '873') | (aou_census_geography['ZIP3'] == '874')] = '87_0134'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '875') | (aou_census_geography['ZIP3'] == '877') | (aou_census_geography['ZIP3'] == '873') | (aou_census_geography['ZIP3'] == '874')] = '87_57'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '880') | (aou_census_geography['ZIP3'] == '881') | (aou_census_geography['ZIP3'] == '882') | (aou_census_geography['ZIP3'] == '883')] = '88_0123'

    # UT
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '840') | (aou_census_geography['ZIP3'] == '841') | (aou_census_geography['ZIP3'] == '843') | (aou_census_geography['ZIP3'] == '844')] = '84_0134'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '845') | (aou_census_geography['ZIP3'] == '846') | (aou_census_geography['ZIP3'] == '847') | (aou_census_geography['ZIP3'] == '860') | (aou_census_geography['ZIP3'] == '864') | (aou_census_geography['ZIP3'] == '865')] = '84_567_86_045'

    # ID
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '832') | (aou_census_geography['ZIP3'] == '833') | (aou_census_geography['ZIP3'] == '835') | (aou_census_geography['ZIP3'] == '838')] = '83_2358'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '836') | (aou_census_geography['ZIP3'] == '837')] = '83_67'

    # WY
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '820') | (aou_census_geography['ZIP3'] == '822') | (aou_census_geography['ZIP3'] == '824') | (aou_census_geography['ZIP3'] == '825') | (aou_census_geography['ZIP3'] == '826') | (aou_census_geography['ZIP3'] == '827') | (aou_census_geography['ZIP3'] == '828') | (aou_census_geography['ZIP3'] == '829') | (aou_census_geography['ZIP3'] == '830')] = '82_02456789_83_0'

    # CO
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '803') | (aou_census_geography['ZIP3'] == '804') | (aou_census_geography['ZIP3'] == '805')] = '80_345'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '806') | (aou_census_geography['ZIP3'] == '807')] = '80_67'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '808') | (aou_census_geography['ZIP3'] == '809') | (aou_census_geography['ZIP3'] == '810')] = '80_89_81_0'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '811') | (aou_census_geography['ZIP3'] == '812') | (aou_census_geography['ZIP3'] == '813') | (aou_census_geography['ZIP3'] == '814') | (aou_census_geography['ZIP3'] == '815') | (aou_census_geography['ZIP3'] == '816')] = '81_123456'

    # TX
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '752') | (aou_census_geography['ZIP3'] == '753')] = '75_23'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '754') | (aou_census_geography['ZIP3'] == '755')] = '75_45'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '756') | (aou_census_geography['ZIP3'] == '757') | (aou_census_geography['ZIP3'] == '758') | (aou_census_geography['ZIP3'] == '759')] = '75_6789'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '760') | (aou_census_geography['ZIP3'] == '761') | (aou_census_geography['ZIP3'] == '762') | (aou_census_geography['ZIP3'] == '763') | (aou_census_geography['ZIP3'] == '764')] = '76_01234'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '768') | (aou_census_geography['ZIP3'] == '769') | (aou_census_geography['ZIP3'] == '790') | (aou_census_geography['ZIP3'] == '791') | (aou_census_geography['ZIP3'] == '792') | (aou_census_geography['ZIP3'] == '793') | (aou_census_geography['ZIP3'] == '794') | (aou_census_geography['ZIP3'] == '795') | (aou_census_geography['ZIP3'] == '796') | (aou_census_geography['ZIP3'] == '797')] = '76_89_79_01234567'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '775') | (aou_census_geography['ZIP3'] == '776') | (aou_census_geography['ZIP3'] == '777')] = '77_567'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '770') | (aou_census_geography['ZIP3'] == '772')] = '77_02'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '779') | (aou_census_geography['ZIP3'] == '780') | (aou_census_geography['ZIP3'] == '781') | (aou_census_geography['ZIP3'] == '782') | (aou_census_geography['ZIP3'] == '788')] = '77_9_78_0128'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '783') | (aou_census_geography['ZIP3'] == '784') | (aou_census_geography['ZIP3'] == '785')] = '78_345'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '786') | (aou_census_geography['ZIP3'] == '787') | (aou_census_geography['ZIP3'] == '789')] = '78_679'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '798') | (aou_census_geography['ZIP3'] == '799')] = '79_89'

    # OK
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '730') | (aou_census_geography['ZIP3'] == '731') | (aou_census_geography['ZIP3'] == '734') | (aou_census_geography['ZIP3'] == '735') | (aou_census_geography['ZIP3'] == '736') | (aou_census_geography['ZIP3'] == '737') | (aou_census_geography['ZIP3'] == '738') | (aou_census_geography['ZIP3'] == '739') | (aou_census_geography['ZIP3'] == '745') | (aou_census_geography['ZIP3'] == '747') | (aou_census_geography['ZIP3'] == '748')] = '73_01456789_74_578'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '740') | (aou_census_geography['ZIP3'] == '741') | (aou_census_geography['ZIP3'] == '743') | (aou_census_geography['ZIP3'] == '744') | (aou_census_geography['ZIP3'] == '746')] = '74_01346'

    # AR
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '716') | (aou_census_geography['ZIP3'] == '717') | (aou_census_geography['ZIP3'] == '718') | (aou_census_geography['ZIP3'] == '719') | (aou_census_geography['ZIP3'] == '727') | (aou_census_geography['ZIP3'] == '728') | (aou_census_geography['ZIP3'] == '729')] = '71_6789_72_789'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '720') | (aou_census_geography['ZIP3'] == '721') | (aou_census_geography['ZIP3'] == '722')] = '72_012'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '723') | (aou_census_geography['ZIP3'] == '724') | (aou_census_geography['ZIP3'] == '725') | (aou_census_geography['ZIP3'] == '726')] = '72_3456'

    # LA
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '703') | (aou_census_geography['ZIP3'] == '705') | (aou_census_geography['ZIP3'] == '706')] = '70_356'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '707') | (aou_census_geography['ZIP3'] == '708')] = '70_78'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '710') | (aou_census_geography['ZIP3'] == '711') | (aou_census_geography['ZIP3'] == '712') | (aou_census_geography['ZIP3'] == '713') | (aou_census_geography['ZIP3'] == '714')] = '71_1234'

    # NE
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '680') | (aou_census_geography['ZIP3'] == '681')] = '68_01'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '683') | (aou_census_geography['ZIP3'] == '684') | (aou_census_geography['ZIP3'] == '685') | (aou_census_geography['ZIP3'] == '686') | (aou_census_geography['ZIP3'] == '687') | (aou_census_geography['ZIP3'] == '688') | (aou_census_geography['ZIP3'] == '689') | (aou_census_geography['ZIP3'] == '690') | (aou_census_geography['ZIP3'] == '691') | (aou_census_geography['ZIP3'] == '693')] = '68_3456789_69_13'

    # KS
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '664') | (aou_census_geography['ZIP3'] == '665') | (aou_census_geography['ZIP3'] == '666') | (aou_census_geography['ZIP3'] == '667') | (aou_census_geography['ZIP3'] == '668') | (aou_census_geography['ZIP3'] == '669') | (aou_census_geography['ZIP3'] == '674') | (aou_census_geography['ZIP3'] == '676') | (aou_census_geography['ZIP3'] == '677')] = '66_456789_67_467'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '670') | (aou_census_geography['ZIP3'] == '671') | (aou_census_geography['ZIP3'] == '672') | (aou_census_geography['ZIP3'] == '673') | (aou_census_geography['ZIP3'] == '675') | (aou_census_geography['ZIP3'] == '678') | (aou_census_geography['ZIP3'] == '679')] = '67_0123589'

    # MO
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '630') | (aou_census_geography['ZIP3'] == '631') | (aou_census_geography['ZIP3'] == '633')] = '63_013'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '634') | (aou_census_geography['ZIP3'] == '635') | (aou_census_geography['ZIP3'] == '644') | (aou_census_geography['ZIP3'] == '645') | (aou_census_geography['ZIP3'] == '646') | (aou_census_geography['ZIP3'] == '650') | (aou_census_geography['ZIP3'] == '651') | (aou_census_geography['ZIP3'] == '652') | (aou_census_geography['ZIP3'] == '653')] = '63_45_64_456_65_0123'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '636') | (aou_census_geography['ZIP3'] == '637') | (aou_census_geography['ZIP3'] == '638') | (aou_census_geography['ZIP3'] == '647') | (aou_census_geography['ZIP3'] == '648') | (aou_census_geography['ZIP3'] == '639') | (aou_census_geography['ZIP3'] == '654') | (aou_census_geography['ZIP3'] == '655') | (aou_census_geography['ZIP3'] == '656') | (aou_census_geography['ZIP3'] == '657') | (aou_census_geography['ZIP3'] == '658')] = '63_6789_64_78_65_45678'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '640') | (aou_census_geography['ZIP3'] == '641') | (aou_census_geography['ZIP3'] == '660') | (aou_census_geography['ZIP3'] == '661') | (aou_census_geography['ZIP3'] == '662')] = '64_01_66_012' # Intentional combining of Kansas City, MO and Kansas City, KS

    # IL
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '609') | (aou_census_geography['ZIP3'] == '617')] = '60_9_61_7'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '612') | (aou_census_geography['ZIP3'] == '613') | (aou_census_geography['ZIP3'] == '614')] = '61_234'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '615') | (aou_census_geography['ZIP3'] == '616')] = '61_56'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '618') | (aou_census_geography['ZIP3'] == '619') | (aou_census_geography['ZIP3'] == '623') | (aou_census_geography['ZIP3'] == '624') | (aou_census_geography['ZIP3'] == '625') | (aou_census_geography['ZIP3'] == '626') | (aou_census_geography['ZIP3'] == '627')] = '61_89_62_34567'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '620') | (aou_census_geography['ZIP3'] == '622') | (aou_census_geography['ZIP3'] == '628') | (aou_census_geography['ZIP3'] == '629')] = '62_0289'

    # MT
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '590') | (aou_census_geography['ZIP3'] == '591') | (aou_census_geography['ZIP3'] == '592') | (aou_census_geography['ZIP3'] == '593') | (aou_census_geography['ZIP3'] == '594') | (aou_census_geography['ZIP3'] == '595') | (aou_census_geography['ZIP3'] == '596') | (aou_census_geography['ZIP3'] == '597')] = '59_01234567'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '598') | (aou_census_geography['ZIP3'] == '599')] = '59_89'

    # ND
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '580') | (aou_census_geography['ZIP3'] == '581')] = '58_01'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '582') | (aou_census_geography['ZIP3'] == '583') | (aou_census_geography['ZIP3'] == '584') | (aou_census_geography['ZIP3'] == '585') | (aou_census_geography['ZIP3'] == '586') | (aou_census_geography['ZIP3'] == '587') | (aou_census_geography['ZIP3'] == '588')] = '58_2345678'

    # SD
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '570') | (aou_census_geography['ZIP3'] == '571') | (aou_census_geography['ZIP3'] == '572') | (aou_census_geography['ZIP3'] == '573') | (aou_census_geography['ZIP3'] == '574') | (aou_census_geography['ZIP3'] == '575') | (aou_census_geography['ZIP3'] == '576') | (aou_census_geography['ZIP3'] == '577')] = '57_01234567'

    # MN
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '553') | (aou_census_geography['ZIP3'] == '560') | (aou_census_geography['ZIP3'] == '561') | (aou_census_geography['ZIP3'] == '562')] = '55_3_56_012'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '563') | (aou_census_geography['ZIP3'] == '564')] = '56_34'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '565') | (aou_census_geography['ZIP3'] == '566') | (aou_census_geography['ZIP3'] == '567')] = '56_567'

    # WI
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '535') | (aou_census_geography['ZIP3'] == '538')] = '53_58'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '541') | (aou_census_geography['ZIP3'] == '542') | (aou_census_geography['ZIP3'] == '543')] = '54_123'

    # IA
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '500') | (aou_census_geography['ZIP3'] == '501') | (aou_census_geography['ZIP3'] == '502') | (aou_census_geography['ZIP3'] == '503') | (aou_census_geography['ZIP3'] == '508')] = '50_01238'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '504') | (aou_census_geography['ZIP3'] == '505') | (aou_census_geography['ZIP3'] == '506') | (aou_census_geography['ZIP3'] == '507')  | (aou_census_geography['ZIP3'] == '510') | (aou_census_geography['ZIP3'] == '511') | (aou_census_geography['ZIP3'] == '512') | (aou_census_geography['ZIP3'] == '513') | (aou_census_geography['ZIP3'] == '514') | (aou_census_geography['ZIP3'] == '515')] = '50_4567_51_012345'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '520') | (aou_census_geography['ZIP3'] == '526') | (aou_census_geography['ZIP3'] == '527') | (aou_census_geography['ZIP3'] == '528')] = '52_0678'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '521') | (aou_census_geography['ZIP3'] == '522') | (aou_census_geography['ZIP3'] == '523') | (aou_census_geography['ZIP3'] == '524') | (aou_census_geography['ZIP3'] == '525')] = '52_12345'

    # MI
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '484') | (aou_census_geography['ZIP3'] == '485')] = '48_45'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '486') | (aou_census_geography['ZIP3'] == '487')] = '48_67'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '488') | (aou_census_geography['ZIP3'] == '489')] = '48_89'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '490') | (aou_census_geography['ZIP3'] == '491')] = '49_01'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '497') | (aou_census_geography['ZIP3'] == '498') | (aou_census_geography['ZIP3'] == '499')] = '49_789'

    # IN
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '460') | (aou_census_geography['ZIP3'] == '479')] = '46_0_47_9'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '461') | (aou_census_geography['ZIP3'] == '478')] = '46_1_47_8'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '463') | (aou_census_geography['ZIP3'] == '464')] = '46_34'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '465') | (aou_census_geography['ZIP3'] == '466')] = '46_56'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '467') | (aou_census_geography['ZIP3'] == '468') | (aou_census_geography['ZIP3'] == '469') | (aou_census_geography['ZIP3'] == '473')] = '46_789_47_3'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '470') | (aou_census_geography['ZIP3'] == '471') | (aou_census_geography['ZIP3'] == '472') | (aou_census_geography['ZIP3'] == '474') | (aou_census_geography['ZIP3'] == '475') | (aou_census_geography['ZIP3'] == '476') | (aou_census_geography['ZIP3'] == '477')] = '47_0124567'

    # OH
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '430') | (aou_census_geography['ZIP3'] == '431') | (aou_census_geography['ZIP3'] == '432') | (aou_census_geography['ZIP3'] == '433')] = '43_0123'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '434') | (aou_census_geography['ZIP3'] == '435') | (aou_census_geography['ZIP3'] == '436') | (aou_census_geography['ZIP3'] == '440') |  (aou_census_geography['ZIP3'] == '448') | (aou_census_geography['ZIP3'] == '449')] = '43_456_44_089'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '437') | (aou_census_geography['ZIP3'] == '438') | (aou_census_geography['ZIP3'] == '439') | (aou_census_geography['ZIP3'] == '446') |  (aou_census_geography['ZIP3'] == '447') | (aou_census_geography['ZIP3'] == '457')] = '43_789_44_67_45_7'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '442') | (aou_census_geography['ZIP3'] == '443') | (aou_census_geography['ZIP3'] == '444') | (aou_census_geography['ZIP3'] == '445')] = '44_2345'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '453') | (aou_census_geography['ZIP3'] == '454') | (aou_census_geography['ZIP3'] == '455') | (aou_census_geography['ZIP3'] == '458')] = '45_3458'

    # KY
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '400') | (aou_census_geography['ZIP3'] == '401') | (aou_census_geography['ZIP3'] == '402')] = '40_012'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '403') | (aou_census_geography['ZIP3'] == '404') | (aou_census_geography['ZIP3'] == '405') | (aou_census_geography['ZIP3'] == '406')] = '40_3456'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '407') | (aou_census_geography['ZIP3'] == '408') | (aou_census_geography['ZIP3'] == '409') | (aou_census_geography['ZIP3'] == '412') | (aou_census_geography['ZIP3'] == '413') | (aou_census_geography['ZIP3'] == '415') | (aou_census_geography['ZIP3'] == '416') | (aou_census_geography['ZIP3'] == '418')| (aou_census_geography['ZIP3'] == '420') | (aou_census_geography['ZIP3'] == '421') | (aou_census_geography['ZIP3'] == '422') | (aou_census_geography['ZIP3'] == '423') | (aou_census_geography['ZIP3'] == '424') | (aou_census_geography['ZIP3'] == '425') | (aou_census_geography['ZIP3'] == '426') | (aou_census_geography['ZIP3'] == '427')] = '40_789_41_23568_42_01234567'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '410') | (aou_census_geography['ZIP3'] == '411') | (aou_census_geography['ZIP3'] == '450') | (aou_census_geography['ZIP3'] == '451') | (aou_census_geography['ZIP3'] == '452') | (aou_census_geography['ZIP3'] == '456')] = '41_01_45_0126'

    # MS
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '386') | (aou_census_geography['ZIP3'] == '387') | (aou_census_geography['ZIP3'] == '388') | (aou_census_geography['ZIP3'] == '389')] = '38_6789'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '393') | (aou_census_geography['ZIP3'] == '397')] = '39_37'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '394') | (aou_census_geography['ZIP3'] == '395') | (aou_census_geography['ZIP3'] == '396')] = '39_456'

    # TN
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '370') | (aou_census_geography['ZIP3'] == '371') | (aou_census_geography['ZIP3'] == '384') | (aou_census_geography['ZIP3'] == '385')] = '37_01_38_45'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '373') | (aou_census_geography['ZIP3'] == '374')] = '37_34'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '376') | (aou_census_geography['ZIP3'] == '377')] = '37_67'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '380') | (aou_census_geography['ZIP3'] == '381') | (aou_census_geography['ZIP3'] == '382') | (aou_census_geography['ZIP3'] == '383')] = '38_0123'

    #AL
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '363') | (aou_census_geography['ZIP3'] == '364')] = '36_34'

    #FL
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '324') | (aou_census_geography['ZIP3'] == '325')] = '32_45'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '338') | (aou_census_geography['ZIP3'] == '349')] = '33_8_34_9'

    # GA
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '304') | (aou_census_geography['ZIP3'] == '308') | (aou_census_geography['ZIP3'] == '309') | (aou_census_geography['ZIP3'] == '313') | (aou_census_geography['ZIP3'] == '314')] = '30_489_31_34'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '301') | (aou_census_geography['ZIP3'] == '307')] = '30_17'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '310') | (aou_census_geography['ZIP3'] == '312')] = '31_02'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '315') | (aou_census_geography['ZIP3'] == '316') | (aou_census_geography['ZIP3'] == '317') | (aou_census_geography['ZIP3'] == '318') | (aou_census_geography['ZIP3'] == '319') | (aou_census_geography['ZIP3'] == '398')] = '31_56789_39_8'

    # SC
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '290') | (aou_census_geography['ZIP3'] == '298') | (aou_census_geography['ZIP3'] == '299')] = '29_089'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '293') | (aou_census_geography['ZIP3'] == '296') | (aou_census_geography['ZIP3'] == '297')] = '29_367'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '294') | (aou_census_geography['ZIP3'] == '295')] = '29_45'

    # NC
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '270') | (aou_census_geography['ZIP3'] == '271') | (aou_census_geography['ZIP3'] == '272') | (aou_census_geography['ZIP3'] == '273') | (aou_census_geography['ZIP3'] == '274')] = '27_01234'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '275') | (aou_census_geography['ZIP3'] == '278') | (aou_census_geography['ZIP3'] == '279') | (aou_census_geography['ZIP3'] == '285')] = '27_589_28_5'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '280') | (aou_census_geography['ZIP3'] == '281') | (aou_census_geography['ZIP3'] == '282')] = '28_012'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '283') | (aou_census_geography['ZIP3'] == '284')] = '28_34'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '286') | (aou_census_geography['ZIP3'] == '287') | (aou_census_geography['ZIP3'] == '288') | (aou_census_geography['ZIP3'] == '289')] = '28_6789'

    # WV
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '250') | (aou_census_geography['ZIP3'] == '251') | (aou_census_geography['ZIP3'] == '252') | (aou_census_geography['ZIP3'] == '253') | (aou_census_geography['ZIP3'] == '254') | (aou_census_geography['ZIP3'] == '255') | (aou_census_geography['ZIP3'] == '256') | (aou_census_geography['ZIP3'] == '257') | (aou_census_geography['ZIP3'] == '258') | (aou_census_geography['ZIP3'] == '259') | (aou_census_geography['ZIP3'] == '260') | (aou_census_geography['ZIP3'] == '261') | (aou_census_geography['ZIP3'] == '262') | (aou_census_geography['ZIP3'] == '263') | (aou_census_geography['ZIP3'] == '264') | (aou_census_geography['ZIP3'] == '265') | (aou_census_geography['ZIP3'] == '266') | (aou_census_geography['ZIP3'] == '267') | (aou_census_geography['ZIP3'] == '268')] = '25_0123456789_26_012345678'

    # VA
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '221') | (aou_census_geography['ZIP3'] == '224') | (aou_census_geography['ZIP3'] == '225')] = '22_145'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '226') | (aou_census_geography['ZIP3'] == '227') | (aou_census_geography['ZIP3'] == '228') | (aou_census_geography['ZIP3'] == '229')] = '22_6789'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '230') | (aou_census_geography['ZIP3'] == '231') | (aou_census_geography['ZIP3'] == '232') | (aou_census_geography['ZIP3'] == '238') | (aou_census_geography['ZIP3'] == '239')] = '23_01289'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '233') | (aou_census_geography['ZIP3'] == '234') | (aou_census_geography['ZIP3'] == '235') | (aou_census_geography['ZIP3'] == '236') | (aou_census_geography['ZIP3'] == '237')] = '23_34567'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '240') | (aou_census_geography['ZIP3'] == '241') | (aou_census_geography['ZIP3'] == '242') | (aou_census_geography['ZIP3'] == '243') | (aou_census_geography['ZIP3'] == '244') | (aou_census_geography['ZIP3'] == '245') | (aou_census_geography['ZIP3'] == '246') | (aou_census_geography['ZIP3'] == '247') | (aou_census_geography['ZIP3'] == '248') | (aou_census_geography['ZIP3'] == '249')] = '24_0123456789'

    # MD
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '206') | (aou_census_geography['ZIP3'] == '207')] = '20_67'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '210') | (aou_census_geography['ZIP3'] == '211') | (aou_census_geography['ZIP3'] == '214')] = '21_014'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '215') | (aou_census_geography['ZIP3'] == '217')] = '21_57'

    # DC
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '200') | (aou_census_geography['ZIP3'] == '203')] = '20_03'

    # DE
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '197') | (aou_census_geography['ZIP3'] == '198') | (aou_census_geography['ZIP3'] == '199') | (aou_census_geography['ZIP3'] == '216') | (aou_census_geography['ZIP3'] == '218') | (aou_census_geography['ZIP3'] == '219')] = '19_789_21_689'

    # PA
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '158') | (aou_census_geography['ZIP3'] == '167') | (aou_census_geography['ZIP3'] == '169')] = '15_8_16_79'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '166') | (aou_census_geography['ZIP3'] == '168')] = '16_68'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '170') | (aou_census_geography['ZIP3'] == '171') | (aou_census_geography['ZIP3'] == '172')] = '17_012'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '173') | (aou_census_geography['ZIP3'] == '174') | (aou_census_geography['ZIP3'] == '175') | (aou_census_geography['ZIP3'] == '176')] = '17_3456'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '177') | (aou_census_geography['ZIP3'] == '178')] = '17_78'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '179') | (aou_census_geography['ZIP3'] == '180') | (aou_census_geography['ZIP3'] == '181') | (aou_census_geography['ZIP3'] == '182') | (aou_census_geography['ZIP3'] == '195') | (aou_census_geography['ZIP3'] == '196')] = '17_9_18_012_19_56'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '183') | (aou_census_geography['ZIP3'] == '184') | (aou_census_geography['ZIP3'] == '185') | (aou_census_geography['ZIP3'] == '186') | (aou_census_geography['ZIP3'] == '187') | (aou_census_geography['ZIP3'] == '188')] = '18_345678'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '189') | (aou_census_geography['ZIP3'] == '190')] = '18_9_19_0'

    # MA
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '010') | (aou_census_geography['ZIP3'] == '011') | (aou_census_geography['ZIP3'] == '013')] = '01_013'

    # NH
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '030') | (aou_census_geography['ZIP3'] == '031') | (aou_census_geography['ZIP3'] == '033') | (aou_census_geography['ZIP3'] == '034')] = '03_0134'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '032') | (aou_census_geography['ZIP3'] == '037')] = '03_27'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '035') | (aou_census_geography['ZIP3'] == '038') | (aou_census_geography['ZIP3'] == '039')] = '03_589'

    # ME
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '040') | (aou_census_geography['ZIP3'] == '041') | (aou_census_geography['ZIP3'] == '042')] = '04_012'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '043') | (aou_census_geography['ZIP3'] == '045') | (aou_census_geography['ZIP3'] == '048')] = '04_358'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '044') | (aou_census_geography['ZIP3'] == '046') | (aou_census_geography['ZIP3'] == '047') | (aou_census_geography['ZIP3'] == '049')] = '04_4679'

    # VT
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '050') | (aou_census_geography['ZIP3'] == '051') | (aou_census_geography['ZIP3'] == '052') | (aou_census_geography['ZIP3'] == '053') | (aou_census_geography['ZIP3'] == '057')] = '05_01237'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '054') | (aou_census_geography['ZIP3'] == '056') | (aou_census_geography['ZIP3'] == '058') | (aou_census_geography['ZIP3'] == '129')] = '05_468_12_9'

    # CT
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '060') | (aou_census_geography['ZIP3'] == '061')] = '06_01'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '064') | (aou_census_geography['ZIP3'] == '065') | (aou_census_geography['ZIP3'] == '066')] = '06_456'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '068') | (aou_census_geography['ZIP3'] == '069')] = '06_89'

    # NJ
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '070') | (aou_census_geography['ZIP3'] == '071') | (aou_census_geography['ZIP3'] == '072') | (aou_census_geography['ZIP3'] == '073')] = '07_0123'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '074') | (aou_census_geography['ZIP3'] == '075')] = '07_45'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '078') | (aou_census_geography['ZIP3'] == '079')] = '07_89'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '077') | (aou_census_geography['ZIP3'] == '087')] = '07_7_08_7'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '080') | (aou_census_geography['ZIP3'] == '081') | (aou_census_geography['ZIP3'] == '082') | (aou_census_geography['ZIP3'] == '083')] = '08_0123'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '085') | (aou_census_geography['ZIP3'] == '086')] = '08_56'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '088') | (aou_census_geography['ZIP3'] == '089')] = '08_89'


    # NY
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '105') | (aou_census_geography['ZIP3'] == '106')] = '10_56'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '107') | (aou_census_geography['ZIP3'] == '108')] = '10_78'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '110') | (aou_census_geography['ZIP3'] == '113')] = '11_03'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '117') | (aou_census_geography['ZIP3'] == '118') | (aou_census_geography['ZIP3'] == '119')] = '11_789'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '120') | (aou_census_geography['ZIP3'] == '121') | (aou_census_geography['ZIP3'] == '122') | (aou_census_geography['ZIP3'] == '123')] = '12_0123'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '124') | (aou_census_geography['ZIP3'] == '125') | (aou_census_geography['ZIP3'] == '127')] = '12_457'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '128') | (aou_census_geography['ZIP3'] == '133') | (aou_census_geography['ZIP3'] == '134') | (aou_census_geography['ZIP3'] == '135') | (aou_census_geography['ZIP3'] == '136')] = '12_8_13_3456'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '130') | (aou_census_geography['ZIP3'] == '131') | (aou_census_geography['ZIP3'] == '132')] = '13_012'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '137') | (aou_census_geography['ZIP3'] == '138') | (aou_census_geography['ZIP3'] == '139') | (aou_census_geography['ZIP3'] == '148') | (aou_census_geography['ZIP3'] == '149')] = '13_789_14_89'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '140') | (aou_census_geography['ZIP3'] == '141') | (aou_census_geography['ZIP3'] == '142') | (aou_census_geography['ZIP3'] == '147')] = '14_0127'
    aou_census_geography['MOD_ZIP3'].loc[(aou_census_geography['ZIP3'] == '143') | (aou_census_geography['ZIP3'] == '144') | (aou_census_geography['ZIP3'] == '145') | (aou_census_geography['ZIP3'] == '146')] = '14_3456'

    zip3_table = aou_census_geography.drop(columns='ZIP3')

#  Identify individuals w/ >1 EHR site and keep the site with the higher timestamp (e.g., later EHR site)
#  Then, drop the time column because it is redundant
ehr_no_dup = ehr_site_table.sort_values(by='TIME', ascending=True).drop_duplicates(subset='person_id', keep='last').drop(columns='TIME')

# Save processed row-level data to avoid redundancy
if registered_tier:
    aou_data = (pop_cleaned.merge(right=survey_table, how='inner', on='person_id').
                merge(right=ehr_no_dup, how='left', on='person_id'))
else:
    aou_data = (pop_cleaned.merge(right=zip3_table, how='inner', on='person_id').
                merge(right=survey_table, how='inner', on='person_id').
                merge(right=ehr_no_dup, how='left', on='person_id'))
aou_data.to_pickle('~/data/aourp_row_level_data.pkl')

if registered_tier:
    pass
else:
    ###  Generates quarterly cohorts for ZIP3 regions
    quarter_end_dates = pd.date_range(start=aou_data['SURVEY_TIME'].min(), end=aou_data['SURVEY_TIME'].max(), freq='3M')
    indiv_cohort_by_quarter = []
    cumu_cohort_by_quarter = []

    for i in range(len(quarter_end_dates) - 1):
        cumu_cohort_by_quarter.append(aou_data[(aou_data['SURVEY_TIME'] < quarter_end_dates[i + 1])])
        indiv_cohort_by_quarter.append(aou_data[(aou_data['SURVEY_TIME'] >= quarter_end_dates[i]) & (
                aou_data['SURVEY_TIME'] < quarter_end_dates[i + 1])])

    cumu_cohort_by_quarter.append(aou_data)
    indiv_cohort_by_quarter.append(aou_data[(aou_data['SURVEY_TIME'] >= quarter_end_dates[-1])])
    quarterly_cumu_cohorts = []
    quarterly_indiv_cohorts = []

    for i, quarterly_cohort in enumerate(cumu_cohort_by_quarter):
        zip3_to_quarterly_cohort = {}
        for zip3 in quarterly_cohort['MOD_ZIP3'].unique():
            if zip3 in ['000', '006', '007', '008', '009', '202', '969']:
                #  These are ZIP3 codes outside the 50 states
                pass
            else:
                cohort_by_zip = quarterly_cohort[quarterly_cohort['MOD_ZIP3'] == zip3]
                cohort_demo = cohort_by_zip.groupby(['AGE', 'GENDER', 'RACE', 'ETH']).size()
                temp_arr = np.zeros(shape=(3, 2, 5, 2))
                for age_idx, age in enumerate(['20-44', '45-64', '65+']):
                    for gender_idx, gender in enumerate(['WOMAN', 'MAN']):
                        for race_idx, race in enumerate(['ASIAN', 'BLACK', 'NHPI', 'MIXED', 'WHITE']):
                            for eth_idx, eth in enumerate(['HL', 'NH']):
                                if (age, gender, race, eth) in cohort_demo.index:
                                    temp_arr[age_idx, gender_idx, race_idx, eth_idx] = cohort_demo[age, gender, race, eth]
                                else:
                                    temp_arr[age_idx, gender_idx, race_idx, eth_idx] = 0

                zip3_to_quarterly_cohort[zip3] = temp_arr.reshape(-1)
        zip3_qc = pd.DataFrame.from_dict(
            {'MOD_ZIP3': zip3_to_quarterly_cohort.keys(), 'DEMOGRAPHICS': zip3_to_quarterly_cohort.values()})
        quarterly_cumu_cohorts.append(zip3_qc)

    for i, quarterly_cohort in enumerate(indiv_cohort_by_quarter):
        zip3_to_quarterly_cohort = {}
        for zip3 in quarterly_cohort['MOD_ZIP3'].unique():
            if zip3 in ['000', '006', '007', '008', '009', '202', '969']:
                #  These are ZIP3 codes outside the 50 states
                pass
            else:
                cohort_by_zip = quarterly_cohort[quarterly_cohort['MOD_ZIP3'] == zip3]
                cohort_demo = cohort_by_zip.groupby(['AGE', 'GENDER', 'RACE', 'ETH']).size()
                temp_arr = np.zeros(shape=(3, 2, 5, 2))
                for age_idx, age in enumerate(['20-44', '45-64', '65+']):
                    for gender_idx, gender in enumerate(['WOMAN', 'MAN']):
                        for race_idx, race in enumerate(['ASIAN', 'BLACK', 'NHPI', 'MIXED', 'WHITE']):
                            for eth_idx, eth in enumerate(['HL', 'NH']):
                                if (age, gender, race, eth) in cohort_demo.index:
                                    temp_arr[age_idx, gender_idx, race_idx, eth_idx] = cohort_demo[age, gender, race, eth]
                                else:
                                    temp_arr[age_idx, gender_idx, race_idx, eth_idx] = 0

                zip3_to_quarterly_cohort[zip3] = temp_arr.reshape(-1)
        zip3_qc = pd.DataFrame.from_dict(
            {'MOD_ZIP3': zip3_to_quarterly_cohort.keys(), 'DEMOGRAPHICS': zip3_to_quarterly_cohort.values()})
        quarterly_indiv_cohorts.append(zip3_qc)

    # Save data to disk
    for i in range(len(quarterly_cumu_cohorts)):
        quarterly_cumu_cohorts[i].to_pickle(f'~/data/aou_quarterly_cumu_cohorts_{i}.pkl')
        quarterly_indiv_cohorts[i].to_pickle(f'~/data/aou_quarterly_indiv_cohorts_{i}.pkl')

###  Generates quarterly cohorts for EHR site regions
aou_no_ehr_site_mask = pd.isna(aou_data['SITE'])
aou_ehr_site = aou_data[~aou_no_ehr_site_mask]
aou_no_ehr_site = aou_data[aou_no_ehr_site_mask]
quarter_end_dates = pd.date_range(start=aou_data['SURVEY_TIME'].min(), end=aou_data['SURVEY_TIME'].max(), freq='3M')
aou_data = aou_ehr_site
indiv_cohort_by_quarter = []
cumu_cohort_by_quarter = []

for i in range(len(quarter_end_dates) - 1):
    cumu_cohort_by_quarter.append(aou_data[(aou_data['SURVEY_TIME'] < quarter_end_dates[i + 1])])
    indiv_cohort_by_quarter.append(aou_data[(aou_data['SURVEY_TIME'] >= quarter_end_dates[i]) & (
                aou_data['SURVEY_TIME'] < quarter_end_dates[i + 1])])

cumu_cohort_by_quarter.append(aou_data)
indiv_cohort_by_quarter.append(aou_data[(aou_data['SURVEY_TIME'] >= quarter_end_dates[-1])])

quarterly_site_cumu_cohorts = []
quarterly_site_indiv_cohorts = []

for i, quarterly_cohort in enumerate(cumu_cohort_by_quarter):
    zip3_to_quarterly_cohort = {}
    for zip3 in quarterly_cohort['SITE'].unique().astype(str):
        cohort_by_zip = quarterly_cohort[quarterly_cohort['SITE'].astype(str) == zip3]
        cohort_demo = cohort_by_zip.groupby(['AGE', 'GENDER', 'RACE', 'ETH']).size()
        temp_arr = np.zeros(shape=(3, 2, 5, 2))
        for age_idx, age in enumerate(['20-44', '45-64', '65+']):
            for gender_idx, gender in enumerate(['WOMAN', 'MAN']):
                for race_idx, race in enumerate(['ASIAN', 'BLACK', 'NHPI', 'MIXED', 'WHITE']):
                    for eth_idx, eth in enumerate(['HL', 'NH']):
                        if (age, gender, race, eth) in cohort_demo.index:
                            temp_arr[age_idx, gender_idx, race_idx, eth_idx] = cohort_demo[age, gender, race, eth]
                        else:
                            temp_arr[age_idx, gender_idx, race_idx, eth_idx] = 0

        zip3_to_quarterly_cohort[zip3] = temp_arr.reshape(-1)
    zip3_qc = pd.DataFrame.from_dict(
        {'SITE': zip3_to_quarterly_cohort.keys(), 'DEMOGRAPHICS': zip3_to_quarterly_cohort.values()})
    quarterly_site_cumu_cohorts.append(zip3_qc)

for i, quarterly_cohort in enumerate(indiv_cohort_by_quarter):
    zip3_to_quarterly_cohort = {}
    for zip3 in quarterly_cohort['SITE'].unique().astype(str):
        cohort_by_zip = quarterly_cohort[quarterly_cohort['SITE'].astype(str) == zip3]
        cohort_demo = cohort_by_zip.groupby(['AGE', 'GENDER', 'RACE', 'ETH']).size()
        temp_arr = np.zeros(shape=(3, 2, 5, 2))
        for age_idx, age in enumerate(['20-44', '45-64', '65+']):
            for gender_idx, gender in enumerate(['WOMAN', 'MAN']):
                for race_idx, race in enumerate(['ASIAN', 'BLACK', 'NHPI', 'MIXED', 'WHITE']):
                    for eth_idx, eth in enumerate(['HL', 'NH']):
                        if (age, gender, race, eth) in cohort_demo.index:
                            temp_arr[age_idx, gender_idx, race_idx, eth_idx] = cohort_demo[age, gender, race, eth]
                        else:
                            temp_arr[age_idx, gender_idx, race_idx, eth_idx] = 0

        zip3_to_quarterly_cohort[zip3] = temp_arr.reshape(-1)
    zip3_qc = pd.DataFrame.from_dict(
        {'SITE': zip3_to_quarterly_cohort.keys(), 'DEMOGRAPHICS': zip3_to_quarterly_cohort.values()})
    quarterly_site_indiv_cohorts.append(zip3_qc)

for i in range(len(quarterly_site_cumu_cohorts)):
    quarterly_site_cumu_cohorts[i].to_pickle(f'~/data/ehr_sites/aou_quarterly_cumu_cohorts_{i}.pkl')
    quarterly_site_indiv_cohorts[i].to_pickle(f'~/data/ehr_sites/aou_quarterly_indiv_cohorts_{i}.pkl')

aou_data = aou_no_ehr_site
indiv_cohort_by_quarter = []

for i in range(len(quarter_end_dates) - 1):
    indiv_cohort_by_quarter.append(aou_data[(aou_data['SURVEY_TIME'] >= quarter_end_dates[i]) & (
                aou_data['SURVEY_TIME'] < quarter_end_dates[i + 1])])
indiv_cohort_by_quarter.append(aou_data[(aou_data['SURVEY_TIME'] >= quarter_end_dates[-1])])

quarterly_site_indiv_cohorts = []

for i, quarterly_cohort in enumerate(indiv_cohort_by_quarter):
    cohort_by_zip = quarterly_cohort
    cohort_demo = cohort_by_zip.groupby(['AGE', 'GENDER', 'RACE', 'ETH']).size()
    temp_arr = np.zeros(shape=(3, 2, 5, 2))
    for age_idx, age in enumerate(['20-44', '45-64', '65+']):
        for gender_idx, gender in enumerate(['WOMAN', 'MAN']):
            for race_idx, race in enumerate(['ASIAN', 'BLACK', 'NHPI', 'MIXED', 'WHITE']):
                for eth_idx, eth in enumerate(['HL', 'NH']):
                    if (age, gender, race, eth) in cohort_demo.index:
                        temp_arr[age_idx, gender_idx, race_idx, eth_idx] = cohort_demo[age, gender, race, eth]
                    else:
                        temp_arr[age_idx, gender_idx, race_idx, eth_idx] = 0

    zip3_to_quarterly_cohort = temp_arr.reshape(-1)
    zip3_qc = pd.Series(zip3_to_quarterly_cohort, name='DEMOGRAPHICS')
    quarterly_site_indiv_cohorts.append(zip3_qc)

for i in range(len(quarterly_site_cumu_cohorts)):
    indiv_cohort = quarterly_site_indiv_cohorts[i]
    indiv_cohort.to_pickle(f'~/data/ehr_sites/aou_quarterly_indiv_cohorts_no_site_{i}.pkl')


aou_data = pd.read_pickle('~/data/aourp_row_level_data.pkl')
cumu_cohort_by_quarter = []
quarterly_site_cumu_cohorts = []

for i in range(len(quarter_end_dates) - 1):
    cumu_cohort_by_quarter.append(aou_data[(aou_data['SURVEY_TIME'] < quarter_end_dates[i+1])])
cumu_cohort_by_quarter.append(aou_data)

for i, quarterly_cohort in enumerate(cumu_cohort_by_quarter):
    zip3_to_quarterly_cohort = {}
    for zip3 in quarterly_cohort['SITE'].unique().astype(str):
        cohort_by_zip = quarterly_cohort[quarterly_cohort['SITE'].astype(str) == zip3]
        cohort_demo = cohort_by_zip.groupby(['AGE', 'GENDER', 'RACE', 'ETH']).size()
        temp_arr = np.zeros(shape=(3, 2, 5, 2))
        for age_idx, age in enumerate(['20-44', '45-64', '65+']):
            for gender_idx, gender in enumerate(['WOMAN', 'MAN']):
                for race_idx, race in enumerate(['ASIAN', 'BLACK', 'NHPI', 'MIXED', 'WHITE']):
                    for eth_idx, eth in enumerate(['HL', 'NH']):
                        if (age, gender, race, eth) in cohort_demo.index:
                            temp_arr[age_idx, gender_idx, race_idx, eth_idx] = cohort_demo[age, gender, race, eth]
                        else:
                            temp_arr[age_idx, gender_idx, race_idx, eth_idx] = 0
        zip3_to_quarterly_cohort[zip3] = temp_arr.reshape(-1)
    zip3_qc = pd.DataFrame.from_dict({'SITE':zip3_to_quarterly_cohort.keys(), 'DEMOGRAPHICS':zip3_to_quarterly_cohort.values()})
    quarterly_site_cumu_cohorts.append(zip3_qc)

for i in range(len(quarterly_site_cumu_cohorts)):
    cumu_cohort = quarterly_site_cumu_cohorts[i]
    cumu_cohort.to_pickle(f'~/data/ehr_sites/aou_quarterly_cumu_cohorts_including_nan_{i}.pkl')

