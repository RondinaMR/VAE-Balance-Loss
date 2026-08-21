from collections import defaultdict
# if only_categorical is True, usecols is overriden
def adult_preprocessing(df):
    print(f'Total NaN values: {df.isna().sum().sum()}')
    print(f'Rows with one or more NaNs: {df.isna().any(axis=1).sum()}')
    print(f'Total rows: {len(df.index)}')
    df = df.dropna()
    print('NaN values dropped')
    print(f'Total NaN values: {df.isna().sum().sum()}')
    print(f'Rows with one or more NaNs: {df.isna().any(axis=1).sum()}')
    print(f'Total rows: {len(df.index)}')
    return df
adult = {
    "id": "01_adult",
    "name": "Adult",
    "train_path": "fairness_datasets/adult/adult.data",        
    "validation_path": None, 
    "target_column": "income",
    "target_positiveclass": ">50K",
    "regression": False,
    "categorical_columns": ['work_class', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income'],
    "sensitive_features": ['race', 'sex', 'native_country'],
    "imbalance_features": ['work_class', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income'],
    "num_rows": 32561,
    'names': ['age', 'work_class', 'fnlwgt', 'education', 'education-num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'],
    'header': None,
    'dtype': {'age': 'int', 'work_class': 'str', 'fnlwgt': 'int', 'education': 'str', 'education-num': 'int', 'marital_status': 'str', 'occupation': 'str', 'relationship': 'str', 'race': 'str', 'sex': 'str', 'capital_gain': 'int', 'capital_loss': 'int', 'hours_per_week': 'int', 'native_country': 'str', 'income': 'str'},
    'usecols': ['age', 'work_class', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'],
    'sep': ",",
    'na_values': "?",
    'preprocess': adult_preprocessing,
    'url': 'https://archive.ics.uci.edu/dataset/2/adult',
    'download-date': '2024-11-04',
    'fair_column_mappings': {
            'sex': {
                'mapping': {' Male': 1.0, ' Female': 0.0},
                'privileged': [1.0],
                'unprivileged': [0.0]
                },
            'race': {
                'mapping': {' White': 0, ' Black': 1, ' Asian-Pac-Islander': 2, ' Amer-Indian-Eskimo': 3, ' Other': 4},
                'privileged': [0],
                'unprivileged': [1, 2, 3, 4]
            },
            'native_country': {
                'mapping': {
                    ' United-States': 0, ' Mexico': 1, ' Philippines': 2, ' Germany': 3, ' Canada': 4, ' Puerto-Rico': 5, ' El-Salvador': 6, ' India': 7, ' Cuba': 8, ' England': 9, ' Jamaica': 10, ' South': 11, ' China': 12, ' Italy': 13, ' Dominican-Republic': 14, ' Vietnam': 15, ' Guatemala': 16, ' Japan': 17, ' Poland': 18, ' Columbia': 19, ' Taiwan': 20, ' Haiti': 21, ' Iran': 22, ' Portugal': 23, ' Nicaragua': 24, ' Peru': 25, ' France': 26, ' Greece': 27, ' Ecuador': 28, ' Ireland': 29, ' Hong': 30, ' Cambodia': 31, ' Trinadad&Tobago': 32, ' Laos': 33, ' Thailand': 34, ' Yugoslavia': 35, ' Outlying-US(Guam-USVI-etc)': 36, ' Honduras': 37, ' Hungary': 38, ' Scotland': 39, ' Holand-Netherlands': 40
                    },
                'privileged': [0],
                'unprivileged': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
            }
    }
}
# bank_marketing = {
#     "id": "02_bank-marketing",
#     "name": "Bank Marketing",
#     "train_path": "fairness_datasets/bank_marketing/bank-additional/bank-additional/bank-additional-full.csv",
#     "validation_path": None,
#     "target_column": "y",
#     "target_positiveclass": "yes",
#     "regression": False,
#     "categorical_columns": ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y'],
#     "sensitive_features": ['job', 'marital', 'education', 'housing'],
#     "imbalance_features": ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y'],
#     "num_rows": 41188,
#     'names': ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'],
#     'header': 0,
#     'dtype': {'age': 'int', 'job': 'str', 'marital': 'str', 'education': 'str', 'default': 'str', 'housing': 'str', 'loan': 'str', 'contact': 'str', 'month': 'str', 'day_of_week': 'str', 'duration': 'int', 'campaign': 'int', 'pdays': 'int', 'previous': 'int', 'poutcome': 'str', 'emp.var.rate': 'float', 'cons.price.idx': 'float', 'cons.conf.idx': 'float', 'euribor3m': 'float', 'nr.employed': 'float', 'y': 'str'},
#     "sep": ";",
#     # "na_values": None,
#     'url': 'https://archive.ics.uci.edu/dataset/222/bank+marketing',
#     'download_date': '2024-11-04',
#     'fair_column_mappings': {
#             'job': {
#                 'mapping': {'admin.': 0.0, 'blue-collar': 1.0, 'technician': 2.0, 'services': 3.0, 'management': 4.0, 'retired': 5.0, 'entrepreneur': 6.0, 'self-employed': 7.0, 'housemaid': 8.0, 'unemployed': 9.0, 'student': 10.0, 'unknown': 11.0},
#                 'privileged': [0.0, 4.0, 6.0],
#                 'unprivileged': [1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.0]
#                 },
#             'marital': {
#                 'mapping': {'divorced': 0.0, 'married': 1.0, 'single': 2.0, 'unknown': 3.0},
#                 'privileged': [1.0],
#                 'unprivileged': [0.0, 2.0, 3.0]
#             },
#             'education': {
#                 'mapping': {
#                     'basic.4y': 0.0, 'high.school': 1.0, 'basic.6y': 2.0, 'basic.9y': 3.0, 'professional.course': 4.0, 'university.degree': 5.0, 'illiterate': 6.0, 'unknown': 7.0},
#                 'privileged': [5.0],
#                 'unprivileged': [0.0,1.0,2.0,3.0,4.0,6.0,7.0]
#             },
#             'housing': {
#                 'mapping': {
#                     'no': 0.0, 'yes': 1.0, 'unknown': 2.0},
#                 'privileged': [0.0],
#                 'unprivileged': [1.0, 2.0]
#             }
#     }
# }
south_german_credit = {
    "id": "03_south-german-credit",
    "name": "South German Credit",
    "train_path": "fairness_datasets/south_german_credit/SouthGermanCredit.asc",
    "validation_path": None,
    "sep": " ",
    "target_column": "kredit",
    "target_positiveclass": "1",
    "regression": False,
    'names': ['laufkont', 'laufzeit', 'moral', 'verw', 'hoehe', 'sparkont', 'beszeit', 'rate', 'famges', 'buerge', 'wohnzeit', 'verm', 'alter', 'weitkred', 'wohn', 'bishkred', 'beruf', 'pers', 'telef', 'gastarb', 'kredit'],
    'header': 0,
    'dtype': {'laufkont': 'str', 'laufzeit': 'int', 'moral': 'str', 'verw': 'str', 'hoehe': 'int', 'sparkont': 'str', 'beszeit': 'str', 'rate': 'str', 'famges': 'str', 'buerge': 'str', 'wohnzeit': 'str', 'verm': 'str', 'alter': 'int', 'weitkred': 'str', 'wohn': 'str', 'bishkred': 'str', 'beruf': 'str', 'pers': 'str', 'telef': 'str', 'gastarb': 'str', 'kredit': 'str'},
    "categorical_columns": ['laufkont', 'moral', 'verw', 'sparkont', 'beszeit', 'rate', 'famges', 'buerge', 'wohnzeit', 'verm', 'weitkred', 'wohn', 'bishkred', 'beruf', 'pers', 'telef', 'gastarb', 'kredit'],
    "sensitive_features": ['laufkont', 'sparkont', 'famges', 'verm', 'wohn', 'gastarb'],
    "imbalance_features": ['laufkont', 'moral', 'verw', 'sparkont', 'beszeit', 'rate', 'famges', 'buerge', 'wohnzeit', 'verm', 'weitkred', 'wohn', 'bishkred', 'beruf', 'pers', 'telef', 'gastarb', 'kredit'],
    "num_rows": 1000,
    'download-date': '2024-10-28',
    'fair_column_mappings': {
            'famges': {
                'mapping': {'1': 0.0, '2': 1.0, '3': 2.0, '4': 3.0}, 
                # 1 : male : divorced/separated; 
                # 2 : female : non-single or male : single
                # 3 : male : married/widowed * 
                # 4 : female : single
                'privileged': [2.0],
                'unprivileged': [0.0, 1.0, 3.0]
                },
            'gastarb': {
                'mapping': {'1': 0.0, '2': 1.0}, 
                # foreign worker?
                #  1 : yes
                #  2 : no 
                'privileged': [1.0],
                'unprivileged': [0.0]
                }
    },
    'url': 'https://archive.ics.uci.edu/dataset/573/south+german+credit+update',
    'usecols': ['laufkont', 'laufzeit', 'moral', 'verw', 'hoehe', 'sparkont', 'beszeit', 'rate', 'famges', 'buerge', 'wohnzeit', 'verm', 'alter', 'weitkred', 'wohn', 'bishkred', 'beruf', 'pers', 'telef', 'gastarb', 'kredit']
}
# gastarb=foreign_work; laufkont='status' (of checking account);
# famges='personal_status_sex', beruf='job', verm=savings; krisk='credit_risk'
# COMPAS
def compas_preprocessing(df):
    # df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score',  'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
    df = df[(df['days_b_screening_arrest'] <= 30)]
    df = df[(df['days_b_screening_arrest'] >= -30)]
    df = df[(df['is_recid'] != -1)]
    df = df[(df['c_charge_degree'] != 'O')]
    df = df[(df['score_text'] != 'N/A')]
    df = df[(df['score_text'] != 'v')]
    df = df.dropna(subset=['days_b_screening_arrest'])
    print('COMPAS Preprocessing number of rows: {}'.format(len(df.index)))
    return df
compas_score_two_years = {
    'id': '04_compas-two-years',
    'name': 'COMPAS Score Two Years',
    'train_path': 'fairness_datasets/compas/compas-scores-two-years.csv',
    'validation_path': None,
    'target_column': 'score_text',
    # Positive (favorable) class = 'Low' only, i.e. the unfavorable outcome is Medium + High.
    # This matches ProPublica's own recoding in "How We Analyzed the COMPAS Recidivism
    # Algorithm", where score_factor is HighScore for every score_text other than 'Low'
    # (Low = decile 1-4, Medium = 5-7, High = 8-10).
    'target_positiveclass': ['Low'],
    'regression': False,
    'names': ['id', 'name', 'first', 'last', 'compas_screening_date', 'sex', 'dob', 'age', 'age_cat', 'race', 'juv_fel_count', 'decile_score1', 'juv_misd_count', 'juv_other_count', 'priors_count1', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'c_charge_degree', 'c_charge_desc', 'is_recid', 'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out', 'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc', 'type_of_assessment', 'decile_score2', 'score_text', 'screening_date', 'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date', 'in_custody', 'out_custody', 'priors_count2', 'start', 'end', 'event', 'two_year_recid'],
    # 'dtype': {'id': 'int', 'name': 'str', 'first': 'str', 'last': 'str', 'compas_screening_date': 'str', 'sex': 'str', 'dob': 'str', 'age': 'int', 'age_cat': 'str', 'race': 'str', 'juv_fel_count': 'int', 'decile_score': 'int', 'juv_misd_count': 'int', 'juv_other_count': 'int', 'priors_count': 'int', 'days_b_screening_arrest': 'int', 'c_jail_in': 'datetime64', 'c_jail_out': 'datetime64', 'c_case_number': 'str', 'c_offense_date': 'str', 'c_arrest_date': 'str', 'c_days_from_compas': 'int', 'c_charge_degree': 'str', 'c_charge_desc': 'str', 'is_recid': 'int', 'r_case_number': 'str', 'r_charge_degree': 'str', 'r_days_from_arrest': 'int', 'r_offense_date': 'str', 'r_charge_desc': 'str', 'r_jail_in': 'str', 'r_jail_out': 'str', 'violent_recid': 'int', 'is_violent_recid': 'int', 'vr_case_number': 'str', 'vr_charge_degree': 'str', 'vr_offense_date': 'str', 'vr_charge_desc': 'str', 'type_of_assessment': 'str', 'decile_score': 'int', 'score_text': 'str', 'screening_date': 'str', 'v_type_of_assessment': 'str', 'v_decile_score': 'int', 'v_score_text': 'str', 'v_screening_date': 'str', 'in_custody': 'str', 'out_custody': 'str', 'priors_count': 'int', 'start': 'str', 'end': 'str', 'event': 'str', 'two_year_recid': 'int'},
    'dtype': {'name': 'str', 'first': 'str', 'last': 'str', 'sex': 'str', 'age_cat': 'str', 'race': 'str', 'decile_score1': 'float', 'priors_count1': 'float', 'days_b_screening_arrest': 'float', 'c_jail_in': 'str', 'c_jail_out': 'str', 'c_charge_degree': 'str', 'c_charge_desc': 'str', 'is_recid': 'str', 'score_text': 'str','two_year_recid': 'str'},
    # 'parse_dates': ['c_jail_in', 'c_jail_out'],
    # 'date_format': '%Y-%m-%d %H:%M:%S',
    'usecols': ['sex', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count1', 'days_b_screening_arrest', 'c_charge_degree', 'c_charge_desc', 'is_recid', 'is_violent_recid', 'r_charge_desc', 'priors_count2', 'score_text', 'two_year_recid'],
    'categorical_columns': ['sex', 'age_cat', 'race', 'c_charge_degree', 'is_recid', 'score_text', 'two_year_recid'],
    'sensitive_features': ['sex', 'age_cat', 'race'],
    'imbalance_features': ['sex', 'age_cat', 'race', 'c_charge_degree', 'is_recid', 'score_text', 'two_year_recid'],
    'fair_column_mappings': {
            'sex': {
                'mapping': {'Male': 1.0, 'Female': 0.0},
                'privileged': [1.0],
                'unprivileged': [0.0]
                },
            'race': {
                'mapping': {'African-American': 0, 'Caucasian': 1, 'Hispanic': 2, 'Native American': 3, 'Asian': 4, 'Other': 5},
                'privileged': [1],
                'unprivileged': [0, 2, 3, 4, 5]
            },
            'age_cat': {
                'mapping': {'25 - 45': 0, 'Less than 25': 1, 'Greater than 45': 2},
                'privileged': [2],
                'unprivileged': [0, 1]
            }
    },
    'num_rows': 7214,
    'header': 0,
    'sep': ',',
    # 'na_values': "?",
    'preprocess': compas_preprocessing,
    'url': 'https://github.com/propublica/compas-analysis',
    'download-date': '2024-11-06'
}
# arrhythmia_defaultdict = defaultdict(lambda: 'float', {
#     # 'age': 'Int64', 
#     'sex': 'str',
#     # 'height': 'Int64', 'weight': 'Int64','QRS_duration': 'Int64', 'P-R_interval': 'Int64', 'Q-T_interval': 'Int64', 'T_interval': 'Int64', 'P_interval': 'Int64', 'QRS': 'Int64', 'T': 'Int64', 'P': 'Int64', 'QRST': 'Int64', 'J': 'Int64', 'heart_rate': 'Int64',
#     # 'DI_Q_wave': 'Int64', 'DI_R_wave': 'Int64', 'DI_S_wave': 'Int64', 'DI_R_prime_wave': 'Int64', 'DI_S_prime_wave': 'Int64', 'DI_intrinsic_deflections': 'Int64',
#     'DI_ragged_R_wave': 'str', 'DI_diphasic_R_wave': 'str', 'DI_ragged_P_wave': 'str', 'DI_diphasic_P_wave': 'str', 'DI_ragged_T_wave': 'str', 'DI_diphasic_T_wave': 'str',
#     # 'DII_Q_wave': 'Int64', 'DII_R_wave': 'Int64', 'DII_S_wave': 'Int64', 'DII_R_prime_wave': 'Int64', 'DII_S_prime_wave': 'Int64', 'DII_intrinsic_deflections': 'Int64',
#     'DII_ragged_R_wave': 'str', 'DII_diphasic_R_wave': 'str', 'DII_ragged_P_wave': 'str', 'DII_diphasic_P_wave': 'str', 'DII_ragged_T_wave': 'str', 'DII_diphasic_T_wave': 'str',
#     # 'DIII_Q_wave': 'Int64', 'DIII_R_wave': 'Int64', 'DIII_S_wave': 'Int64', 'DIII_R_prime_wave': 'Int64', 'DIII_S_prime_wave': 'Int64', 'DIII_intrinsic_deflections': 'Int64',
#     'DIII_ragged_R_wave': 'str', 'DIII_diphasic_R_wave': 'str', 'DIII_ragged_P_wave': 'str', 'DIII_diphasic_P_wave': 'str', 'DIII_ragged_T_wave': 'str', 'DIII_diphasic_T_wave': 'str',
#     # 'AVR_Q_wave': 'Int64', 'AVR_R_wave': 'Int64', 'AVR_S_wave': 'Int64', 'AVR_R_prime_wave': 'Int64', 'AVR_S_prime_wave': 'Int64', 'AVR_intrinsic_deflections': 'Int64',
#     'AVR_ragged_R_wave': 'str', 'AVR_diphasic_R_wave': 'str', 'AVR_ragged_P_wave': 'str', 'AVR_diphasic_P_wave': 'str', 'AVR_ragged_T_wave': 'str', 'AVR_diphasic_T_wave': 'str',
#     # 'AVL_Q_wave': 'Int64', 'AVL_R_wave': 'Int64', 'AVL_S_wave': 'Int64', 'AVL_R_prime_wave': 'Int64', 'AVL_S_prime_wave': 'Int64', 'AVL_intrinsic_deflections': 'Int64',
#     'AVL_ragged_R_wave': 'str', 'AVL_diphasic_R_wave': 'str', 'AVL_ragged_P_wave': 'str', 'AVL_diphasic_P_wave': 'str', 'AVL_ragged_T_wave': 'str', 'AVL_diphasic_T_wave': 'str',
#     # 'AVF_Q_wave': 'Int64', 'AVF_R_wave': 'Int64', 'AVF_S_wave': 'Int64', 'AVF_R_prime_wave': 'Int64', 'AVF_S_prime_wave': 'Int64', 'AVF_intrinsic_deflections': 'Int64',
#     'AVF_ragged_R_wave': 'str', 'AVF_diphasic_R_wave': 'str', 'AVF_ragged_P_wave': 'str', 'AVF_diphasic_P_wave': 'str', 'AVF_ragged_T_wave': 'str', 'AVF_diphasic_T_wave': 'str',
#     # 'V1_Q_wave': 'Int64', 'V1_R_wave': 'Int64', 'V1_S_wave': 'Int64', 'V1_R_prime_wave': 'Int64', 'V1_S_prime_wave': 'Int64', 'V1_intrinsic_deflections': 'Int64',
#     'V1_ragged_R_wave': 'str', 'V1_diphasic_R_wave': 'str', 'V1_ragged_P_wave': 'str', 'V1_diphasic_P_wave': 'str', 'V1_ragged_T_wave': 'str', 'V1_diphasic_T_wave': 'str',
#     # 'V2_Q_wave': 'Int64', 'V2_R_wave': 'Int64', 'V2_S_wave': 'Int64', 'V2_R_prime_wave': 'Int64', 'V2_S_prime_wave': 'Int64', 'V2_intrinsic_deflections': 'Int64',
#     'V2_ragged_R_wave': 'str', 'V2_diphasic_R_wave': 'str', 'V2_ragged_P_wave': 'str', 'V2_diphasic_P_wave': 'str', 'V2_ragged_T_wave': 'str', 'V2_diphasic_T_wave': 'str',
#     # 'V3_Q_wave': 'Int64', 'V3_R_wave': 'Int64', 'V3_S_wave': 'Int64', 'V3_R_prime_wave': 'Int64', 'V3_S_prime_wave': 'Int64', 'V3_intrinsic_deflections': 'Int64',
#     'V3_ragged_R_wave': 'str', 'V3_diphasic_R_wave': 'str', 'V3_ragged_P_wave': 'str', 'V3_diphasic_P_wave': 'str', 'V3_ragged_T_wave': 'str', 'V3_diphasic_T_wave': 'str',
#     # 'V4_Q_wave': 'Int64', 'V4_R_wave': 'Int64', 'V4_S_wave': 'Int64', 'V4_R_prime_wave': 'Int64', 'V4_S_prime_wave': 'Int64', 'V4_intrinsic_deflections': 'Int64',
#     'V4_ragged_R_wave': 'str', 'V4_diphasic_R_wave': 'str', 'V4_ragged_P_wave': 'str', 'V4_diphasic_P_wave': 'str', 'V4_ragged_T_wave': 'str', 'V4_diphasic_T_wave': 'str',
#     # 'V5_Q_wave': 'Int64', 'V5_R_wave': 'Int64', 'V5_S_wave': 'Int64', 'V5_R_prime_wave': 'Int64', 'V5_S_prime_wave': 'Int64', 'V5_intrinsic_deflections': 'Int64',
#     'V5_ragged_R_wave': 'str', 'V5_diphasic_R_wave': 'str', 'V5_ragged_P_wave': 'str', 'V5_diphasic_P_wave': 'str', 'V5_ragged_T_wave': 'str', 'V5_diphasic_T_wave': 'str',
#     # 'V6_Q_wave': 'Int64', 'V6_R_wave': 'Int64', 'V6_S_wave': 'Int64', 'V6_R_prime_wave': 'Int64', 'V6_S_prime_wave': 'Int64', 'V6_intrinsic_deflections': 'Int64',
#     'V6_ragged_R_wave': 'str', 'V6_diphasic_R_wave': 'str', 'V6_ragged_P_wave': 'str', 'V6_diphasic_P_wave': 'str', 'V6_ragged_T_wave': 'str', 'V6_diphasic_T_wave': 'str',
#     'class': 'str'
# })
# arrhythmia = {
#     'id': '05_arrhythmia',
#     'name': 'Arrhythmia',
#     'train_path': 'fairness_datasets/arrhythmia/arrhythmia.data',
#     'validation_path': None,
#     'target_column': 'class',
#     'regression': False,
#     'names': ['age', 'sex', 'height', 'weight', 'QRS_duration', 'P-R_interval', 'Q-T_interval', 'T_interval', 'P_interval', 'QRS', 'T', 'P', 'QRST', 'J', 'heart_rate'] + \
#     ['DI_Q_wave', 'DI_R_wave', 'DI_S_wave', 'DI_R_prime_wave', 'DI_S_prime_wave', 'DI_intrinsic_deflections', 'DI_ragged_R_wave', 'DI_diphasic_R_wave', 'DI_ragged_P_wave', 'DI_diphasic_P_wave', 'DI_ragged_T_wave', 'DI_diphasic_T_wave'] + \
#     ['DII_Q_wave', 'DII_R_wave', 'DII_S_wave', 'DII_R_prime_wave', 'DII_S_prime_wave', 'DII_intrinsic_deflections', 'DII_ragged_R_wave', 'DII_diphasic_R_wave', 'DII_ragged_P_wave', 'DII_diphasic_P_wave', 'DII_ragged_T_wave', 'DII_diphasic_T_wave'] + \
#     ['DIII_Q_wave', 'DIII_R_wave', 'DIII_S_wave', 'DIII_R_prime_wave', 'DIII_S_prime_wave', 'DIII_intrinsic_deflections', 'DIII_ragged_R_wave', 'DIII_diphasic_R_wave', 'DIII_ragged_P_wave', 'DIII_diphasic_P_wave', 'DIII_ragged_T_wave', 'DIII_diphasic_T_wave'] + \
#     ['AVR_Q_wave', 'AVR_R_wave', 'AVR_S_wave', 'AVR_R_prime_wave', 'AVR_S_prime_wave', 'AVR_intrinsic_deflections', 'AVR_ragged_R_wave', 'AVR_diphasic_R_wave', 'AVR_ragged_P_wave', 'AVR_diphasic_P_wave', 'AVR_ragged_T_wave', 'AVR_diphasic_T_wave'] + \
#     ['AVL_Q_wave', 'AVL_R_wave', 'AVL_S_wave', 'AVL_R_prime_wave', 'AVL_S_prime_wave', 'AVL_intrinsic_deflections', 'AVL_ragged_R_wave', 'AVL_diphasic_R_wave', 'AVL_ragged_P_wave', 'AVL_diphasic_P_wave', 'AVL_ragged_T_wave', 'AVL_diphasic_T_wave'] + \
#     ['AVF_Q_wave', 'AVF_R_wave', 'AVF_S_wave', 'AVF_R_prime_wave', 'AVF_S_prime_wave', 'AVF_intrinsic_deflections', 'AVF_ragged_R_wave', 'AVF_diphasic_R_wave', 'AVF_ragged_P_wave', 'AVF_diphasic_P_wave', 'AVF_ragged_T_wave', 'AVF_diphasic_T_wave'] + \
#     ['V1_Q_wave', 'V1_R_wave', 'V1_S_wave', 'V1_R_prime_wave', 'V1_S_prime_wave', 'V1_intrinsic_deflections', 'V1_ragged_R_wave', 'V1_diphasic_R_wave', 'V1_ragged_P_wave', 'V1_diphasic_P_wave', 'V1_ragged_T_wave', 'V1_diphasic_T_wave'] + \
#     ['V2_Q_wave', 'V2_R_wave', 'V2_S_wave', 'V2_R_prime_wave', 'V2_S_prime_wave', 'V2_intrinsic_deflections', 'V2_ragged_R_wave', 'V2_diphasic_R_wave', 'V2_ragged_P_wave', 'V2_diphasic_P_wave', 'V2_ragged_T_wave', 'V2_diphasic_T_wave'] + \
#     ['V3_Q_wave', 'V3_R_wave', 'V3_S_wave', 'V3_R_prime_wave', 'V3_S_prime_wave', 'V3_intrinsic_deflections', 'V3_ragged_R_wave', 'V3_diphasic_R_wave', 'V3_ragged_P_wave', 'V3_diphasic_P_wave', 'V3_ragged_T_wave', 'V3_diphasic_T_wave'] + \
#     ['V4_Q_wave', 'V4_R_wave', 'V4_S_wave', 'V4_R_prime_wave', 'V4_S_prime_wave', 'V4_intrinsic_deflections', 'V4_ragged_R_wave', 'V4_diphasic_R_wave', 'V4_ragged_P_wave', 'V4_diphasic_P_wave', 'V4_ragged_T_wave', 'V4_diphasic_T_wave'] + \
#     ['V5_Q_wave', 'V5_R_wave', 'V5_S_wave', 'V5_R_prime_wave', 'V5_S_prime_wave', 'V5_intrinsic_deflections', 'V5_ragged_R_wave', 'V5_diphasic_R_wave', 'V5_ragged_P_wave', 'V5_diphasic_P_wave', 'V5_ragged_T_wave', 'V5_diphasic_T_wave'] + \
#     ['V6_Q_wave', 'V6_R_wave', 'V6_S_wave', 'V6_R_prime_wave', 'V6_S_prime_wave', 'V6_intrinsic_deflections', 'V6_ragged_R_wave', 'V6_diphasic_R_wave', 'V6_ragged_P_wave', 'V6_diphasic_P_wave', 'V6_ragged_T_wave', 'V6_diphasic_T_wave'] + \
#     ['DI_JJ_wave', 'DI_Q_wave_amp', 'DI_R_wave_amp', 'DI_S_wave_amp', 'DI_R_prime_wave_amp', 'DI_S_prime_wave_amp', 'DI_P_wave_amp', 'DI_T_wave_amp', 'DI_QRSA', 'DI_QRSTA'] + \
#     ['DII_JJ_wave', 'DII_Q_wave_amp', 'DII_R_wave_amp', 'DII_S_wave_amp', 'DII_R_prime_wave_amp', 'DII_S_prime_wave_amp', 'DII_P_wave_amp', 'DII_T_wave_amp', 'DII_QRSA', 'DII_QRSTA'] + \
#     ['DIII_JJ_wave', 'DIII_Q_wave_amp', 'DIII_R_wave_amp', 'DIII_S_wave_amp', 'DIII_R_prime_wave_amp', 'DIII_S_prime_wave_amp', 'DIII_P_wave_amp', 'DIII_T_wave_amp', 'DIII_QRSA', 'DIII_QRSTA'] + \
#     ['AVR_JJ_wave', 'AVR_Q_wave_amp', 'AVR_R_wave_amp', 'AVR_S_wave_amp', 'AVR_R_prime_wave_amp', 'AVR_S_prime_wave_amp', 'AVR_P_wave_amp', 'AVR_T_wave_amp', 'AVR_QRSA', 'AVR_QRSTA'] + \
#     ['AVL_JJ_wave', 'AVL_Q_wave_amp', 'AVL_R_wave_amp', 'AVL_S_wave_amp', 'AVL_R_prime_wave_amp', 'AVL_S_prime_wave_amp', 'AVL_P_wave_amp', 'AVL_T_wave_amp', 'AVL_QRSA', 'AVL_QRSTA'] + \
#     ['AVF_JJ_wave', 'AVF_Q_wave_amp', 'AVF_R_wave_amp', 'AVF_S_wave_amp', 'AVF_R_prime_wave_amp', 'AVF_S_prime_wave_amp', 'AVF_P_wave_amp', 'AVF_T_wave_amp', 'AVF_QRSA', 'AVF_QRSTA'] + \
#     ['V1_JJ_wave', 'V1_Q_wave_amp', 'V1_R_wave_amp', 'V1_S_wave_amp', 'V1_R_prime_wave_amp', 'V1_S_prime_wave_amp', 'V1_P_wave_amp', 'V1_T_wave_amp', 'V1_QRSA', 'V1_QRSTA'] + \
#     ['V2_JJ_wave', 'V2_Q_wave_amp', 'V2_R_wave_amp', 'V2_S_wave_amp', 'V2_R_prime_wave_amp', 'V2_S_prime_wave_amp', 'V2_P_wave_amp', 'V2_T_wave_amp', 'V2_QRSA', 'V2_QRSTA'] + \
#     ['V3_JJ_wave', 'V3_Q_wave_amp', 'V3_R_wave_amp', 'V3_S_wave_amp', 'V3_R_prime_wave_amp', 'V3_S_prime_wave_amp', 'V3_P_wave_amp', 'V3_T_wave_amp', 'V3_QRSA', 'V3_QRSTA'] + \
#     ['V4_JJ_wave', 'V4_Q_wave_amp', 'V4_R_wave_amp', 'V4_S_wave_amp', 'V4_R_prime_wave_amp', 'V4_S_prime_wave_amp', 'V4_P_wave_amp', 'V4_T_wave_amp', 'V4_QRSA', 'V4_QRSTA'] + \
#     ['V5_JJ_wave', 'V5_Q_wave_amp', 'V5_R_wave_amp', 'V5_S_wave_amp', 'V5_R_prime_wave_amp', 'V5_S_prime_wave_amp', 'V5_P_wave_amp', 'V5_T_wave_amp', 'V5_QRSA', 'V5_QRSTA'] + \
#     ['V6_JJ_wave', 'V6_Q_wave_amp', 'V6_R_wave_amp', 'V6_S_wave_amp', 'V6_R_prime_wave_amp', 'V6_S_prime_wave_amp', 'V6_P_wave_amp', 'V6_T_wave_amp', 'V6_QRSA', 'V6_QRSTA'] + \
#     ['class'],
#     'header': None,
#     # ['T', 'P', 'QRST', 'J', 'heart_rate'] With NAN
#     'dtype': arrhythmia_defaultdict,
#     'categorical_columns': [ 
#         'sex', 'DI_ragged_R_wave', 'DI_diphasic_R_wave', 'DI_ragged_P_wave', 'DI_diphasic_P_wave', 'DI_ragged_T_wave', 'DI_diphasic_T_wave',
#         'DII_ragged_R_wave', 'DII_diphasic_R_wave', 'DII_ragged_P_wave', 'DII_diphasic_P_wave', 'DII_ragged_T_wave', 'DII_diphasic_T_wave',
#         'DIII_ragged_R_wave', 'DIII_diphasic_R_wave', 'DIII_ragged_P_wave', 'DIII_diphasic_P_wave', 'DIII_ragged_T_wave', 'DIII_diphasic_T_wave',
#         'AVR_ragged_R_wave', 'AVR_diphasic_R_wave', 'AVR_ragged_P_wave', 'AVR_diphasic_P_wave', 'AVR_ragged_T_wave', 'AVR_diphasic_T_wave',
#         'AVL_ragged_R_wave', 'AVL_diphasic_R_wave', 'AVL_ragged_P_wave', 'AVL_diphasic_P_wave', 'AVL_ragged_T_wave', 'AVL_diphasic_T_wave',
#         'AVF_ragged_R_wave', 'AVF_diphasic_R_wave', 'AVF_ragged_P_wave', 'AVF_diphasic_P_wave', 'AVF_ragged_T_wave', 'AVF_diphasic_T_wave',
#         'V1_ragged_R_wave', 'V1_diphasic_R_wave', 'V1_ragged_P_wave', 'V1_diphasic_P_wave', 'V1_ragged_T_wave', 'V1_diphasic_T_wave',
#         'V2_ragged_R_wave', 'V2_diphasic_R_wave', 'V2_ragged_P_wave', 'V2_diphasic_P_wave', 'V2_ragged_T_wave', 'V2_diphasic_T_wave',
#         'V3_ragged_R_wave', 'V3_diphasic_R_wave', 'V3_ragged_P_wave', 'V3_diphasic_P_wave', 'V3_ragged_T_wave', 'V3_diphasic_T_wave',
#         'V4_ragged_R_wave', 'V4_diphasic_R_wave', 'V4_ragged_P_wave', 'V4_diphasic_P_wave', 'V4_ragged_T_wave', 'V4_diphasic_T_wave',
#         'V5_ragged_R_wave', 'V5_diphasic_R_wave', 'V5_ragged_P_wave', 'V5_diphasic_P_wave', 'V5_ragged_T_wave', 'V5_diphasic_T_wave',
#         'V6_ragged_R_wave', 'V6_diphasic_R_wave', 'V6_ragged_P_wave', 'V6_diphasic_P_wave', 'V6_ragged_T_wave', 'V6_diphasic_T_wave',
#         'class'
#     ],
#     'sensitive_features': ['sex'],
#     'imbalance_features': ['sex', 'class'],
#     'num_rows': 452,
#     'sep': ',',
#     'na_values': "?",
#     'url': 'https://archive.ics.uci.edu/dataset/arrhythmia',
#     'download-date': '2024-11-12'
# }
# default_credit_card_defaultdict = defaultdict(lambda: 'int', {'SEX': 'str', 'EDUCATION': 'str', 'MARRIAGE': 'str', 'PAY_0': 'str', 'PAY_2': 'str', 'PAY_3': 'str', 'PAY_4': 'str', 'PAY_5': 'str', 'PAY_6': 'str', 'default payment next month': 'str'})
# default_credit_card = {
#     'id': '06_default-credit-card',
#     'name': 'Default of Credit Card Clients',
#     'train_path': 'fairness_datasets/default_of_credit_card_clients/default of credit card clients.csv',
#     'validation_path': None,
#     'target_column': 'default payment next month',
#     'regression': False,
#     'names': ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default payment next month'],
#     'header': 0,
#     'dtype': default_credit_card_defaultdict,
#     'categorical_columns': ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'default payment next month'],
#     'sensitive_features': ['SEX'],
#     'imbalance_features': ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'default payment next month'],
#     'num_rows': 30000,
#     'sep': ';',
#     'url': 'https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients',
#     'download-date': '2024-11-13'
# }
# drug_consumption_defaultdict = defaultdict(lambda: 'str')
# drug_consumption = {
#     'id': '07_drug-consumption',
#     'name': 'Drug Consumption',
#     'train_path': 'fairness_datasets/drug_consumption_quantified/drug_consumption.data',
#     'validation_path': None,
#     'target_column': 'legalh',
#     'regression': False,
#     'names': ['id', 'age', 'gender', 'education', 'country', 'ethnicity', 'nscore', 'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss', 'alcohol', 'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack', 'ecstasy', 'heroin', 'ketamine', 'legalh', 'lsd', 'meth', 'mushrooms', 'nicotine', 'semer', 'vsa'],
#     'dtype': drug_consumption_defaultdict,
#     'usecols': ['age', 'gender', 'education', 'country', 'ethnicity', 'nscore', 'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss', 'legalh'],
#     'categorical_columns': ['age', 'gender', 'education', 'country', 'ethnicity', 'nscore', 'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss', 'legalh'],
#     'sensitive_features': ['age', 'gender', 'country', 'ethnicity'],
#     'imbalance_features': ['age', 'gender', 'education', 'country', 'ethnicity', 'nscore', 'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss', 'legalh'],
#     'num_rows': 1885,
#     'header': None,
#     'sep': ',',
#     'url': 'https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified',
#     'download-date': '2024-11-14'
# }
# hearth_disease_defaultdict = defaultdict(lambda: 'str', {'age': 'float', 'trestbps': 'float', 'chol': 'float', 'thalach': 'float', 'oldpeak': 'float'})
# hearth_disease= {
#     'id': '08_hearth-disease',
#     'name': 'Hearth Disease',
#     'train_path': 'fairness_datasets/heart_disease/processed.cleveland.data',
#     'validation_path': None,
#     'target_column': 'num',
#     'regression': False,
#     'names': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'],
#     'dtype': hearth_disease_defaultdict,
#     # 'usecols': ,
#     'categorical_columns': ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'num'],
#     'sensitive_features': ['age', 'sex'],
#     'imbalance_features': ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'num'],
#     'num_rows': 303,
#     'header': None,
#     'sep': ',',
#     'na_values': '?',
#     'url': 'https://archive.ics.uci.edu/dataset/45/heart+disease',
#     'download-date': '2024-11-14'
# }

