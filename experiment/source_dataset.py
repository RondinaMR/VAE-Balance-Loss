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
                'unprivileged': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
            }
    }
}

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
    'target_positiveclass': ['Low', 'Medium'],
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


