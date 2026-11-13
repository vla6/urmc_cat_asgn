# urmc_nn_asgn/config.py
# Defines constants / configuration for this project

# Data columns 

DATA_COLS_ID = ['VISIT_ID']
DATA_COLS_CAT = ['icd10_cm_code', 'ccsr_code']
DATA_COLS_BINARY = ['FEMALE', 'AWEEKEND','ELECTIVE', 
                    'HOSP_CONTROL_For_profit', 'HOSP_CONTROL_Nonprofit',
                    'HOSP_CONTROL_Public']
DATA_COLS_CONT = ['AGE', 'LOS']
DATA_COL_TARGET = 'visit_cost'

# Feature cols for convenience
DATA_COLS_NUM = DATA_COLS_BINARY + DATA_COLS_CONT
DATA_COLS_FEATURES = DATA_COLS_CAT + DATA_COLS_NUM

# Training parameters

BATCH_SIZE = 256
NUM_WORKERS = 0
DROPOUT = 0.2
NN_DIM_1 = 128
NN_DIM_2 = 64