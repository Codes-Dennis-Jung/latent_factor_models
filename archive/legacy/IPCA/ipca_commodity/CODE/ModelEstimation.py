# =================================================
#
#       Factor Construction for Option Returns
#
# =================================================

import pandas as pd
import numpy as np
from ipca_prop.ipca import IPCARegressor
# from ipca import IPCARegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

import pickle
import os


# =============================================================================
#                            Settings
# =============================================================================
SampleFreq = 'Monthly'  # Daily, Weekly, Monthly
data_path = './../Data/'
results_path = '../weekly_RESULTS_rank_large_04_2018_07_2021/'
input_file = 'finaldata_large_weekly.csv'
Kmax = 9

# =============================================================================
#                            Load FX Data
# =============================================================================

# Load options data
fxdata = pd.read_csv(data_path+input_file, sep=',')
depvar = 'FX'
#Remove obs with missing LOCATION
fxdata.dropna(subset=['LOCATION'], inplace=True)



# all char
# predvars     = ['new_addresses', 'active_addresses', 'transaction_count', 'average_transaction_value', 'volumeto', 'ahcc', 'bidask', 'volshockSTD15', 'volshockSTD30', 'marketcap', 'capm_alpha', 'capm_beta', 'idio_vol', 'yzvol', 'var95', 'es95', 'r1_0', 'r7_1', 'r14_1', 'r21_1', 'r30_1']
# predvars_lag = ['new_addresses', 'active_addresses', 'transaction_count', 'average_transaction_value', 'volumeto', 'ahcc', 'bidask', 'volshockSTD15', 'volshockSTD30', 'marketcap', 'capm_alpha', 'capm_beta', 'idio_vol', 'yzvol', 'var95', 'es95', 'r1_0', 'r7_1', 'r14_1', 'r21_1', 'r30_1']


# weekly
predvars     = ['new_addresses', 'active_addresses', 'transaction_count', 'average_transaction_value', 'volumeto', 'ahcc', 'bidask', 'volshockSTD15', 'volshockSTD30', 'marketcap', 'capm_alpha', 'capm_beta', 'idio_vol', 'yzvol', 'var95', 'es95', 'r1_0', 'r7_1', 'r14_1', 'r21_1', 'r30_1']
predvars_lag = ['new_addresses', 'active_addresses', 'transaction_count', 'average_transaction_value', 'volumeto', 'ahcc', 'bidask', 'volshockSTD15', 'volshockSTD30', 'marketcap', 'capm_alpha', 'capm_beta', 'idio_vol', 'yzvol', 'var95', 'es95', 'r1_0', 'r7_1', 'r14_1', 'r21_1', 'r30_1']


# daily
# predvars     = ['new_addresses', 'active_addresses', 'average_transaction_value', 'volumeto', 'ahcc', 'bidask', 'volshockSTD15', 'volshockSTD30', 'marketcap', 'capm_alpha', 'capm_beta', 'idio_vol', 'yzvol', 'var95', 'r1_0', 'r7_1', 'r14_1', 'r21_1', 'r30_1']
# predvars_lag = ['new_addresses', 'active_addresses', 'average_transaction_value', 'volumeto', 'ahcc', 'bidask', 'volshockSTD15', 'volshockSTD30', 'marketcap', 'capm_alpha', 'capm_beta', 'idio_vol', 'yzvol', 'var95', 'r1_0']



# Location code to numerical id
loc_encoder = LabelEncoder()
fxdata['LOCATION'] = loc_encoder.fit_transform(fxdata['LOCATION'])

# Create DATE
fxdata.drop(['DATE'], axis=1, inplace=True)
fxdata.rename(index=str, columns={'TIME_M': 'DATE'}, inplace=True)
fxdata['DATE'] = pd.to_datetime(fxdata['DATE'], dayfirst=True) # + pd.offsets.MonthEnd(0)

# fxdata['FX'] = fxdata.groupby(['LOCATION'])['FX'].shift(-1)
for col in predvars_lag:
    fxdata[col] = fxdata.groupby(['LOCATION'])[col].shift(+1)
#     if col in ['r1_0']:
#         fxdata[col] = fxdata.groupby(['LOCATION'])[col].shift(+1)
#         print(['Lagging observations in ', col])
    print(['Lagging observations in ', col])

# remove the first observation due to lagging
# fxdata['DATE_LAG'] = pd.to_datetime(fxdata['DATE_LAG'], dayfirst=True) # + pd.offsets.MonthEnd(0)
# fxdata = fxdata.loc[~(fxdata.DATE_LAG==pd.Timestamp(2021, 7,11)), :]

# remove the first observations for missing momentum characteristics
fxdata = fxdata.loc[ (fxdata.DATE>=pd.Timestamp(2018, 4, 1)), :] # (2018, 3,20) (2018, 4, 1) (2020, 1, 1)
fxdata = fxdata.loc[ (fxdata.DATE<=pd.Timestamp(2021, 7, 1)), :] # (2021, 7, 1) (2020, 1, 1)

# remove those observations which are available only for BTC
# fxdata = fxdata.loc[~(fxdata.DATE==pd.Timestamp(2018, 4,12)), :]
# fxdata = fxdata.loc[~(fxdata.DATE==pd.Timestamp(2018, 4,13)), :]
# fxdata = fxdata.loc[~(fxdata.DATE==pd.Timestamp(2018, 4,14)), :]
# fxdata = fxdata.loc[~(fxdata.DATE==pd.Timestamp(2018, 4,15)), :]
# fxdata = fxdata.loc[~(fxdata.DATE==pd.Timestamp(2018, 4,16)), :]
# fxdata = fxdata.loc[~(fxdata.DATE==pd.Timestamp(2019, 3,31)), :]
# fxdata = fxdata.loc[~(fxdata.DATE==pd.Timestamp(2019, 4, 1)), :]

#Remove obs with missing dependent variable
fxdata.dropna(subset=['FX'], inplace=True)

# Drop NaN in predictors
predvars_remove = predvars
predvars_remove = []
predvars_remove = ['capm_alpha', 'r30_1']

for col in predvars_remove:
    fxdata.dropna(subset=[col], inplace=True)
    print(['Dropping NaN in ', col])

# Check whether ETH BTC LTC belong to the sample
# fxdata.loc[fxdata['LOCATION'] == 'ETH', :]

#Extract dates
dates = fxdata['DATE'].unique()
dates = pd.DataFrame(dates, columns=['DATE'])
dates = dates.sort_values(by='DATE')
fxdata['DATE'] = fxdata['DATE'].apply(lambda x: x.toordinal())
fxdata.set_index(['LOCATION', 'DATE'], inplace=True)
ResultStore = dict()

# Add constant
fxdata['const'] = 1
predvars.append('const')

fxdata = fxdata.loc[:, ['FX']+predvars]

size_cs = fxdata.loc[:, ['FX']]

# Number of cryptocurrencies in cross-section
def Size_CS(series):
    temp = series.loc[~series.isna()]
    try:

        series.loc[~series.isna()] = len(temp)
        
    except ValueError:
        raise ValueError("CHECK DATA")

    return series

# Size of the cross-section
for col in ['FX']:
    size_cs[col] = fxdata[col].groupby(level='DATE').apply(Size_CS)

# temp = size_cs.loc[size_cs.FX == 10]
datetime.fromordinal(737065)
np.min(size_cs.FX)

# Scale variables in cross-section
def Scaler_CS(series):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    temp = series.loc[~series.isna()]
    try:
#         series.loc[~series.isna()] = np.squeeze(scaler.fit_transform(temp.values.reshape((-1, 1))))
#         series.loc[series.isna()] = 0 # Replace missing obs with CS mean
        
#         temp2 = np.squeeze(scaler.fit_transform(temp.values.reshape((-1, 1))))
#         series.loc[~series.isna()] = temp2 - temp2.sum()/len(temp2)
#         series.loc[series.isna()] = 0 # Replace missing obs with CS mean

#         temp2 = np.squeeze(temp.rank())
#         series.loc[~series.isna()] = (temp2/(1+len(temp2)) - 0.5)
#         series.loc[series.isna()] = 0 # Replace missing obs with CS mean

#         series.loc[series.isna()] = np.median(temp)
#         series = np.squeeze(series.rank())
#         series = ((series - 1)/(np.max(series) - 1) - 0.5)
        
        temp2 = np.squeeze(temp.rank())
        n_cs = len(temp2)
        series.loc[~series.isna()] = (temp2 - (1+n_cs)/2)/(n_cs/8)
        series.loc[series.isna()] = 0 # Replace missing obs with CS mean
        
    except ValueError:
        raise ValueError("CHECK DATA")

    return series

# Carry out scaling
for col in predvars:
    if col in ['const']:
        continue
    fxdata[col] = fxdata[col].groupby(level='DATE').apply(Scaler_CS)
    print(col)

fxdata['const'] = 1.0

fxdata.reset_index(inplace=True)
fxdata.dropna(axis=0, how='any', inplace=True)

# =============================================================================
#                      IPCA Estimation
# =============================================================================

regr = IPCARegressor(n_factors=1, intercept=True, iter_tol=10e-4, max_iter=1000)


# Unrestricted Case: With intercept
first_fit = True
for i in range(-1, Kmax):
    print(i+1)
    casename = 'K='+str(i+1)+'_Unrestr'
    if first_fit:
        regr.n_factors = i+1
        regr.fit(Panel=fxdata.to_numpy(), refit=False)
        first_fit = False
    else:
        regr.n_factors = i+1
        regr.fit(Panel=fxdata.to_numpy(), refit=True)

    # Prepare data for ex-post binning
    fxdata_aux = fxdata[['LOCATION', 'DATE']+[depvar]+predvars].copy()

    # Compute fitted values
    varlist = ['LOCATION', 'DATE']+[depvar]+predvars
    fxdata_aux[depvar[0]+'_hat'] = regr.predict(np.delete(fxdata_aux[varlist].to_numpy(), 2, axis=1),
                                           mean_factor=False)
    fxdata_aux[depvar[0]+'_hat_pred'] = regr.predict(np.delete(fxdata_aux[varlist].to_numpy(), 2, axis=1),
                                           mean_factor=True)

    # Store
    ResultStore[casename] = [float(regr.r2_total),
                             float(regr.r2_pred),
                             float(regr.r2_pred_cond),
                             float(regr.r2_total_x),
                             float(regr.r2_pred_x),
                             float(regr.r2_pred_cond_x),
                             float(regr.IC),
                             regr.Gamma_Est,
                             regr.Factors_Est,
                             fxdata_aux]


# Restricted Case: No intercept
for i in range(Kmax):
    print(i+1)
    casename = 'K='+str(i+1)+'_Restr'

    regr.n_factors = i+1
    regr.intercept = False
    regr.fit(Panel=fxdata.to_numpy(), refit=True)

    # Prepare data for ex-post binning
    fxdata_aux = fxdata[['LOCATION', 'DATE']+[depvar]+predvars].copy()

    # Compute fitted values
    varlist = ['LOCATION', 'DATE']+[depvar]+predvars
    fxdata_aux[depvar[0]+'_hat'] = regr.predict(np.delete(fxdata_aux[varlist].to_numpy(), 2, axis=1),
                                           mean_factor=False)
    fxdata_aux[depvar[0]+'_hat_pred'] = regr.predict(np.delete(fxdata_aux[varlist].to_numpy(), 2, axis=1),
                                           mean_factor=True)


    ResultStore[casename] = [float(regr.r2_total),
                             float(regr.r2_pred),
                             float(regr.r2_pred_cond),
                             float(regr.r2_total_x),
                             float(regr.r2_pred_x),
                             float(regr.r2_pred_cond_x),
                             float(regr.IC),
                             regr.Gamma_Est,
                             regr.Factors_Est,
                             fxdata_aux]




# =============================================================================
#                  Write e result cases file
# =============================================================================

with open(results_path+'IPCAResults.pkl', 'wb') as handle:
    ExistResults = pickle.dump(ResultStore,handle, protocol=2)


# =============================================================================
#                  Construct R2 table / Other Outputs
# =============================================================================
R2_Results = pd.DataFrame(columns=['nfactors', 'intercept', 'r2_tot',
                                    'r2_pred', 'r2_pred_cond', 'r2_tot_x', 'r2_pred_x', 'r2_pred_cond_x', 'IC'])
for case in ResultStore.keys():
    print(case)
    temp = ResultStore[case]
    if 'Unrestr' in case:
        intercept_bool = 'yes'
    else:
        intercept_bool = 'no'
    k = int(case[2:3])

    R2_Results = R2_Results.append({'nfactors': k,
                                    'intercept': intercept_bool,
                                    'r2_tot': temp[0],
                                    'r2_pred': temp[1],
                                    'r2_pred_cond': temp[2],
                                    'r2_tot_x': temp[3],
                                    'r2_pred_x': temp[4],
                                    'r2_pred_cond_x': temp[5],
                                    'IC': temp[6]}, ignore_index=True)

    Factors = ResultStore[case][8]
    columns = ['F'+str(i+1) for i in range(k)]
    if 'Unrestr' in case:
        columns += ['Intercept']
    Factors = pd.DataFrame(data=Factors.T,
                           columns=columns,
                           index=np.squeeze(dates.values))
    Factors.to_csv(results_path+'FACTORS/FactorEst_'+case+'.csv')

    Gamma = ResultStore[case][7]
    Gamma = pd.DataFrame(data=Gamma,
                         columns=columns,
                         index=predvars)
    Gamma.to_csv(results_path+'GAMMA/GammaEst_'+case+'.csv')

    fxdata_hat = ResultStore[case][9].copy()
    fxdata_hat['LOCATION'] = loc_encoder.inverse_transform(fxdata_hat['LOCATION'])
    fxdata_hat['DATE'] = fxdata_hat['DATE'].apply(lambda x: datetime.fromordinal(x))
    fxdata_hat.to_csv(results_path+'FittedValPanel_'+case+'.csv')

R2_Results.to_csv(results_path+'R2_Summary.csv')
