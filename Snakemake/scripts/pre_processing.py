from snakemake.script import snakemake
import pandas as pd
import numpy as np
import typing 
from scipy.signal import resample, savgol_filter

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
tqdm.pandas()

# read dataframes
raw_data = pd.read_pickle(snakemake.input.raw_data)
stimulations = pd.read_pickle(snakemake.input.stimulations)
spikes = pd.read_pickle(snakemake.input.spikes)

# specify window size
if not snakemake.params.use_bristol_processing:
    lower = 15
    upper = 15
else:
    lower = 30
    upper = 30
    
# align by minimum
def ts_to_idx_col_min(column, data:pd.Series, lower, upper):
    # index array holds the index of the spike and it should be the minimum 
    index_array = data.index.get_indexer(column, method='nearest')
    for i in range(len(index_array)):
        bounds = [index_array[i]-lower, index_array[i]+upper]  
        bounds_arange = np.arange(index_array[i]-lower, index_array[i]+upper)  
        min_index = bounds_arange[data.iloc[bounds[0] :bounds[1]].argmin()]  # 
        # check if index array value aligns with min_index, if not replace it by min index to
        # ensure alignment of all spikes
        if not (index_array[i] ==  (min_index)):
            index_array[i] = min_index 
    return index_array

# align by (max - 2)
def ts_to_idx_col_max(column, data:pd.Series, lower, upper):
    # index array holds the index of the spike and it should be the minimum 
    index_array = data.index.get_indexer(column, method='nearest')
    #print(index_array)
    for i in range(len(index_array)):
        bounds = [index_array[i]-lower, index_array[i]+upper]  
        bounds_arange = np.arange(index_array[i]-lower, index_array[i]+upper)  
        if not snakemake.params.use_bristol_processing:
            mid_index = bounds_arange[data.iloc[bounds[0] :bounds[1]].argmax()]-2 
        else:
            mid_index = bounds_arange[data.iloc[bounds[0] :bounds[1]].argmin()]-2 
             # two points before max as alignment 
        # check if index array value aligns with min_index, if not replace it by min index to
        # ensure alignment of all spikes
        if not (index_array[i] ==  (mid_index)):
            index_array[i] = mid_index 
    return index_array


# align by negative peak of first derivative  
def ts_to_idx_col_fd_min(column, data:pd.Series, lower, upper):
    # index array holds the index of the spike and it should be the minimum 
    index_array = data.index.get_indexer(column, method='nearest')
    #print(index_array)
    for i in range(len(index_array)):
        bounds = [index_array[i]-lower, index_array[i]+upper]  
        bounds_arange = np.arange(index_array[i]-lower, index_array[i]+upper)  
        if not snakemake.params.use_bristol_processing:
            data_piece = data.iloc[bounds[0] :bounds[1]]
            upsampling_factor = 2
            data_piece_upsampled = pd.Series(resample(data_piece, upsampling_factor * len(data_piece)))
            fd_data_piece = pandas_gradient(data_piece_upsampled).iloc[1:]
            fd_min = fd_data_piece[10:30].argmin() + 10 + 1
            mid_index_temp = int(round(fd_min/upsampling_factor,0))
            mid_index = bounds_arange[mid_index_temp]
        else:
            data_piece = data.iloc[bounds[0] :bounds[1]]
            fd_data_piece = pandas_gradient(data_piece).reset_index(drop=True).iloc[1:]
            fd_min = fd_data_piece[20:40].argmin() +20 + 1
            mid_index = bounds_arange[fd_min]
        # check if index array value aligns with min_index, if not replace it by min index to
        # ensure alignment of all spikes
        if not (index_array[i] ==  (mid_index)):
            index_array[i] = mid_index 
    return index_array

# helper function to get bounds of data slice
def bounds(row, idx_val:typing.Union[str,int]=0, lower:int=0,upper:int=0):
    v=row[idx_val]
    return [v+lower, v+upper]

# compute gradient
def pandas_gradient(series):
    #return series.diff()
    return pd.Series(np.gradient(series.values), index= series.index)

# compute zerocrossing
def zerocrossings(series: pd.Series) -> pd.Series:
    signs = np.sign(series.values)
    return pd.Series((signs[i] != signs[i - 1] for i in range(1,len(series))),index=series.index[1:])

fig, ax = plt.subplots()

# compute first and second derivative of signal
def calculate_fd_sd(row, data:pd.Series, idx_window_start_iloc=0, idx_window_end_iloc=1):
    start = row[idx_window_start_iloc]
    end = row[idx_window_end_iloc]
    raw = data.iloc[start:end]
    if not snakemake.params.use_bristol_processing:
        raw_upsampled = pd.Series(resample(raw, 60))
    else:
        raw_upsampled = pd.Series(raw).reset_index(drop=True)
    fd = pandas_gradient(raw_upsampled)
    fd_zero = zerocrossings(fd)
    sd = pandas_gradient(fd)
    return [fd.iloc[1:].to_list(), sd.iloc[2:].to_list(), fd_zero.iloc[1:].to_list(), raw.values]

# create df with index of raw signal 
ap_window_iloc = spikes[["spike_ts"]]\
    .progress_apply(ts_to_idx_col_fd_min, args=(raw_data,),lower=lower, upper=upper,  axis=0)\
        .progress_apply(bounds, axis=1, result_type='expand',lower=-lower, upper=upper )\
        .rename(columns={0:'start_iloc', 1:'end_iloc'})


# create df with derivatives 
ap_derivatives = ap_window_iloc[['start_iloc','end_iloc']]\
        .progress_apply(calculate_fd_sd, args=(raw_data,), axis=1, result_type='expand')\
        .rename(columns={0:'fd', 1:'sd', 2:'fd_crossings', 3: "raw"})

# merge df with indices and with spike times and track
ap_track_window = ap_window_iloc.merge(spikes, on='spike_idx')


# all to pickle as output files
ap_window_iloc.to_pickle(snakemake.output.ap_window_iloc)
ap_derivatives.to_pickle(snakemake.output.ap_derivatives)
ap_track_window.to_pickle(snakemake.output.ap_track_window)
