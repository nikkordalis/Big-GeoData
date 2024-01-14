#### DEFINITIONS FOR LAB 04.1

def fix_time_SPOT(t):
    import datetime
    t_fix = datetime.datetime(int(t[:4]), int(t[4:6]), int(t[6:]))
    return t_fix

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Takes input of series, window_size, order, deriv, and rate
    DEFAULT: savitzky_golay(values, 21, 1, deriv=0, rate=1)
    window_size must be ODD
    """
    from math import factorial
    import numpy as np
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def chen_proc(orig_input):
    import itertools
    import numpy as np, pandas as pd
    """Following Chen et al. 2004"""
    #Remove impossible values, linear fit
    orig_input[orig_input.diff() > 0.4] = np.nan
    
    #Remove remaining missed clouds/water
    orig_input[orig_input < 0] = np.nan
    
    #Remove series with too few values
    if np.nansum(np.isnan(orig_input)) > 300: 
        return False, False
        
    #Mask out non-vegetated areas (de Jong et al.)
    mns = [] 
    for yr in orig_input.groupby(orig_input.index.year):
        yrdata = yr[1]
        mns.append(np.nanmean(yrdata.values))
    
    if np.nanmean(mns) < 0.1: 
        return False, False
    
    #STEP 1 - Do the interpolation    
    else:
        data = orig_input.interpolate('linear').bfill().ffill()
        N_0 = data.copy()
        
        #STEP 2 - Fit SG Filter, using best fitting parameters
        arr = np.empty((4,3))
        for m,d in list(itertools.product([4,5,6,7],[2,3,4])):
            data_smoothed = pd.Series(savitzky_golay(data.values, 2*m + 1, d, deriv=0, rate=1), index=data.index)
            err = np.nansum(data_smoothed - data)**2
            arr[m-4,d-2] = err
        m,d = np.where(arr == np.nanmin(arr))
        m,d = m[0]+4, d[0]+2
        data_smoothed = pd.Series(savitzky_golay(data.values, 2*m + 1, d, deriv=0, rate=1), index=data.index)
        diff = N_0 - data_smoothed
        
        #STEP 3 - Create weights array based on difference from curve
        max_dif = np.nanmax(np.abs(diff.values))
        weights = np.zeros(np.array(data_smoothed.values).shape)
        weights[data_smoothed >= N_0] = 1
        weights[data_smoothed < N_0] = 1 - (np.abs(diff.values)/max_dif)[data_smoothed < N_0]
        
        #STEP 4 - Replace values that were smoothed downwards with original values
        data_smoothed[N_0 >= data_smoothed] = N_0[N_0 >= data_smoothed]
        data_smoothed[N_0 < data_smoothed] = data_smoothed[N_0 < data_smoothed]
        
        #STEP 5 - Resmooth with different parameters
        data_fixed = pd.Series(savitzky_golay(data_smoothed.values, 9, 6, deriv=0, rate=1), index=data.index)
        
        #STEP 6 - Calculate the fitting effect 
        Fe_0 = np.nansum(np.abs(data_fixed.values - N_0.values) * weights)            
        
        data = data_fixed.copy()
        
        count = 0
        #while Fe_0 >= Fe and Fe <= Fe_1:
        #while Fe_0 >= Fe:
        while count < 5:
            #STEP 4 - Replace values that were smoothed downwards with original values
            data[N_0 >= data] = N_0[N_0 >= data]
            data[N_0 < data] = data[N_0 < data]
            
            #STEP 5 - Resmooth with different parameters
            data_fixed = pd.Series(savitzky_golay(data.values, 9, 6, deriv=0, rate=1), index=data.index)
            #data_fixed = pd.Series(savitzky_golay(data.values, 5, 2, deriv=0, rate=1), index=data.index)

            #STEP 6 - Calculate the fitting effect 
            Fe = np.nansum(np.abs(data_fixed.values - N_0.values) * weights)
            data = data_fixed.copy()
            Fe_0 = Fe.copy()
            count += 1
            #if Fe <= Fe_0:
                
                #data_fixed = data_fixed_smoothed.copy()
                #data_fixed[N_0 >= data_fixed_smoothed] = N_0[N_0 >= data_fixed_smoothed]
                #data_fixed[N_0 < data_fixed_smoothed] = data_fixed_smoothed[N_0 < data_fixed_smoothed]
            #if Fe > Fe_0:
            #    break
        return np.array(data.values), True


#### These functions were used to clip the whole-africa NDVI Dataset to the Subset we use in Lab 04
def Encode(h5file, group, name, array):
    import tables
    FILTERS = tables.Filters(complib='zlib', complevel=5)
    atom = tables.Atom.from_dtype(array.dtype)
    ds = h5file.create_carray(group, name, atom, array.shape, filters=FILTERS)
    ds[:] = array
    
def slice_data(Alldata, outfid, date, skip, gt, cs, minX, maxX, minY, maxY):
    import tables
    import numpy as np
    FILTERS = tables.Filters(complib='zlib', complevel=5)
    
    #Slice down to spatial extent desired
    cXmin = int((minX - gt[0])/gt[1])
    cXmax = int((maxX - gt[0])/gt[1])
    cYmin = int((maxY - gt[3])/gt[5])#Reverse this because of top-down indexing
    cYmax = int((minY - gt[3])/gt[5])
    if skip != 0:
        slice = Alldata[cYmin:cYmax:skip, cXmin:cXmax:skip,:].copy() #Copy it to it's own memory block
    elif skip == 0:
        slice = Alldata[cYmin:cYmax, cXmin:cXmax,:].copy() #Copy it to it's own memory block
    if skip != 0:
        gt_slice = [minX, gt[1]*skip, 0, maxY, 0, gt[5]*skip]
    else:
        gt_slice = [minX, gt[1], 0, maxY, 0, gt[5]]
        
    h5file = tables.open_file(outfid, mode='w', title='NDVI', filters=FILTERS)
    Encode(h5file, '/', 'ndvi', np.array(slice))
    Encode(h5file, '/', 'date', date)
    h5file.set_node_attr('/', 'cloud', -3)
    h5file.set_node_attr('/', 'ice', -2)
    h5file.set_node_attr('/', 'shadow', -1)
    h5file.set_node_attr('/', 'ice_shadow', -2.5)
    h5file.set_node_attr('/', 'minX', minX)
    h5file.set_node_attr('/', 'maxX', maxX)
    h5file.set_node_attr('/', 'minY', minY)
    h5file.set_node_attr('/', 'maxY', maxY)
    h5file.set_node_attr('/', 'cellsize', gt[1])
    h5file.set_node_attr('/', 'coordinate_system', cs)
    h5file.close()  
    del h5file  
    
#### These functions can be used to multiprocess data
def multi_proc_ts(input_dataset, base_pkl, output_dataset, num_proc=5):
    """
    
    Input Dataset is the NDVI data, base_pkl is where to store the temporary files.
    Output Dataset is where to save the final product, and num_proc defines the number of cores to use.
    
    """
    
    import time, itertools, pickle, h5py, tables, os, shutil
    import numpy as np, pandas as pd
    from multiprocessing import Pool
    
    def PKL(data, fid):
        """
        Cleanly dumps data into fid
        """
        out = open(fid, 'wb')
        pickle.dump(data, out)
        out.close()
        
    def process_time_series(i):
        time0 = time.time() #Get the time of the start of each run
        x, y = i
        ts = pd.Series(data[x,y,:], index=date)
        proc_ts, flag = chen_proc(ts) #Do the time series fixing
        if flag:
            PKL(proc_ts, base_pkl + '{}.pickle'.format(str(x) + '_' + str(y))) #Save each individual data series
            del proc_ts
        print(i, '/', numel, time.time() - time0)
    
    #Load input data
    ds = h5py.File(input_dataset) 
    data = ds['ndvi']
    date = ds['date'][:]
    date = list(map(lambda x: fix_time_SPOT(x), date))
    
    #Generate a list of indicies -- a set of each possible x/y pair
    pairlist = list(itertools.product(range(data.shape[0]), range(data.shape[1])))
    numel = data.shape[0]*data.shape[1]
    
    #Create the empty directory for temporary files
    if not os.path.exists(base_pkl):
        os.mkdir(base_pkl)
    
    #Start a multiprocessing pool to do the processing
    p = Pool(processes=num_proc)
    
    #Process each x/y pair in a separate process
    for _ in p.imap_unordered(process_time_series, pairlist):
        pass    
    outshp = data.shape
    del data
    
    #Create an empty array to hold the output -- this could also be a dask array, or an HDF file directly, depending on memory constraints
    outarr = np.empty((outshp))
    for i in pairlist:
        try:
            #Load each of the temporary files and save them to the output array
            x, y = i
            proc_ts = pickle.load(open(base_pkl + '{}.pickle'.format(str(x) + '_' + str(y)), 'rb'))
            outarr[x, y, :] = proc_ts
        except:
            pass
    
    #Save the final cleaned up data to an output file
    FILTERS = tables.Filters(complib='zlib', complevel=5)
    h5file = tables.open_file(output_dataset, mode='w', title='NDVI', filters=FILTERS)
    Encode(h5file, '/', 'ndvi', outarr)
    Encode(h5file, '/', 'date', date)
    h5file.close()  
    del h5file  
    
    shutil.rmtree(base_pkl, ignore_errors=True)
    