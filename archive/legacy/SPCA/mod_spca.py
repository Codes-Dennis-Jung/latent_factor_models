
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm, block_diag
from scipy.sparse.linalg import svds
from scipy import stats
from scipy.stats import chi2, kurtosis, skew
from typing import Dict, Tuple, Optional, List, Union, Any, Callable
import warnings

def handle_missing_values(
    rt: np.ndarray,
    gt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Handle missing values in return and factor data by removing columns with NaN values.
    
    Parameters:
    -----------
    rt : np.ndarray
        Asset returns matrix (n x T)
    gt : np.ndarray
        Factor proxy matrix (d x T)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Clean return and factor matrices with NaN values handled
    '''
    if np.isnan(gt[0]).any() or np.isnan(gt[-1]).any():
        idxnang = ~np.isnan(gt.sum(axis=0))
        gt = np.nan_to_num(gt[:, idxnang], nan=0)
        rt = np.nan_to_num(rt[:, idxnang], nan=0)
    
    return rt, gt
'''
def f_pval2s(
    t_ret: np.ndarray, 
    T: Union[float, np.ndarray]
    ) -> np.ndarray:
    t_ret = np.asarray(t_ret, dtype=float)
    nvar = len(t_ret)
    pval2s = np.full(nvar, np.nan)
    
    # Handle T properly based on its type
    if isinstance(T, (int, float)):
        T = np.repeat(float(T), nvar)
    else:
        T = np.asarray(T, dtype=float)
        if len(T) < nvar:
            T = np.repeat(T[0], nvar)
            
    pval2s = 2 * (stats.t.cdf(-abs(t_ret), df=T-1)) 
    return pval2s
'''
def f_pval2s(t_ret, df):
    # Two-sided p-values for hypothesis test H0: mu=0 vs H1: mu≠0
    tstat = t_ret.flatten('F')  # equivalent to tstat(:)
    
    pval = np.zeros((1, len(tstat)))  # preallocate like MATLAB
    
    for i in range(len(tstat)):  # MATLAB 1:size(tstat,1) -> Python range(len(tstat))
        pval[0, i] = 2*((stats.t.cdf(-abs(tstat[i]), df-1)))
    return pval
def compute_newey_west_lag(
    n: int, 
    method: str = 'stock-watson'
    ) -> int:
    '''
    Compute the optimal lag length for Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors.
    The function implements three different methods to determine the optimal lag length based on sample size.
    
    Parameters:
    -----------
    n : int
        Sample size (number of observations). Must be a positive integer.
    method : str, optional
        Method to compute lag length. Options are:
        - 'stock-watson': Stock and Watson (2007) method
          Formula: floor(4*(n/100)^(2/9))
          Recommended for general time series applications
        - 'greene': Greene (2008) method
          Formula: floor(n^(1/3))
          More conservative approach, generally produces longer lags
        - 'newey-west': Original Newey-West (1994) method
          Formula: floor(4*(n/100)^(2/9))
          Classic approach, identical to Stock-Watson in implementation
        Default is 'stock-watson'
    
    Returns:
    --------
    int
        Recommended lag length. Will be at least 1 and at most (n-1).
    
    Raises:
    -------
    ValueError
        - If n is not a positive integer
        - If method is not one of the allowed options
        
    Examples:
    --------
    >>> compute_newey_west_lag(100, 'stock-watson')
    4
    >>> compute_newey_west_lag(100, 'greene')
    4
    
    References:
    ----------
    - Stock, J. H., & Watson, M. W. (2007). Introduction to Econometrics.
    - Greene, W. H. (2008). Econometric Analysis.
    - Newey, W. K., & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation.
    
    Notes:
    ------
    The function automatically ensures the returned lag length is between 1 and (n-1),
    regardless of the method used. This prevents invalid lag lengths in small samples.
    '''
    if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Sample size 'n' must be a positive integer")
        
    if method.lower() == 'stock-watson' or method.lower() == 'newey-west':
        # Stock-Watson (2007) and Newey-West (1994) method
        lag = int(np.floor(4 * (n/100)**(2/9)))
    
    elif method.lower() == 'greene':
        # Greene (2008) method
        lag = int(np.floor(n**(1/3)))
    
    else:
        raise ValueError("Method must be one of: 'stock-watson', 'greene', 'newey-west'")
    
    # Ensure lag is at least 1 and less than sample size
    lag = max(1, min(lag, n-1))
    
    return lag
def onatski_test(
    eigenvalues: NDArray[np.float64],
    kmax: int = 10,
    alpha: float = 0.05
) -> Tuple[int, NDArray[np.float64]]:
    """
    Implements Onatski's (2009) test for determining the number of factors
    given pre-computed eigenvalues.
    
    Parameters:
    -----------
    eigenvalues : NDArray[np.float64]
        Array of eigenvalues in descending order
    kmax : int, optional (default=10)
        Maximum number of factors to consider
    alpha : float, optional (default=0.05)
        Significance level between 0 and 1
        
    Returns:
    --------
    Tuple[int, NDArray[np.float64]]
        - k: Estimated number of factors
        - critical_values: Array of critical values for each k
        
    Raises:
    -------
    ValueError
        If kmax is greater than or equal to len(eigenvalues) - 3
        If alpha is not between 0 and 1
        If eigenvalues array is too short (needs at least 4 values)
    """
    # Input validation
    if len(eigenvalues) < 4:
        raise ValueError("Eigenvalues array must contain at least 4 values")
    
    if kmax >= len(eigenvalues) - 3:
        raise ValueError(f"kmax must be less than len(eigenvalues) - 3. Got kmax={kmax}")
        
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1. Got alpha={alpha}")

    # Ensure eigenvalues are sorted in descending order
    eigenvalues_sorted: NDArray[np.float64] = np.sort(eigenvalues)[::-1]
    
    # Initialize variables
    k: int = 0
    critical_values: NDArray[np.float64] = np.zeros(kmax, dtype=np.float64)
    
    # Implement Onatski's iterative procedure
    for j in range(kmax):
        # Compute j-th order difference
        diff: NDArray[np.float64] = eigenvalues_sorted[j:j+3] - eigenvalues_sorted[j+1:j+4]
        
        # Compute median of differences
        median_diff: float = float(np.median(diff))
        
        # Compute variance estimator
        variance: float = float(np.sum((diff - median_diff)**2) / (len(diff) - 1))
        
        # Compute critical value
        critical_value: float = median_diff + stats.norm.ppf(1 - alpha) * np.sqrt(variance)
        critical_values[j] = critical_value
        
        # Check if eigenvalue difference exceeds critical value
        if eigenvalues_sorted[j] - eigenvalues_sorted[j+1] > critical_value:
            k = j + 1
        else:
            break
    
    return k, critical_values

def create_kfold_indices(
    T: int, 
    K: int, 
    random_state: Optional[int] = None
    ) -> np.ndarray:
    """
    Create K-fold indices matching MATLAB's crossvalind('Kfold', T, K).
    MATLAB crossvalind does random partitioning by default.
    
    Parameters:
    -----------
    T : int
        Number of time periods
    K : int
        Number of folds
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        Array of fold indices (1-based, matching MATLAB)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize indices array (1-based to match MATLAB)
    indices = np.zeros(T, dtype=int)
    
    # Shuffle the time indices first (this matches MATLAB's random partitioning)
    shuffled_indices = np.random.permutation(T)
    
    # Assign fold numbers sequentially to shuffled indices
    for i in range(T):
        indices[shuffled_indices[i]] = (i % K) + 1  # 1-based fold numbers
        
    return indices

def calculate_combined_score(
    MaxSR: float, 
    RMS: float, 
    sigmae: float, 
    TSR2: float,
    prev_MaxSR: float = None, 
    prev_RMS: float = None,
    prev_sigmae: float = None,
    prev_TSR2: float = None,
    weights: Dict[str, float] = None
    ) -> float:
    """
    Calculate combined score using improvement ratios for SPCA metrics.
    
    Parameters:
    -----------
    MaxSR : float
        Current Maximum Sharpe ratio
    RMS : float
        Current Root mean square pricing error
    sigmae : float
        Current Idiosyncratic variance ratio
    TSR2 : float
        Current Time series R-squared
    prev_MaxSR : float, optional
        Previous Maximum Sharpe ratio
    prev_RMS : float, optional
        Previous Root mean square pricing error
    prev_sigmae : float, optional
        Previous Idiosyncratic variance ratio
    prev_TSR2 : float, optional
        Previous Time series R-squared
    weights : Dict[str, float], optional
        Custom weights for different components. Default weights:
        - RMS improvement: 0.25
        - sigmae improvement: 0.25
        - SR improvement: 0.25
        - TSR2 improvement: 0.25
        
    Returns:
    --------
    float
        Combined score based on improvements (normalized between 0 and 1)
    """
    # Default weights if none provided
    if weights is None:
        weights = {
            'rms': 0.4,
            'sigmae': 0.4,
            'sr': 0.1,
            'tsr2': 0.1
        }
    
    # Calculate improvement ratios for each metric
    def calc_improvement(current: float, previous: float, higher_better: bool = True) -> float:
        """
        Calculate improvement ratio with proper handling for better/worse metrics.
        
        Parameters:
        -----------
        current : float
            Current metric value
        previous : float
            Previous metric value
        higher_better : bool
            True if higher values are better, False if lower values are better
        
        Returns:
        --------
        float
            Normalized improvement ratio
        """
        if previous is None:
            return 0.5  # Neutral score if no previous value
        
        if higher_better:
            improvement = (current - previous) / max(abs(previous), 1e-6)
        else:
            improvement = (previous - current) / max(abs(previous), 1e-6)
            
        # Normalize improvement to [0,1] using sigmoid-like function
        return 0.5 * (1 + np.tanh(improvement))
    
    # Calculate individual improvement scores
    sr_score = calc_improvement(MaxSR, prev_MaxSR, higher_better=True)
    rms_score = calc_improvement(RMS, prev_RMS, higher_better=False)
    sigmae_score = calc_improvement(sigmae, prev_sigmae, higher_better=False)
    tsr2_score = calc_improvement(TSR2, prev_TSR2, higher_better=True)
    
    # Combined score with weights
    score = (
        weights['rms'] * rms_score +
        weights['sigmae'] * sigmae_score +
        weights['sr'] * sr_score +
        weights['tsr2'] * tsr2_score
    )
    
    return max(0, min(1, score))

def calculate_oos_metrics(
    rhatbar_test: np.ndarray,
    ghatbar_test: np.ndarray,
    mimi: np.ndarray,
    b: np.ndarray,
    p: int  
    ) -> Tuple[float, float, float]:
    """
    Calculate out-of-sample performance metrics.
    
    Parameters:
    -----------
    rhatbar_test : np.ndarray
        Centered test returns
    ghatbar_test : np.ndarray
        Centered test factors
    mimi : np.ndarray
        Mimicking portfolio weights
    b : np.ndarray
        SDF loadings
    p : int
        Factor index
        
    Returns:
    --------
    Tuple[float, float, float]
        MaxSR, RMS, sigmae
    """
    # Calculate projected factors on test set
    ghat_pred = rhatbar_test.T @ mimi[:, :, p].T
    
    # Calculate RMS (Root Mean Square pricing error)
    pricing_errors = ghatbar_test.T - ghat_pred
    RMS = np.sqrt(np.mean(pricing_errors**2))
    
    # Calculate sigmae (1 - R²)
    total_var = np.sum(ghatbar_test.T**2)
    if total_var > 0:
        sigmae = np.sum(pricing_errors**2) / total_var
    else:
        sigmae = 1.0
        
    # Calculate MaxSR using test data
    test_returns = rhatbar_test + np.mean(rhatbar_test, axis=1, keepdims=True)
    portfolio_returns = test_returns.T @ b[p, :]
    MaxSR = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-10)
    
    return MaxSR, RMS, sigmae

def kfoldcv_spca_modified(
    M: int,
    ffold: int,
    param: Dict[str, Any],
    tuningrange: np.ndarray,
    score_weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
    """
    Modified cross-validation function for SPCA with improvement-based metrics.
    
    Parameters:
    -----------
    M : int
        Number of Monte Carlo iterations
    ffold : int
        Number of folds
    param : Dict[str, Any]
        Model parameters dictionary containing:
        - rt: Asset returns matrix (N x T)
        - gt: Factor matrix (K x T)
        - usep: Number of factors to use
        - pmax: Maximum number of factors (default: 1)
    tuningrange : np.ndarray
        Array of number of assets to test
    score_weights : Dict[str, float], optional
        Custom weights for different components in scoring
        
    Returns:
    --------
    List[Dict[str, Any]]
        Results for each factor configuration with optimal parameters
    """
    # Extract parameters
    gt = param['gt']
    rt = param['rt']
    lat_fcts = param['usep']
    pmax = param.get('pmax', 1)
    T = rt.shape[1]
    d = gt.shape[0]
    
    # Initialize arrays for storing CV metrics
    tsr2 = np.zeros((ffold, M, len(tuningrange), pmax))
    MaxSR_cv = np.zeros((ffold, M, len(tuningrange), pmax))
    RMS_cv = np.zeros((ffold, M, len(tuningrange), pmax))
    sigmae_cv = np.zeros((ffold, M, len(tuningrange), pmax))
    
    # Initialize storage for best configurations
    optimal_assets = np.zeros(pmax)
    best_metrics = {p: {} for p in range(pmax)}
    
    # Perform cross-validation
    for m in range(M):
        rep_seed = m
        indices = create_kfold_indices(T, ffold, random_state=rep_seed)
        
        for i in range(ffold):
            # Create train/test splits
            test = (indices == i + 1)
            train = ~test
            
            # Split data
            rt_test = rt[:, test]
            rt_train = rt[:, train]
            gt_test = gt[:, test]
            gt_train = gt[:, train]
            
            # Center the data
            rhatbar_test = rt_test - np.mean(rt_test, axis=1, keepdims=True)
            ghatbar_test = gt_test - np.mean(gt_test, axis=1, keepdims=True)
            
            # Test each number of assets
            for jj, num_assets in enumerate(tuningrange):
                try:
                    # Run SPCA on training data
                    prm = param.copy()
                    prm['rt'] = rt_train
                    prm['gt'] = gt_train
                    prm['tuning'] = num_assets
                    prm['usep'] = np.arange(1, pmax+1)
                    
                    train_res = SPCA_cv(prm, True)
                    
                    # Process each number of factors
                    for p in range(pmax):
                        # Calculate TSR2 using test data
                        total_var = np.sum(ghatbar_test.T**2)
                        if total_var != 0:
                            tsr2[i, m, jj, p] = np.mean(1 - np.sum((ghatbar_test.T - 
                                rhatbar_test.T @ train_res['mimi'][:, :, p].T)**2)/total_var)
                        else:
                            tsr2[i, m, jj, p] = 0
                            
                        # Calculate additional metrics on test set
                        MaxSR, RMS, sigmae = calculate_oos_metrics(
                            rhatbar_test=rhatbar_test,
                            ghatbar_test=ghatbar_test,
                            mimi=train_res['mimi'],
                            b=train_res['b'],
                            p=p
                        )
                        
                        MaxSR_cv[i, m, jj, p] = MaxSR
                        RMS_cv[i, m, jj, p] = RMS
                        sigmae_cv[i, m, jj, p] = sigmae
                        
                except Exception as e:
                    print(f"Error in fold {i}, rep {m}, assets {num_assets}, p {p}: {str(e)}")
                    tsr2[i, m, jj, p] = 0
                    MaxSR_cv[i, m, jj, p] = 0
                    RMS_cv[i, m, jj, p] = np.inf
                    sigmae_cv[i, m, jj, p] = 1
    
    # Rest of the function remains the same...
    # Calculate median metrics across folds and iterations
    tsr2_median = np.median(tsr2, axis=(0, 1))
    MaxSR_median = np.median(MaxSR_cv, axis=(0, 1))
    RMS_median = np.median(RMS_cv, axis=(0, 1))
    sigmae_median = np.median(sigmae_cv, axis=(0, 1))
    
    # Calculate scores using improvement ratios
    scores = np.zeros((len(tuningrange), pmax))
    
    for p in range(pmax):
        # Track previous values for improvement calculations
        prev_metrics = {
            'MaxSR': None,
            'RMS': None,
            'sigmae': None,
            'TSR2': None
        }
        
        for jj in range(len(tuningrange)):
            # Calculate combined score using improvement ratios
            score = calculate_combined_score(
                MaxSR=MaxSR_median[jj, p],
                RMS=RMS_median[jj, p],
                sigmae=sigmae_median[jj, p],
                TSR2=tsr2_median[jj, p],
                prev_MaxSR=prev_metrics['MaxSR'],
                prev_RMS=prev_metrics['RMS'],
                prev_sigmae=prev_metrics['sigmae'],
                prev_TSR2=prev_metrics['TSR2'],
                weights=score_weights
            )
            scores[jj, p] = score
            
            # Update previous metrics
            prev_metrics['MaxSR'] = MaxSR_median[jj, p]
            prev_metrics['RMS'] = RMS_median[jj, p]
            prev_metrics['sigmae'] = sigmae_median[jj, p]
            prev_metrics['TSR2'] = tsr2_median[jj, p]
    
    # Run final SPCA with optimal parameters
    res_lst = []
    for kk, ll in zip(lat_fcts, range(len(lat_fcts))):
        # Find optimal number of assets based on maximum score
        best_idx = np.argmax(scores[:, ll])
        optimal_assets[ll] = int(tuningrange[best_idx])
        
        # Store best metrics
        best_metrics[ll] = {
            'MaxSR': MaxSR_median[best_idx, ll],
            'RMS': RMS_median[best_idx, ll],
            'sigmae': sigmae_median[best_idx, ll],
            'TSR2': tsr2_median[best_idx, ll],
            'score': scores[best_idx, ll]
        }
                
        # Run final SPCA with optimal number of assets
        prm = param.copy()
        prm['usep'] = np.array([kk])
        prm['tuning'] = np.array([optimal_assets[ll]])
        res = SPCA_cv(prm, True)
        
        # Store comprehensive results
        res['optimal_assets'] = optimal_assets[ll-1]
        res['cv_metrics'] = {
            'scores': scores[:, ll-1],
            'MaxSR_progression': MaxSR_median[:, ll],
            'RMS_progression': RMS_median[:, ll],
            'sigmae_progression': sigmae_median[:, ll],
            'TSR2_progression': tsr2_median[:, ll],
            'tuning_range': tuningrange,
            'best_metrics': best_metrics[ll],
            'cv_summary': {
                'MaxSR_cv': MaxSR_cv[:, :, :, ll],
                'RMS_cv': RMS_cv[:, :, :, ll],
                'sigmae_cv': sigmae_cv[:, :, :, ll],
                'tsr2_cv': tsr2[:, :, :, ll]
            }
        }
        
        res_lst.append(res)
    
    return res_lst

def kfoldcv_spca_modified_2(
    M: int,
    ffold: int,
    param: Dict[str, Any],
    tuningrange: np.ndarray,
    score_weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
    """
    Optimized version of cross-validation function for SPCA with improvement-based metrics.
    """
    # Extract parameters once
    gt = param['gt']
    rt = param['rt']
    lat_fcts = param['usep']
    pmax = param.get('pmax', 1)
    T = rt.shape[1]
    d = gt.shape[0]
    
    # Pre-allocate arrays using np.zeros - more efficient than initializing empty
    tsr2 = np.zeros((ffold, M, len(tuningrange), pmax))
    MaxSR_cv = np.zeros((ffold, M, len(tuningrange), pmax))
    RMS_cv = np.zeros((ffold, M, len(tuningrange), pmax))
    sigmae_cv = np.zeros((ffold, M, len(tuningrange), pmax))
    
    # Pre-compute train/test splits for all iterations
    all_indices = [create_kfold_indices(T, ffold, random_state=m) for m in range(M)]
    
    # Vectorized data splitting
    def get_train_test_data(indices, test_mask):
        train_mask = ~test_mask
        rt_test = rt[:, test_mask]
        rt_train = rt[:, train_mask]
        gt_test = gt[:, test_mask]
        gt_train = gt[:, train_mask]
        
        # Vectorized centering
        rhatbar_test = rt_test - np.mean(rt_test, axis=1, keepdims=True)
        ghatbar_test = gt_test - np.mean(gt_test, axis=1, keepdims=True)
        
        return rt_train, gt_train, rhatbar_test, ghatbar_test

    # Vectorized OOS metrics calculation
    def calculate_metrics_vectorized(rhatbar_test, ghatbar_test, mimi, b, p):
        # Calculate projected factors
        ghat_pred = rhatbar_test.T @ mimi[:, :, p].T
        pricing_errors = ghatbar_test.T - ghat_pred
        
        # Calculate RMS
        RMS = np.sqrt(np.mean(pricing_errors**2))
        
        # Calculate sigmae
        total_var = np.sum(ghatbar_test.T**2)
        sigmae = np.sum(pricing_errors**2) / total_var if total_var > 0 else 1.0
        
        # Calculate MaxSR
        test_returns = rhatbar_test + np.mean(rhatbar_test, axis=1, keepdims=True)
        portfolio_returns = test_returns.T @ b[p, :]
        MaxSR = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-10)
        
        return MaxSR, RMS, sigmae
    
    # Main cross-validation loop with optimized processing
    for m in range(M):
        indices = all_indices[m]
        
        for i in range(ffold):
            test_mask = (indices == i + 1)
            rt_train, gt_train, rhatbar_test, ghatbar_test = get_train_test_data(indices, test_mask)
            
            # Process each number of assets in parallel using numpy operations
            for jj, num_assets in enumerate(tuningrange):
                try:
                    # Prepare parameters for SPCA
                    prm = param.copy()
                    prm.update({
                        'rt': rt_train,
                        'gt': gt_train,
                        'tuning': num_assets,
                        'usep': np.arange(1, pmax+1)
                    })
                    
                    # Run SPCA
                    train_res = SPCA_cv(prm, True)
                    
                    # Calculate metrics for each factor using vectorized operations
                    for p in range(pmax):
                        MaxSR, RMS, sigmae = calculate_metrics_vectorized(
                            rhatbar_test, ghatbar_test, train_res['mimi'], train_res['b'], p
                        )
                        
                        # Store results
                        MaxSR_cv[i, m, jj, p] = MaxSR
                        RMS_cv[i, m, jj, p] = RMS
                        sigmae_cv[i, m, jj, p] = sigmae
                        
                        # Calculate TSR2
                        total_var = np.sum(ghatbar_test.T**2)
                        if total_var > 0:
                            tsr2[i, m, jj, p] = 1 - np.sum((ghatbar_test.T - 
                                rhatbar_test.T @ train_res['mimi'][:, :, p].T)**2) / total_var
                        else:
                            tsr2[i, m, jj, p] = 0
                            
                except Exception as e:
                    print(f"Error in fold {i}, rep {m}, assets {num_assets}: {str(e)}")
                    tsr2[i, m, jj, p] = 0
                    MaxSR_cv[i, m, jj, p] = 0
                    RMS_cv[i, m, jj, p] = np.inf
                    sigmae_cv[i, m, jj, p] = 1
    
    # Calculate median metrics efficiently using numpy operations
    tsr2_median = np.median(tsr2, axis=(0, 1))
    MaxSR_median = np.median(MaxSR_cv, axis=(0, 1))
    RMS_median = np.median(RMS_cv, axis=(0, 1))
    sigmae_median = np.median(sigmae_cv, axis=(0, 1))
    
    # Initialize results storage
    optimal_assets = np.zeros(pmax)
    best_metrics = {p: {} for p in range(pmax)}
    res_lst = []
    
    # Process results for each factor configuration
    for kk, ll in zip(lat_fcts, range(len(lat_fcts))):
        scores = np.zeros(len(tuningrange))
        prev_metrics = {'MaxSR': None, 'RMS': None, 'sigmae': None, 'TSR2': None}
        
        # Calculate scores using vectorized operations
        for jj in range(len(tuningrange)):
            scores[jj] = calculate_combined_score(
                MaxSR=MaxSR_median[jj, ll],
                RMS=RMS_median[jj, ll],
                sigmae=sigmae_median[jj, ll],
                TSR2=tsr2_median[jj, ll],
                prev_MaxSR=prev_metrics['MaxSR'],
                prev_RMS=prev_metrics['RMS'],
                prev_sigmae=prev_metrics['sigmae'],
                prev_TSR2=prev_metrics['TSR2'],
                weights=score_weights
            )
            
            # Update previous metrics
            prev_metrics.update({
                'MaxSR': MaxSR_median[jj, ll],
                'RMS': RMS_median[jj, ll],
                'sigmae': sigmae_median[jj, ll],
                'TSR2': tsr2_median[jj, ll]
            })
        
        # Find optimal parameters and run final SPCA
        best_idx = np.argmax(scores)
        optimal_assets[ll] = int(tuningrange[best_idx])
        
        # Store best metrics
        best_metrics[ll] = {
            'MaxSR': MaxSR_median[best_idx, ll],
            'RMS': RMS_median[best_idx, ll],
            'sigmae': sigmae_median[best_idx, ll],
            'TSR2': tsr2_median[best_idx, ll],
            'score': scores[best_idx]
        }
        
        # Run final SPCA with optimal parameters
        prm = param.copy()
        prm['usep'] = np.array([kk])
        prm['tuning'] = np.array([optimal_assets[ll]])
        res = SPCA_cv(prm, True)
        
        # Store comprehensive results
        res.update({
            'optimal_assets': optimal_assets[ll],
            'cv_metrics': {
                'scores': scores,
                'MaxSR_progression': MaxSR_median[:, ll],
                'RMS_progression': RMS_median[:, ll],
                'sigmae_progression': sigmae_median[:, ll],
                'TSR2_progression': tsr2_median[:, ll],
                'tuning_range': tuningrange,
                'best_metrics': best_metrics[ll],
                'cv_summary': {
                    'MaxSR_cv': MaxSR_cv[:, :, :, ll],
                    'RMS_cv': RMS_cv[:, :, :, ll],
                    'sigmae_cv': sigmae_cv[:, :, :, ll],
                    'tsr2_cv': tsr2[:, :, :, ll]
                }
            }
        })
        
        res_lst.append(res)
    
    return res_lst

def kfoldcv_tsr2(
    M: int, 
    K: int,
    params: Dict[str, Any],
    tuningrange: Union[np.ndarray, list],
    ) -> Dict[str, Any]:   
    # Convert tuningrange to numpy array if it's a list
    if isinstance(tuningrange, list):
        tuningrange = np.array(tuningrange)
    
    # Extract parameters
    lat_fcts = params['usep']
    if 'pmax' not in params:
        params['pmax'] = 1

    gt = params['gt']
    rt = params['rt']
    pmax = params['pmax']
    
    T = rt.shape[1]
    tsr2 = np.zeros((K, M, len(tuningrange), pmax))
    
    # Repeat K-fold CV M times
    for m in range(M):
        # Generate splits for this repetition
        rep_seed = m
        indices = create_kfold_indices(T, K, random_state=rep_seed)
        
        for i in range(K):
            # Create train/test masks (note: indices are 1-based)
            test = (indices == i + 1)
            train = ~test
            
            # Split data into train and test sets
            rt_test = rt[:, test]
            rt_train = rt[:, train]
            gt_test = gt[:, test]
            gt_train = gt[:, train]
            
            # Center the test data
            rhatbar = rt_test - np.mean(rt_test, axis=1, keepdims=True)
            ghatbar = gt_test - np.mean(gt_test, axis=1, keepdims=True)
            
            # Try different tuning parameters
            for j, w in enumerate(tuningrange):
                prm = params.copy()
                prm['rt'] = rt_train
                prm['gt'] = gt_train
                prm['tuning'] = w
                prm['usep'] = np.arange(1, pmax+1)
                res = SPCA_cv(prm, False)
                mimi = res['mimi']
                
                # Calculate out-of-sample R² for each number of factors
                total_var  = np.sum((ghatbar.T)**2)
                for p in range(pmax):
                    if total_var != 0: # Avoid division by zero by zero # axis = 0, keepdims = True)
                        tsr2[i, m, j, p] = np.mean(1 - np.sum((ghatbar.T - rhatbar.T @ mimi[:, :, p].T)**2, axis = 0, keepdims = True)/np.sum((ghatbar.T)**2, axis = 0, keepdims = True))
                    else: 
                        tsr2[i, m, j, p] = 0
                '''        
                for p in range(pmax):
                    tsr2[i, m, j, p] = np.mean(1 - np.sum((ghatbar.T -\
                        rhatbar.T @ mimi[:, :, p].T)**2)/np.sum((ghatbar.T)**2))
                '''
                    
    # Process results
    re: Dict[str, Any] = {}
    re['tsr2'] = np.median(tsr2, axis=(0, 1)) # mean
    
    tuning_by_fct = np.argmax(re['tsr2'], axis=0).tolist()
    tune_range = [tuningrange[x] for x in tuning_by_fct]
    tunes = [tune_range[x] for x in (lat_fcts-1)]

    # Compute final results for each factor configuration
    res_lst = []
    for kk, ll in zip(lat_fcts, tunes):
        prm = params.copy()
        prm['usep'] = kk
        prm['tuning'] = ll
        res = SPCA_cv(prm, True)
        res['Optimal Number of Assets'] = ll
        res_lst.append(res)

    return res_lst

def kfoldcv_tsr_oos(
    M: int, 
    K: int,
    params: Dict[str, Any],
    tuningrange: Union[np.ndarray, List[float]],
    split_perc: float = 0.5
) -> List[Dict[str, Any]]:
    """
    K-fold cross-validation for time series R² optimization in risk premium estimation.
    
    Parameters:
    -----------
    M : int 
        Number of times to repeat K-fold CV
    K : int
        Number of folds
    params : Dict[str, Any]
        Parameters dictionary containing:
            - gt: NDArray[np.float64] (d x T matrix)
            - rt: NDArray[np.float64] (n x T matrix) 
            - pmax: int, Maximum number of factors
            - usep: NDArray[np.int_], Factor indices
            - tuning: float, Tuning parameter
    tuningrange : Union[np.ndarray, List[float]]
        Range of w parameters to evaluate
    split_perc : float, optional
        Percentage for train/test split, defaults to 0.5
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of dictionaries containing results for each factor configuration
    """
    
    # Convert tuningrange to numpy array if it's a list
    if isinstance(tuningrange, list):
        tuningrange = np.array(tuningrange)
    
    # Extract parameters
    lat_fcts = params['usep']
    if 'pmax' not in params:
        params['pmax'] = 1

    gt = params['gt']
    rt = params['rt']
    pmax = params['pmax']
    
    # Sample split
    split_point = int(gt.shape[1] * split_perc)   
    gt_is = gt[:, :split_point]
    gt_oos = gt[:, split_point:]
    
    rt_is = rt[:, :split_point]
    rt_oos = rt[:, split_point:]
    
    print('In-sample #: {}'.format(gt_is.shape[1]))
    print('Out-of-sample #: {}'.format(gt_oos.shape[1]))
        
    T = rt_is.shape[1]
    tsr2 = np.zeros((K, M, len(tuningrange), pmax))
    
    # Repeat K-fold CV M times
    for m in range(M):
        # Generate splits for this repetition
        rep_seed = m + 815
        indices = create_kfold_indices(T, K, random_state=rep_seed)
        
        for i in range(K):
            # Create train/test masks (note: indices are 1-based)
            test = (indices == i + 1)
            train = ~test
            
            # Split data into train and test sets
            rt_test = rt_is[:, test]
            rt_train = rt_is[:, train]
            gt_test = gt_is[:, test]
            gt_train = gt_is[:, train]
            
            # Center the test data
            rhatbar = rt_test - np.mean(rt_test, axis=1, keepdims=True)
            ghatbar = gt_test - np.mean(gt_test, axis=1, keepdims=True)
            
            # Try different tuning parameters
            for j, w in enumerate(tuningrange):
                prm = params.copy()
                prm['rt'] = rt_train
                prm['gt'] = gt_train
                prm['tuning'] = w
                prm['usep'] = np.arange(1, pmax+1)
                res = SPCA_cv(prm, False)
                mimi = res['mimi']
                
                # Calculate out-of-sample R² for each number of factors
                total_var  = np.sum((ghatbar.T)**2)
                for p in range(pmax):
                    if total_var != 0: # Avoid division by zero by zero
                        tsr2[i, m, j, p] = np.mean(1 - np.sum((ghatbar.T -\
                            rhatbar.T @ mimi[:, :, p].T)**2)/np.sum((ghatbar.T)**2))
                    else: 
                        tsr2[i, m, j, p] = 0
                '''        
                for p in range(pmax):
                    tsr2[i, m, j, p] = np.mean(1 - np.sum((ghatbar.T -\
                        rhatbar.T @ mimi[:, :, p].T)**2)/np.sum((ghatbar.T)**2))
                '''
                    
    # Process results
    re: Dict[str, Any] = {}
    re['tsr2'] = np.mean(tsr2, axis=(0, 1))
    
    tuning_by_fct = np.argmax(re['tsr2'], axis=0).tolist()
    tune_range = [tuningrange[x] for x in tuning_by_fct]
    tunes = [tune_range[x] for x in (lat_fcts-1)]

    # Compute final results for each factor configuration
    res_lst = []
    for kk, ll in zip(lat_fcts, tunes):
        prm = params.copy()
        prm['rt'] = rt_oos
        prm['gt'] = gt_oos
        prm['usep'] = kk
        prm['tuning'] = ll
        res = SPCA_cv(prm, True)
        res['Optimal Number of Assets'] = ll
        res_lst.append(res)

    return res_lst

def create_valid_mask(x, y):
    """
    Create boolean filter for arrays x (n x T) and y (k x T)
    identifying positions where both have valid (non-NaN) values.
    
    Parameters:
    x : array of shape (n x T)
    y : array of shape (k x T)
    
    Returns:
    numpy.ndarray of shape (T,) with True for valid positions
    """
    x = np.where(x == 0, np.nan, x.astype(float))
    y = np.where(y == 0, np.nan, y.astype(float))
    
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Time dimensions must match. Got x: {x.shape}, y: {y.shape}")
    
    # Check for NaN in each array along first dimension
    x_valid = ~np.any(np.isnan(x), axis=0)  # T-length boolean array
    y_valid = ~np.any(np.isnan(y), axis=0)  # T-length boolean array
    
    # Combine masks - True only where both x and y are valid
    valid_mask = x_valid & y_valid
    
    return valid_mask

def SPCA_cv(
    param_spca: Dict[str, Any],
    return_variance: bool
    ) -> Dict[str, Any]:
    '''
    Performs supervised PCA estimates of risk premium with cross-validation.
    
    Parameters:
    -----------
    param_spca : Dict[str, Any]
        Dictionary containing:
        - rt: np.ndarray 
            n x T matrix of returns
        - gt: np.ndarray
            d x T matrix of factor proxies
        - usep: Union[int, np.ndarray]
            Number of factors to use (1 x J vector)
        - tuning: int
            Number of test assets to use at each step
        - q: Optional[int]
            Newey-West lag parameter, default is 1
            
    return_variance : bool
        Whether to compute and return variance estimates
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - Gammahat_nozero: d x J vector of risk premia estimates
        - b: J x N vector of SDF loading
        - mimi: d x N x J weights matrix for mimicking portfolio
        - avarhat_nozero: d x J vector of variance estimator for risk premia
        - alphahat: d x J vector of pricing errors
        - avarhat_alpha: d x J vector of variance estimator for alpha
        - Newey-West lag parameter for HAC standard errors
        Additional statistics including t-stats, p-values, eigenvalues, etc.
    '''
    
    def vec(a):
        return a.flatten('F').reshape(-1, 1)
    
    # INITIALIZATION
    param = param_spca
        
    # INPUT
    rt = param['rt']
    gt = param['gt']
    rt, gt = handle_missing_values(rt, gt)
    
    #valid_mask = create_valid_mask(rt, gt)
    
    #rt = rt[valid_mask]
    #gt = gt[valid_mask]

    usep = param['usep']
    N0 = int(param['tuning'])
    
    T = rt.shape[1]
    n = rt.shape[0]
    d = gt.shape[0]
    if isinstance(usep, (int, np.integer)):
        usep = np.array([usep])
    J = len(usep)
    pmax = np.max(usep)
    q = compute_newey_west_lag(T, method='stock-watson')
        
    # Center the data
    rtbar = rt - np.tile(np.mean(rt, axis=1, keepdims=True),(1,T))
    gtbar = gt - np.tile(np.mean(gt, axis=1, keepdims=True), (1,T))
    
    # Initialize arrays
    etahat_all = np.zeros((d, pmax))
    gammahat_all = np.zeros((pmax, 1))
    B = np.zeros((pmax, n))
    mrt = np.mean(rt, axis=1).reshape(-1,1)
    vhat_all = np.zeros((pmax, T))
    Sigma = np.cov(rt)
    #k = kurtosis(rt.T).reshape(-1,1)
    #sk = skew(rt.T).reshape(-1,1)
    
    # Initialize working copies
    rt_0 = rtbar.copy()
    gt0 = gtbar.copy()
    mrt_0 = mrt.copy()
    
    # Initialize results arrays
    Top = np.zeros((0, pmax))
    TopCorr = np.zeros((0, pmax))
    Gammahat_nozero = np.zeros((d, J))
    Gammahat_tstat = np.zeros((d, J))
    Gammahat_pval = np.zeros((d, J))
    what_full = np.zeros((d, J))
    Gammahat_se = np.zeros((d, J))
    tsr2 = np.zeros((d, J))
    W_phat = np.zeros((d, J))
    pvchiW_phat = np.zeros((d, J))
    sharR = np.zeros((d, J))
    num_assets = np.zeros((d, J))
    b = np.zeros((J, n))
    mimi = np.zeros((d, n, J))
    avarhat_nozero = np.zeros((d, J))
    alphahat = np.zeros((d, J))
    avarhat_alpha = np.zeros((d, J))
    #R2F = np.full((1, pmax), np.nan)
    #R2G = np.full((d, pmax), np.nan)
    RMS = np.zeros(J)
    sigmae = np.zeros(J)
    lambda_ = np.zeros((pmax, 1))
    lambda_norm = np.zeros((pmax, 1))
    
    # Main estimation loop
    kk = 0
    while kk < pmax:
        # Compute correlations
        COR = np.abs(np.corrcoef(np.hstack((rt_0.T, gt0.T)), rowvar=False)\
            [:-gt0.shape[0], -gt0.shape[0]:])
        # Select assets based on correlations
        L = np.max(COR, axis=1)
        bb = np.sort(L)
        if N0 < n:
            II = (L >= bb[n-N0])
        else:
            II = L > -1
            
        kk = kk + 1
        
        if kk == 1:
            Index = np.zeros((n, 1))
            num_selected = np.sum(II)
            Top = np.zeros((num_selected, pmax))
            TopCorr = np.zeros((num_selected, pmax))
        else:
            Index = np.column_stack((Index, np.zeros(n)))
            
        Index[II, kk-1] = 1
        
        ### NEW
        selected_indices = np.where(II)[0]
        n_selected = len(selected_indices)
        # Fill up to the minimum of available space and selected indices
        n_fill = min(Top.shape[0], n_selected)
        #if d > 0:
        #    Top[:n_fill, kk-1] = None
        #    TopCorr[:n_fill, kk-1] = None
        #else:
        #    Top[:n_fill, kk-1] = selected_indices[:n_fill]
        #    TopCorr[:n_fill, kk-1] = COR[II][:n_fill].flatten()
        Top[:n_fill, kk-1] = selected_indices[:n_fill]
        TopCorr[:n_fill, kk-1] = COR[II][:n_fill].flatten()
        ### NEW
        
        # Perform PCA
        U, s, Vt = svds(rt_0[II, :], k=1, which='LM',solver='arpack')

        # For covariance matrix X.T @ X / n, eigenvalues are (s^2) / n
        s_ = s[::-1]
        n_samples = rt_0[II, :].shape[0]
        lambda_norm[kk-1, 0] = (s ** 2) / n_samples
        lambda_[kk-1, 0] = s_**2
    
        # Store results and update projections
        V = Vt.T
        B[kk-1, II] = U[:, 0].T / s[0]
        gammahat_all[kk-1, 0] = U[:, 0].T @ mrt_0[II] / s[0]
        etahat_all[:, kk-1] = gt0 @ V[:, 0]
        
        # Project out explained components
        gt0 = gt0 - gt0 @ V[:, 0:1].reshape(-1,1) @ V[:, 0:1].reshape(-1,1).T
        mrt_0 = mrt_0 - rt_0 @ V[:, 0].reshape(-1,1) @ gammahat_all[kk-1, 0].reshape(-1,1)
        rt_0 = rt_0 - rt_0 @ V[:, 0:1].reshape(-1,1) @ V[:, 0:1].reshape(-1,1).T
        vhat_all[kk-1, :] = V[:, 0].T
        
        #betahat = eigvec[:, (n-phat):n]
        #vhat = betahat.T @ rt
        
    # Process results for each number of factors
    for jj in range(len(usep)):
        phat = usep[jj]
        vhat = vhat_all[0:phat, :]
        etahat = etahat_all[:, 0:phat]
        gammahat = gammahat_all[0:phat, :]
        Sigmavhat = vhat @ vhat.T / T
        what = gtbar - etahat @ vhat
        #what_full[:, jj] = what.reshape(-1,1)
        
        if return_variance:
            # Newy-West Estimation
            Pi11hat = np.zeros((d*phat, d*phat))
            Pi12hat = np.zeros((d*phat, phat))
            Pi22hat = np.zeros((phat, phat))
            Pi13hat = np.zeros((d*phat, d))
            Pi33hat = np.zeros((d, d))
            
            for t in range(T):
                Pi11hat = Pi11hat + vec(what[:, t:t+1] @ vhat[:, t:t+1].T) @ vec(what[:, t:t+1] @ vhat[:, t:t+1].T).T/T
                Pi12hat = Pi12hat + vec(what[:, t:t+1] @ vhat[:, t:t+1].T) @ vhat[:, t:t+1].T/T
                Pi22hat = Pi22hat + vhat[:, t:t+1] @ vhat[:, t:t+1].T/T

                Pi13hat = Pi13hat + vec(what[:, t:t+1] @ vhat[:, t:t+1].T) @ what[:, t:t+1].T/T
                Pi33hat = Pi33hat + what[:, t:t+1] @ what[:, t:t+1].T/T

                for s in range(1, min(t+1, q+1)):
                    Pi11hat = Pi11hat + 1/T*(1-s/(q+1))*(vec(what[:, t:t+1] @ vhat[:, t:t+1].T) @ vec(what[:, t-s:t-s+1] @ vhat[:, t-s:t-s+1].T).T + 
                                                         vec(what[:, t-s:t-s+1] @ vhat[:, t-s:t-s+1].T) @ vec(what[:, t:t+1] @ vhat[:, t:t+1].T).T)
                    Pi12hat = Pi12hat + 1/T*(1-s/(q+1))*(vec(what[:, t:t+1] @ vhat[:, t:t+1].T) @ vhat[:, t-s:t-s+1].T + 
                                                         vec(what[:, t-s:t-s+1] @ vhat[:, t-s:t-s+1].T) @ vhat[:, t:t+1].T)
                    Pi22hat = Pi22hat + 1/T*(1-s/(q+1))*(vhat[:, t:t+1] @ vhat[:, t-s:t-s+1].T + vhat[:, t-s:t-s+1] @ vhat[:, t:t+1].T)

                    Pi13hat = Pi13hat + 1/T*(1-s/(q+1))*(vec(what[:, t:t+1] @ vhat[:, t:t+1].T) @ what[:, t-s:t-s+1].T + 
                                                         vec(what[:, t-s:t-s+1] @ vhat[:, t-s:t-s+1].T) @ what[:, t:t+1].T)
                    Pi33hat = Pi33hat + 1/T*(1-s/(q+1))*(what[:, t:t+1] @ what[:, t-s:t-s+1].T + what[:, t-s:t-s+1] @ what[:, t:t+1].T)

            
            # Compute variance estimates
            avarhat_nozero[:, jj] = np.diag(np.kron(gammahat.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi11hat @ np.kron(np.linalg.inv(Sigmavhat) @ gammahat, np.eye(d))/T +
                                np.kron(gammahat.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi12hat @ etahat.T/T + 
                                (np.kron(gammahat.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi12hat @ etahat.T).T/T +
                                etahat @ Pi22hat @ etahat.T/T)
        
        # Compute risk premia and related statistics
        Gammahat_nozero[:, jj] = np.ravel(etahat @ gammahat)
        
        B_sub = B[0:phat, :]
        fhat = B_sub @ rt
        fhatbar = fhat - np.mean(fhat, axis=1, keepdims=True)
                
        mimi[:, :, jj] = (gt - np.mean(gt, axis=1).reshape(-1, 1)) @ fhatbar.T @ \
                        np.linalg.inv(fhatbar @ fhatbar.T) @ B_sub
                        
        b[jj, :] = np.mean(fhat, axis=1).T @ np.linalg.inv(fhatbar @ fhatbar.T/T) @ B_sub
        
        # Compute performance metrics
        tsr2[:, jj] = 1 - np.sum((gtbar.T - rtbar.T @ mimi[:, :, jj].T)**2)/np.sum((gtbar.T)**2)
        
        # estimate alpha = E(g) - eta*gamma
        alphahat[:, jj] = np.mean(gt, axis=1) - Gammahat_nozero[:, jj]
        if return_variance:
            avarhat_alpha[:, jj] = np.diag(np.kron(gammahat.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi11hat @ np.kron(np.linalg.inv(Sigmavhat) @ gammahat, np.eye(d))/T -
                                          np.kron(gammahat.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi13hat/T - 
                                          (np.kron(gammahat.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi13hat).T/T +
                                          Pi33hat/T)            
            if d < 2:
                W_phat[:, jj] = T * (etahat @ np.linalg.inv(np.linalg.inv(Sigmavhat) @ \
                                    Pi11hat @ np.linalg.inv(Sigmavhat)) @ etahat.T)
                pvchiW_phat[:, jj] = chi2.sf(W_phat[:, jj], phat)
            else:
                W_phat[:, jj] = np.nan
                pvchiW_phat[:, jj] = np.nan
        
        # Calculate RMS (Root Mean Square pricing error)
        RMS[jj] = np.sqrt((alphahat[:, jj].T @ alphahat[:, jj]) / n)
        # Calculate sigmae (unexplained variance ratio)
        sigmae[jj] = np.mean(what**2)
        
        # Compute t-statistics and p-values
        Gammahat_tstat[:, jj] = Gammahat_nozero[:, jj] / np.sqrt(avarhat_nozero[:, jj])
        Gammahat_pval[:, jj] = f_pval2s(Gammahat_tstat[:, jj], n-1)
        Gammahat_se[:, jj] = np.sqrt(avarhat_nozero[:, jj])
        
        # Compute performance ratios
        sharR[:, jj] = np.sqrt((b[jj, :] @ mrt)*(b[jj, :] @ Sigma @ b[jj, :].T)**(-1) * \
                              (b[jj, :] @ mrt))
        
        #Miota = np.eye(n) - np.ones((n,1)) @ np.ones((1,n)) / n
        #R2F[0, jj] = (rtbar.T @ Miota @ b[jj, :].reshape(1, -1) @ np.linalg.inv(b[jj, :].reshape(1, -1).T @ Miota @ b[jj, :].reshape(1, -1)) @ 
        #                  b[jj, :].reshape(1, -1).T @ Miota @ rtbar) / (rtbar.T @ Miota @ rtbar)
        #R2G[:, jj] = (1 / np.diag(gtbar @ gtbar.T)) * np.diag(etahat @ (vhat @ vhat.T) @ etahat.T)
        
        
        num_assets[:, jj] = np.count_nonzero(b[jj, :])
                
        #num_opt_fact = onatski_test(eigenvalues = eigenval, kmax=pmax, alpha = 0.05) #### here
        #RMS = np.sqrt((alphahat.T @ alphahat) / n) #### here
        #sigmae = np.sum(np.diag(np.cov(alphahat))) / np.sum(np.diag(np.cov(rt.T)))

    # Compile results
    res = {
        'Gammahat_nozero': Gammahat_nozero, # The estimated risk premia without zero-beta adjustment, corresponding to γ̂ᵍ in the paper (Section 2.6)
        'b': b, # The estimated SDF loadings described in Section 2.7
        'mimi': mimi, # The estimated stochastic discount factor (SDF) values described in Section 2.9
        'alphahat': alphahat, # The estimated pricing error (alpha) for tradable factors - Section 4.2.1 provides a concrete application evaluating Berkshire Hathaway's alpha
        'RMS': RMS,
        'sigmae': sigmae, 
        'vhat_all':fhat,
        'Top': Top,
        'TopCorr': TopCorr,
        'resid': what,
        #'R2F':R2F,
        #'R2G':R2G,
        'etahat': etahat_all, # Estimated factor loadings η̂ from equation (4) in the paper
        'gammahat': gammahat_all, # Estimated risk premia γ̂ for the latent factors
        'pmax': pmax, # Maximum number of factors considered
        'tstat': Gammahat_tstat, # t-statistics for the risk premia estimates
        'pval': Gammahat_pval, # Associated p-values for testing significance
        'se': Gammahat_se, # Standard errors for the risk premia estimates, based on the asymptotic distribution derived in Theorem 2
        'eigenvalues': lambda_, # Eigenvalues of the returns covariance matrix used for factor extraction
        'eigenval_norm':lambda_norm,
        'wtest': pvchiW_phat, # The corresponding p-value under chi-square distribution
        'wstat': W_phat, # The Wald statistic testing factor recovery
        'R2': tsr2, # Out-of-sample R² for evaluating the hedging portfolio performance (Section 4.1.2)
        'Sharpe Ratio': sharR, #  Sharpe ratio of the constructed portfolios (discussed in Section 4.1.3)
        '# of assets': num_assets, # Number of assets selected at each iteration (parameter q in Algorithm 5)
        'Newey-West Lags': q, # Number of lags used in computing Newey-West standard errors for inference
        #'Onatski_Opt-LatFct': num_opt_fact # optimal number of latent factors entering the SDF according to the Onatski (2010)
    }
    
    if return_variance:
        res['avarhat_nozero'] = avarhat_nozero # Asymptotic variance matrix for risk premia (Φ in Theorem 2)
        res['avarhat_alpha'] = avarhat_alpha # Asymptotic variance matrix for alphas (Φ̃ in Theorem 3)
        
    return res




'''
#### Example #### 

import pandas as pd
import os
from measurements import *
from data_store import *

# Load data
return_monhtly=load_from_HDFStore(group='Raw_data',
                                    path=os.getcwd()+r'\Data\Input',
                                    dataset='return_series_monthly',
                                    filename='Commodity_HDFStore').loc['1980-01-31':,:]

names_lst = load_yaml_const(filename = '\const.yaml', num_lst = 7)
fullnames_lst = load_yaml_const(filename = '\const.yaml', num_lst = 6)
graph_lst = load_yaml_const(filename = '\const.yaml', num_lst = 8)

def load_fct_rets(names_lst,fullnames_lst,graph_lst,return_monhtly): 
    
    lst_ = []
    for k,j in zip(names_lst,fullnames_lst):
        l = load_from_HDFStore(group = 'Factor_data', path = os.getcwd() + r'\Data\Input',
                                   dataset = str(k)+'_perf_monthly', 
                                   filename = 'Commodity_HDFStore_Returns_Load')
        x, y = equal_length(return_monhtly,l.iloc[:,-1].to_frame())
        y.columns = [j]
        lst_.append(y)
    fct_rets = pd.concat(lst_, axis = 1)
    fct_rets.columns = graph_lst
    return x, fct_rets

return_monhtly, fct_rets = load_fct_rets(names_lst,fullnames_lst,graph_lst,return_monhtly) 
#fct_rets['BMK'] = return_monhtly.mean(1,numeric_only=True).to_frame()
#rt,gt = equal_length(return_monhtly,bmk)
#bmk = return_monhtly.mean(1).to_frame()

lst_num_fct = [1,2,3,4,5]
names = fct_rets.columns.tolist()

res_spca = {}
for i,k in enumerate(names):
    print(i,k)
    # Data pre-processing
    #rt, gt = equal_length(return_monhtly.dropna(axis='columns', how='any'), #return_monhtly.fillna(0),
    #                      fct_rets.iloc[:,[i]].fillna(0))
    rt, gt = equal_length(return_monhtly,fct_rets.iloc[:,[i]].dropna())
    rt = rt.fillna(0).T.to_numpy()
    gt = gt.fillna(0).T.to_numpy()
    pmax = max(lst_num_fct)
    print(fct_rets.iloc[:,[i]].columns[0])

    # SPCA
    param_spca = {'pmax': pmax, 'rt': rt, 'gt': gt}
    param_spca['usep'] = np.array(lst_num_fct)
    param_spca['tuning'] = np.arange(5, len(return_monhtly.columns)+1, 1)
    tuningrange_SPCA = np.arange(5, len(return_monhtly.columns)+1, 1)
    SPCAres = kfoldcv_tsr2(M=2, K=3, params=param_spca, 
                           tuningrange=tuningrange_SPCA)
    res_spca[k] = SPCAres

tbl = []
for i,k in enumerate(list(res_spca.keys())):#names):
    for l in range(len(lst_num_fct)):
        val = res_spca[k][l]
        tbl.append(pd.DataFrame({'Latent Factor':np.ravel(val['pmax']),
                      'alphahat':np.ravel(val['alphahat']),
                      'tstat':abs(np.ravel(val['tstat'])),
                      'pval':np.ravel(val['pval']),
                      'se':np.ravel(val['se']),
                      'wtest':np.ravel(val['wtest']),
                      'wstat':np.ravel(val['wstat']),
                      'R2':np.ravel(val['R2'])*100,
                      'SR':np.ravel(val['Sharpe Ratio']),
                      'RP':np.ravel(val['Gammahat_nozero'])*10000,
                      '# Assets':np.ravel(val['# of assets']),
                      'log(eigenvalues)': val['eigenvalues'],
                      'OOS R2': np.median(val['OOS tsr2'][:,[l]])*100
                      },
                     index=[k]))
res = pd.concat(tbl).sort_values(by=['Latent Factor'])
res.index.name = 'Obsrv. Factor'
res = res.reset_index().set_index(['Latent Factor','Obsrv. Factor'])

# pval significance
res_pv = res.reset_index().pivot(index='Obsrv. Factor', 
                        columns='Latent Factor', 
                        values='pval')
res_pv = pd.DataFrame({
    'pval (min)': res_pv.min(axis=1),
    'Lat. Factor': res_pv.idxmin(axis=1)
})

# tstat
res_tstat = res.reset_index().pivot(index='Obsrv. Factor', 
                        columns='Latent Factor', 
                        values='tstat')#.mean(axis=0)

res_pv['tstat'] = [res_tstat.loc[idx, res_pv.loc[idx, 'Lat. Factor']] 
                           for idx in res_pv.index]

# se
res_se = res.reset_index().pivot(index='Obsrv. Factor', 
                        columns='Latent Factor', 
                        values='se')#.mean(axis=0)

res_pv['se'] = [res_se.loc[idx, res_pv.loc[idx, 'Lat. Factor']] 
                           for idx in res_pv.index]

# wtest
res_wtest = res.reset_index().pivot(index='Obsrv. Factor', 
                        columns='Latent Factor', 
                        values='wtest')#.mean(axis=0)

res_pv['wtest'] = [res_wtest.loc[idx, res_pv.loc[idx, 'Lat. Factor']] 
                           for idx in res_pv.index]

# wstat
res_wstat = res.reset_index().pivot(index='Obsrv. Factor', 
                        columns='Latent Factor', 
                        values='wstat')#.mean(axis=0)

#res_pv['wstat'] = [res_wstat.loc[idx, res_pv.loc[idx, 'Lat. Factor']] 
#                           for idx in res_pv.index]

# R2
res_r2 = res.reset_index().pivot(index='Obsrv. Factor', 
                        columns='Latent Factor', 
                        values='R2')#.mean(axis=0)

res_pv['R2'] = [res_r2.loc[idx, res_pv.loc[idx, 'Lat. Factor']] 
                           for idx in res_pv.index]

# OOS R2
res_r2_ = res.reset_index().pivot(index='Obsrv. Factor', 
                        columns='Latent Factor', 
                        values='OOS R2')#.mean(axis=0)

res_pv['OOS R2'] = [res_r2_.loc[idx, res_pv.loc[idx, 'Lat. Factor']] 
                           for idx in res_pv.index]

# RP
res_rp = res.reset_index().pivot(index='Obsrv. Factor', 
                        columns='Latent Factor', 
                        values='RP')

res_pv['RP'] = [res_rp.loc[idx, res_pv.loc[idx, 'Lat. Factor']] 
                           for idx in res_pv.index]

# SR
res_sr = res.reset_index().pivot(index='Obsrv. Factor', 
                        columns='Latent Factor', 
                        values='SR')#.mean(axis=0)

res_pv['SR'] = [res_sr.loc[idx, res_pv.loc[idx, 'Lat. Factor']] 
                           for idx in res_pv.index]

# Number of Assets
res_num = res.reset_index().pivot(index='Obsrv. Factor', 
                        columns='Latent Factor', 
                        values='# Assets')

res_pv['# Assets'] = [res_num.loc[idx, res_pv.loc[idx, 'Lat. Factor']] 
                           for idx in res_pv.index]

# alphahat
res_alphahat = res.reset_index().pivot(index='Obsrv. Factor', 
                        columns='Latent Factor', 
                        values='alphahat')#.mean(axis=0)

res_pv['alphahat'] = [res_alphahat.loc[idx, res_pv.loc[idx, 'Lat. Factor']] 
                           for idx in res_pv.index]

sing_portf = res_pv[res_pv['pval (min)'] < 0.10].sort_values(by=['R2'], 
                                                             ascending=False).round(3)

from measurements import add_stars

sing_portf['sign. p-value'] = sing_portf['pval (min)'].apply(add_stars).values
'''