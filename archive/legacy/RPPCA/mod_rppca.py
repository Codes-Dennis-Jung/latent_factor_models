from typing import Dict, Tuple, Union
import numpy as np
import numpy.typing as npt
from scipy.stats import chi2, t
import warnings
warnings.filterwarnings('ignore')

NDArrayFloat = npt.NDArray[np.float64]

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


def compute_newey_west_lag(
    n: int, 
    method: str = 'stock-watson'
    ) -> int:
    '''
    Compute the optimal lag length for Newey-West HAC standard errors.    
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

def f_pval2S(t_ret, df):
    # Two-sided p-values for hypothesis test H0: mu=0 vs H1: mu≠0
    tstat = t_ret.flatten('F')  # equivalent to tstat(:)
    
    pval = np.zeros((1, len(tstat)))  # preallocate like MATLAB
    
    for i in range(len(tstat)):  # MATLAB 1:size(tstat,1) -> Python range(len(tstat))
        pval[0, i] = 2*((t.cdf(-abs(tstat[i]), df-1)))
    return pval

def lettau_rp(K, X, F, eigval):
    """
    This function implements the RP-PCA methodology in Lettau and Pelger (2020a,b)
    It also performs the Onatski test for the number of latent factors
    Input:
    K = number of factors taken into consideration
    X = T x N matrix of portfolio excess returns
    F = T x K matrix of latent factors
    eigval = vector of ordered eigenvalues to perform Onatski algoritmh
    output:
    MaxSR = Max Sharpe ratio obtained by combining the latent factors
    weights = weights attacched to the factors to get MaxSR (This differs
    form iSR as it is normalized by variance and take into account
    correlation between factors (if factors are correlated)
    iSR = individual Sharpe ratio to each factor 
    RMS = square root of the average pricing errors of the TS regressions
    sigmae = unexplained variance implied by the model
    maxf = optimal number of factors suggested by the Onatski criterion
    """
    # Compute statistics for Lettau and Pelger (2020ab)
    T = X.shape[0]  # time obs
    N = X.shape[1]  # cs obs

    # normalize factors to have positive average
    for k in range(K):
        if np.mean(F[:, k]) < 0:
            F[:, k] = (-1) * F[:, k]

    Bmat = np.zeros((N, F.shape[1]))
    emat = np.zeros((T, N))
    alpha = np.zeros(N)
    xmat = np.column_stack([np.ones((T, 1)), F])

    for n in range(N):
        tempout = np.linalg.inv(xmat.T @ xmat) @ xmat.T @ X[:, n]
        atemp = tempout[0]
        btemp = tempout[1:]
        etemp = X[:, n] - (atemp * np.ones(T) + F @ btemp)
        alpha[n] = atemp
        Bmat[n, :] = btemp
        emat[:, n] = etemp

    # Evaluate performance
    RMS = np.sqrt((alpha.T @ alpha) / N)
    sigmae = np.sum(np.diag(np.cov(emat))) / np.sum(np.diag(np.cov(X)))
    
    # Convert F to 2D array if it's not already
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    
    sigmaF = np.cov(F, rowvar=False)  
    muF = np.mean(F, axis=0)
    
    # Handle single factor case
    if K == 1:
        sigmaF = np.array([[sigmaF]])
        muF = np.array([muF])
    
    # Matrix division in MATLAB is equivalent to solve in numpy
    weights = np.linalg.solve(sigmaF, muF.reshape(-1, 1)).flatten()
    Fcomb = F @ weights
    MaxSR = float(np.mean(Fcomb) / np.std(Fcomb))
    iSR = muF / np.sqrt(np.diag(sigmaF))

    # Onatski (2010) test remains the same
    rmax = K
    r = 0
    rmaxtemp = rmax
    iter = 0
    j = rmaxtemp + 1

    while (rmax - r) > 0:
        yo = eigval[j-1:j+4]
        xo = np.array([(j-1)**(2/3), j**(2/3), (j+1)**(2/3), (j+2)**(2/3), (j+3)**(2/3)])
        xo = np.column_stack([np.ones(5), xo])
        beta = np.linalg.inv(xo.T @ xo) @ xo.T @ yo
        th = 2 * abs(beta[1])
        i = j - 1
        diff = eigval[0:i] - eigval[1:i+1]
        temp = np.where(diff - th >= 0)[0]
        r = temp.max() + 1 if len(temp) > 0 else 0

        if len(diff) == r:
            rmax = rmaxtemp
        
        iter += 1
        j = r + 1
        rmaxtemp = rmaxtemp - 1

    maxf = rmax

    return MaxSR, weights, iSR, RMS, sigmae, maxf

def my_orth(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Direct translation of my_orth.m"""
    T = F.shape[0]
    K = F.shape[1]
    H = np.eye(K)
    Z = np.zeros((T, K))
    c = np.ones((T, 1))
    Z[:, 0] = F[:, 0]
    
    for k in range(K-1):
        X = np.hstack([c, F[:, :k+1]])
        coeff = np.linalg.inv(X.T @ X) @ X.T @ F[:, k+1]
        Xfitted = X[:, 1:] @ coeff[1:]
        Z[:, k+1] = F[:, k+1] - Xfitted
        H[k+1, :k+1] = -coeff[1:].T
    
    # Normalize factors in Z to have positive average
    for k in range(Z.shape[1]):
        if (np.mean(Z[:, k]) < 0):
            Z[:, k] = (-1) * Z[:, k]
            H[k, :] = (-1) * H[k, :]
    
    H = H.T
    
    return Z, H

def vec(x: NDArrayFloat) -> NDArrayFloat:
    """
    Helper function to replicate MATLAB's vec operation

    Args:
        x: Input array to be vectorized

    Returns:
        Flattened array in column-major (Fortran) order
    """
    return x.flatten('F')

def rppca_rfs(
    rt: NDArrayFloat,
    gt: NDArrayFloat,
    pmax: int,
    indc: int,
    j: int,
    w: float
) -> Dict[str, Union[NDArrayFloat, float]]:
    """
    Implementation of RPPCA with all dimension fixes
    
    Args:
        rt: n by T matrix of returns
        gt: d by T matrix of factor proxies
        pmax: upperbound of the number of latent factors
        q: number of lags used in Newey-West standard errors
        indc: 0/1 indicator to include a constant in giglio and xiu
        j: number that indicates the observable factor g_t
        w: weight to be applied to the means in lettau pelger
    """
    rt, gt = handle_missing_values(rt, gt)
    Ir = (~np.isnan(rt)).astype(float)
    Ig = (~np.isnan(gt)).astype(float)

    if (np.sum(Ir[:,0]) == 0 or Ig[0,0] == 0):
        rt = rt[:, 1:]
        gt = gt[:, 1:]

    if (np.isnan(gt[0,0]) or np.isnan(gt[0,-1])):
        idxnang = np.where(~np.isnan(np.sum(gt, axis=0)))[0]
        gt = gt[:, idxnang]
        rt = rt[:, idxnang]

    Ir = (~np.isnan(rt)).astype(float)
    Ig = (~np.isnan(gt)).astype(float)

    # Initialization
    T = rt.shape[1]
    n = rt.shape[0]
    d = gt.shape[0]
    
    # Newey-West lag parameter for HAC standard errors
    q = compute_newey_west_lag(T, method='stock-watson')

    alphahat = np.full((n, pmax), np.nan)
    R2F = np.full((1, pmax), np.nan)
    R2G = np.full((d, pmax), np.nan)

    if indc == 1:
        Gammahat = np.full((d+1, pmax), np.nan)
        avarhat = np.full((d+1, pmax), np.nan)
    else:
        Gammahat = np.full((d, pmax), np.nan)
        avarhat = np.full((d, pmax), np.nan)

    gthat = np.full((d, T, pmax), np.nan)
    W = np.full((pmax, 1), np.nan)
    pvchiW = np.full((pmax, 1), np.nan)

    # Initialization for Lettau step
    MaxSRmat = np.zeros((1, pmax))
    RMSmat = np.zeros((1, pmax))
    sigmaemat = np.zeros((1, pmax))
    maxfmat = np.zeros((1, pmax))
    weightsmat = np.zeros((pmax, pmax))
    iSRmat = np.zeros((pmax, pmax))

    # Estimation
    rtbar = rt - np.mean(rt, axis=1, keepdims=True)
    rbar = np.mean(rt, axis=1, keepdims=True)

    # Giglio Xiu Method
    gtbar = gt.copy()
    for k in range(d):
        observed = (Ig[k,:] == 1)
        meanvalue = np.mean(gtbar[k, observed])
        gtbar[k,observed] = gtbar[k,observed] - meanvalue

    # Step 1: PCA
    tempD, tempV = np.linalg.eig((1/T) * (rt @ rt.T) + w * (rbar @ rbar.T)) 
    tempeig = np.real(tempD)
    tempidx = np.argsort(tempeig)
    eigvec = tempV[:, tempidx]

    if j == 1:
        MATvarexp = np.zeros((pmax, 1))
        for counter in range(pmax):
            varexp = np.sum(tempeig[-(counter+1):]) / np.sum(tempeig)
            MATvarexp[counter, 0] = varexp

    Gammahatp = np.zeros_like(Gammahat)
    Gammahatse = np.zeros_like(Gammahat)
    vhatold = None
    betahathold = None
    H = None

    for phat in range(1, pmax + 1):
        # Step 2a: Recover latent factors and portfolios' risk exposures
        betahat = eigvec[:, (n-phat):n]
        vhat = betahat.T @ rt

        # Flip betahat and vhat
        betahat = np.fliplr(betahat)
        vhat = np.flipud(vhat)

        # Make factors orthogonal and rotate loadings accordingly
        if phat > 1:
            Z, H = my_orth(vhat.T)
            vhatold = vhat.copy()
            betahathold = betahat.copy()
            vhat = Z
            rotbeta = np.linalg.inv(H) @ betahat.T
            betahat = rotbeta.T
            vhat = vhat.T

            meanstmp = np.mean(vhat, axis=1, keepdims=True)
            signsvtmp = np.sign(meanstmp)
            signsvmattmp = signsvtmp @ np.ones((1, vhat.shape[1]))
            vhat = signsvmattmp * vhat
            meanstmp = np.mean(vhat, axis=1, keepdims=True)
            signsvmattmp = signsvtmp @ np.ones((1, betahat.shape[0]))
            betahat = betahat * signsvmattmp.T

        # Step 2b: Estimate PC lambdas (gamma)
        if indc == 1:
            X = np.hstack([np.ones((n,1)), betahat])
            Gammatilde = np.linalg.inv(X.T @ X) @ X.T @ rbar
        else:
            Gammatilde = np.linalg.inv(betahat.T @ betahat) @ betahat.T @ rbar

        # Step 3: Spanning loadings (eta) and mes. error (what)
        vhatwithmean = vhat.copy()
        vhat = vhat - np.mean(vhat, axis=1, keepdims=True)
        Sigmavhat = vhat @ vhat.T / T
        etahat = gtbar @ vhat.T @ np.linalg.inv(vhat @ vhat.T)
        what = gtbar - etahat @ vhat
            
        # Step 4: Factor g prices of risk (gamma_g) - Fixed dimension handling
        if indc == 1:
            # Extract first element and rest of Gammatilde
            gamma_0 = float(np.real(Gammatilde[0]))  # Convert to scalar
            gamma_rest = etahat @ Gammatilde[1:]  # Matrix multiplication
            
            # Stack results vertically
            Gammahat[:, phat-1] = np.hstack([gamma_0, gamma_rest.ravel()])
        else:
            Gammahat[:, phat-1] = (etahat @ Gammatilde).ravel()
        
        # Step 5: Goodness of Fit
        '''
        Miota = np.eye(n) - np.ones((n,1)) @ np.ones((1,n)) / n
        numerator = float(rbar.T @ Miota @ betahat @ np.linalg.inv(betahat.T @ Miota @ betahat) @ 
                        betahat.T @ Miota @ rbar)
        denominator = float(rbar.T @ Miota @ rbar)
        R2F[0, phat-1] = numerator / denominator
        
        # Fix for R2G calculation
        gtbar_diag = np.diag(gtbar @ gtbar.T)
        vhat_cov = vhat @ vhat.T
        spanning_term = np.diag(etahat @ vhat_cov @ etahat.T)
        R2G[:, phat-1] = spanning_term / gtbar_diag
        
        or
        
        Miota = np.eye(n) - np.ones((n,1)) @ np.ones((1,n)) / n
        R2F[0, phat-1] = (rbar.T @ Miota @ betahat @ np.linalg.inv(betahat.T @ Miota @ betahat) @ 
                          betahat.T @ Miota @ rbar) / (rbar.T @ Miota @ rbar)
        R2G[:, phat-1] = (1 / np.diag(gtbar @ gtbar.T)) * np.diag(etahat @ (vhat @ vhat.T) @ etahat.T)
        '''
        # Calculate R2F (cross-sectional R-squared)
        Miota = np.eye(n) - np.ones((n,1)) @ np.ones((1,n)) / n
        temp_mat = rbar.T @ Miota @ betahat @ np.linalg.inv(betahat.T @ Miota @ betahat) @ betahat.T @ Miota @ rbar
        numerator = float(np.real(temp_mat))
        denominator = float(np.real(rbar.T @ Miota @ rbar))
        R2F[0, phat-1] = numerator / denominator if denominator != 0 else 0
        
        # Calculate R2G (factor spanning R-squared)
        gtbar_var = np.real(np.diag(gtbar @ gtbar.T))
        vhat_cov = vhat @ vhat.T
        spanning_var = np.real(np.diag(etahat @ vhat_cov @ etahat.T))
        R2G[:, phat-1] = np.divide(spanning_var, gtbar_var, where=gtbar_var!=0, out=np.zeros_like(spanning_var))
    	
        # Step 6: Newey-West Estimation of Avar 
        Pi11hat = np.zeros((d*phat, d*phat))
        Pi12hat = np.zeros((d*phat, phat))
        Pi22hat = np.zeros((phat, phat))
        '''
        for t in range(T):
            # In MATLAB: what(:,t) is d×1 and vhat(:,t) is phat×1
            what_t = what[:,[t]]  # d×1
            vhat_t = vhat[:,[t]]  # phat×1
            
            # MATLAB: what(:,t)*vhat(:,t)' is d×phat
            what_vhat = what_t @ vhat_t.T  # d×phat
            
            # vec(what(:,t)*vhat(:,t)') is (d*phat)×1
            vec_what_vhat = vec(what_vhat).reshape(-1, 1)  # Ensure column vector
            
            # Update matrices with proper dimensions
            Pi11hat = Pi11hat + (vec_what_vhat @ vec_what_vhat.T) / T  # (d*phat)×(d*phat)
            Pi12hat = Pi12hat + vec_what_vhat @ vhat_t.T / T  # (d*phat)×phat
            Pi22hat = Pi22hat + vhat_t @ vhat_t.T / T  # phat×phat
            
            for s in range(min(t, q)):
                what_ts = what[:,[t-s-1]]  # d×1
                vhat_ts = vhat[:,[t-s-1]]  # phat×1
                
                what_vhat_lag = what_ts @ vhat_ts.T  # d×phat
                vec_what_vhat_lag = vec(what_vhat_lag).reshape(-1, 1)  # (d*phat)×1
                
                kernel_weight = (1 - (s+1)/(q+1)) / T
                
                Pi11hat = Pi11hat + kernel_weight * (
                    vec_what_vhat @ vec_what_vhat_lag.T +
                    vec_what_vhat_lag @ vec_what_vhat.T
                )
                
                Pi12hat = Pi12hat + kernel_weight * (
                    vec_what_vhat @ vhat_ts.T +
                    vec_what_vhat_lag @ vhat_t.T
                )
                
                Pi22hat = Pi22hat + kernel_weight * (
                    vhat_t @ vhat_ts.T +
                    vhat_ts @ vhat_t.T
                )
        '''
        
        def vec(matrix):
            return matrix.flatten('F') 
        
        for t in range(T):  # Note: Python uses 0-based indexing, but we keep 1-based for consistency
            what_vhat = np.outer(what[:, t], vhat[:, t])
            vec_what_vhat = vec(what_vhat)
            
            Pi11hat = Pi11hat + np.outer(vec_what_vhat, vec_what_vhat) / T
            Pi12hat = Pi12hat + np.outer(vec_what_vhat, vhat[:, t]) / T
            Pi22hat = Pi22hat + np.outer(vhat[:, t], vhat[:, t]) / T
            
            for s in range(1, min(t, q) + 1):
                weight = (1 - s/(q + 1)) / T
                
                what_vhat_t_s = np.outer(what[:, t-s], vhat[:, t-s])
                vec_what_vhat_t_s = vec(what_vhat_t_s)
                
                # First term + second term for Pi11hat
                Pi11hat = Pi11hat + weight * (np.outer(vec_what_vhat, vec_what_vhat_t_s) + 
                                np.outer(vec_what_vhat_t_s, vec_what_vhat))
                
                # First term + second term for Pi12hat
                Pi12hat = Pi12hat + weight * (np.outer(vec_what_vhat, vhat[:, t-s]) + 
                                np.outer(vec_what_vhat_t_s, vhat[:, t]))
                
                # First term + second term for Pi22hat
                Pi22hat = Pi22hat + weight * (np.outer(vhat[:, t], vhat[:, t-s]) + 
                                np.outer(vhat[:, t-s], vhat[:, t]))
        
        # Cross-sectional pricing errors using PCs - Second Fix
        if indc == 1:
            X = np.hstack([np.ones((n,1)), betahat])
            alphahat[:, phat-1] = rbar.flatten() - (X @ Gammatilde).flatten()
        else:
            alphahat[:, phat-1] = rbar.flatten() - (betahat @ Gammatilde).flatten()
        '''
        # Calculate avarhat
        if indc == 1:
            avarhat[:, phat-1] = np.concatenate([
                [(1 - np.mean(betahat, axis=0) @ np.linalg.inv(betahat.T @ betahat/n) @ np.mean(betahat, axis=0).T)**(-1) * 
                  np.var(alphahat[:, phat-1])/n],
                np.diag(
                    np.kron(Gammatilde[1:].T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi11hat @
                    np.kron(np.linalg.inv(Sigmavhat) @ Gammatilde[1:], np.eye(d))/T +
                    np.kron(Gammatilde[1:].T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi12hat @ etahat.T/T +
                    (np.kron(Gammatilde[1:].T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi12hat @ etahat.T).T/T +
                    etahat @ Pi22hat @ etahat.T/T
                ) +
                np.diag(np.var(alphahat[:, phat-1]) @ etahat @ ### np.diag(np.var(alphahat[:, phat-1]) * etahat @ 
                       np.linalg.inv(betahat.T @ betahat/n - np.mean(betahat, axis=0).T @ np.mean(betahat, axis=0)) @
                       etahat.T)/n
            ])
        else:
            avarhat[:, phat-1] = np.diag(
                np.kron(Gammatilde.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi11hat @
                np.kron(np.linalg.inv(Sigmavhat) @ Gammatilde, np.eye(d))/T +
                np.kron(Gammatilde.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi12hat @ etahat.T/T +
                (np.kron(Gammatilde.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi12hat @ etahat.T).T/T +
                etahat @ Pi22hat @ etahat.T/T
            )
        '''
        # Calculate avarhat
        if indc == 1:
            # First term: variance component for the constant
            constant_term = (1 - np.mean(betahat, axis=0) @ np.linalg.inv(betahat.T @ betahat/n) @ 
                            np.mean(betahat, axis=0).T)**(-1) * np.var(alphahat[:, phat-1])/n
            
            # Second term: HAC variance components
            hac_term = np.diag(
                np.kron(Gammatilde[1:].T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi11hat @
                np.kron(np.linalg.inv(Sigmavhat) @ Gammatilde[1:], np.eye(d))/T +
                np.kron(Gammatilde[1:].T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi12hat @ etahat.T/T +
                (np.kron(Gammatilde[1:].T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi12hat @ etahat.T).T/T +
                etahat @ Pi22hat @ etahat.T/T
            )
            
            # Third term: Additional variance component
            beta_adjustment = np.linalg.inv(betahat.T @ betahat/n - np.mean(betahat, axis=0).T @ np.mean(betahat, axis=0))
            additional_term = np.diag(
                np.var(alphahat[:, phat-1]) * (etahat @ beta_adjustment @ etahat.T)
            )/n
            
            # Combine all terms
            avarhat[:, phat-1] = np.concatenate([
                [constant_term],  # Scalar term for the constant
                hac_term + additional_term  # Vector terms for the factors
            ])
        else:
            # Case without constant: only HAC variance components
            avarhat[:, phat-1] = np.diag(
                np.kron(Gammatilde.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi11hat @
                np.kron(np.linalg.inv(Sigmavhat) @ Gammatilde, np.eye(d))/T +
                np.kron(Gammatilde.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi12hat @ etahat.T/T +
                (np.kron(Gammatilde.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi12hat @ etahat.T).T/T +
                etahat @ Pi22hat @ etahat.T/T
            )
        # Step 7: P-values gamma_g
        gammat = Gammahat[:, phat-1] / np.sqrt(avarhat[:, phat-1]) # if with constant first row is always the constant
        Gammahatp[:, phat-1] = f_pval2S(gammat, n-(1+indc)) # if with constant first row is always the constant
        Gammahatse[:, phat-1] = np.sqrt(avarhat[:, phat-1])

        # Step 8: recovery of gt/cleaned factor (eta x vhat)
        gthattmp = etahat @ vhat
        gthat[:, :, phat-1] = gthattmp

        # Step 9: Wald test for the null that the factor g is weak
        if d < 2:
            W[phat-1] = T * (etahat @ (np.linalg.inv(np.linalg.inv(Sigmavhat) @ 
                        Pi11hat @ np.linalg.inv(Sigmavhat))) @ etahat.T)
            pvchiW[phat-1] = chi2.sf(W[phat-1], phat)
        else:
            W[phat-1] = np.nan
            pvchiW[phat-1] = np.nan

        # Lettau step
        tempeig_s = np.sort(tempeig)[::-1]
        MaxSR, weights, iSR, RMS, sigmae, maxf = lettau_rp(phat, rt.T, vhatwithmean.T, tempeig_s)
        MaxSRmat[0, phat-1] = MaxSR
        RMSmat[0, phat-1] = RMS
        sigmaemat[0, phat-1] = sigmae
        maxfmat[0, phat-1] = maxf
        weightsmat[phat-1, :phat] = weights
        iSRmat[phat-1, :phat] = iSR

    # Step 11: choose number of latent factors
    phat = np.zeros((3, 1))
    ppmax = 20
    eig_subset = tempeig[::-1][:ppmax]  # Last ppmax elements in reverse order
        
    # Calculate objectives matching MATLAB exactly
    median_eigs_1 = np.median(eig_subset)
    obj = eig_subset + 0.25 * np.arange(1, ppmax + 1).reshape(-1, 1) * (np.log(n) + np.log(T)) * (n**(-0.5) + T**(-0.5)) * median_eigs_1
    phat[0] = np.argmin(obj) - 1

    median_eigs_2 = np.median(eig_subset)
    obj2 = eig_subset + 0.50 * np.arange(1, ppmax + 1).reshape(-1, 1) * (np.log(n) + np.log(T)) * (n**(-0.5) + T**(-0.5)) * median_eigs_2
    phat[1] = np.argmin(obj2) - 1

    median_eigs_3 = np.median(eig_subset)
    obj3 = eig_subset + 0.75 * np.arange(1, ppmax + 1).reshape(-1, 1) * (np.log(n) + np.log(T)) * (n**(-0.5) + T**(-0.5)) * median_eigs_3
    phat[2] = np.argmin(obj3) - 1
        
    res = {
            # Latent Factor Estimates:
            'vhat': vhatwithmean.T, # Estimated latent factors with mean included, transposed for time-series format
            'betahat': betahat, # Portfolio factor loadings (risk exposures)
            'vhatold': vhatold.T if vhatold is not None else None, # Original estimated factors before orthogonalization/rotation
            'betahathold': betahathold if betahathold is not None else None, # Original factor loadings before rotation
            'H': H if 'H' in locals() else None, # Rotation matrix used to make factors orthogonal
            
            # RP-PCA Results:
            'eigenvalues': np.sort(tempeig)[::-1], # Sorted eigenvalues from PCA in descending order
            'pmax': phat, # Suggested optimal number of factors using different penalty criteria
            
            # Risk and Performance Metrics:    
            'MaxSR': MaxSRmat, # Maximum Sharpe ratio for each number of factors
            'RMS': RMSmat, # Root mean square error of the model
            'sigmae': sigmaemat, # Standard error estimates / idiosynchratic variance
            'weights': weightsmat, # Portfolio weights for each factor specification
            'iSR': iSRmat, # Individual Sharpe ratios for each factor
            'maxnf': maxfmat, # Maximum number of factors used in each specification
            
            # Factor Analysis Results:
            'etahat': etahat, # Spanning loadings (how observable factors load on latent factors)
            'gthat': gthat,  # Cleaned factor proxies (observable factors projected onto latent factors)
            'gtbar': gtbar, # Demeaned observable factors with NaN values excluded
            'R2F': R2F, # Cross-sectional R-squared for pricing equations
            'R2G': R2G, # Time series R-squared for factor spanning regressions
            
            # Stats on Risk Premium Estimates:
            'Gammahat': Gammahat,  # Estimated risk premia for observable factors
            'Gammahatp': Gammahatp, # P-values for risk premium estimates
            'Gammahatse': Gammahatse, # Standard errors for risk premium estimates
            'avarhat': avarhat, # Asymptotic variance of risk premium estimates
            'gammatilde': Gammatilde, # Risk premia estimates for latent factors
            
            # Model Diagnostics / Weak Factor:
            'alphahat': alphahat, # Pricing errors from the cross-sectional regressions
            'wstat': W, # Wald test statistics for factor weakness
            'wtestp': pvchiW, # P-values from Wald tests
            'Newey-West Lags': q # Lag lenght for Newey West
        }

    return res
   
'''
############## Example #################################
import sys
sys.path.append('../')
import os
from Helper.measurements import *
from Helper.data_store import *

# Load data
return_monhtly=load_from_HDFStore(group='Raw_data',
                                    path=os.path.dirname(os.getcwd())+r'\Raw_Data\Input',
                                    dataset='return_series_monthly',
                                    filename='Commodity_HDFStore').loc['1990-01-31':,:]

names_lst = load_yaml_const(filename = os.path.dirname(os.getcwd())+r'\const.yaml', num_lst = 7)
fullnames_lst = load_yaml_const(filename = os.path.dirname(os.getcwd())+r'\const.yaml', num_lst = 6)
graph_lst = load_yaml_const(filename = os.path.dirname(os.getcwd())+r'\const.yaml', num_lst = 8)

def load_fct_rets(names_lst,fullnames_lst,graph_lst,return_monhtly): 
    
    lst_ = []
    for k,j in zip(names_lst,fullnames_lst):
        l = load_from_HDFStore(group = 'Factor_data', path = os.path.dirname(os.getcwd())+r'\Raw_Data\Input',
                                   dataset = str(k)+'_perf_monthly', 
                                   filename = 'Commodity_HDFStore_Returns')
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
names = fct_rets.columns.tolist()
res_spca = {}
for i,k in enumerate(names):
    # Data pre-processing
    rt, gt = equal_length(return_monhtly.fillna(0),fct_rets.iloc[:,[i]].dropna(0))
    rt = rt.T.to_numpy()
    gt = gt.T.to_numpy()
    print(fct_rets.iloc[:,[i]].columns[0])
    # Set parameters
    tuningrange = np.linspace(-1, 40, 15) # weight for mean adjustment
    
res =    rppca_rfs(
    rt=rt,
    gt=gt,
    pmax=5,
    indc=1,
    j=1,
    w=10)
    
    newey west 
    und
    data handler
    r2 hochaber pvalue?

from Models.mod_rppca_adv import *

res_2 = rppca_rfs_old(
    rt=rt,
    gt=gt, 
    pmax=2,
    indc=1, 
    j=1, 
    w=10
    ) 
     
'''