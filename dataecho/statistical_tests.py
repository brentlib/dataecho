from typing import Union
import numpy as np
import scipy.stats as stats

# difference of means test (Welch's t-test)
def welch_t_test(
        sample_size_1: int, 
        sample_size_2: int, 
        sample_mean_1: Union[float, int], 
        sample_mean_2: Union[float, int], 
        sample_std_1: Union[float, int], 
        sample_std_2: Union[float, int]
    ) -> float:
    """
    Performs Welch's t-test for comparing two sample means, accounting for unequal variances. 
    This implementation of Welch's t-test does not require the raw data, only statistics.

    Args:
        sample_size_1 (int): The size of the first sample.
        sample_size_2 (int): The size of the second sample.
        sample_mean_1 (float): The mean of the first sample.
        sample_mean_2 (float): The mean of the second sample.
        sample_std_1 (float): The standard deviation of the first sample.
        sample_std_2 (float): The standard deviation of the second sample.

    Raises:
        ValueError: If sample sizes are less than 2
        ValueError: If standard deviations are negative
        TypeError: If inputs are not numeric

    Returns:
        float: The p-value of the Welch's t-test.

    Example:
        >>> p_value = welch_t_test(7, 14, 3.8571, 3.2857, 0.6901, 0.7263)
        >>> print(f"The p-value is {p_value:.4f}")
        The p-value is 0.1029

    References:
        
    """
    # input validation
    if not isinstance(sample_size_1, int) or not isinstance(sample_size_2, int):
        raise TypeError("Error: Sample sizes must be integers")
    if not all(isinstance(x, (int, float)) for x in [sample_mean_1, sample_mean_2, sample_std_1, sample_std_2]):
        raise TypeError("Error: Means and standard deviations must be numeric")
    if sample_size_1 < 2 or sample_size_2 < 2:
        raise ValueError("Error: Sample sizes must be at least 2")
    if sample_std_1 < 0 or sample_std_2 < 0:
        raise ValueError("Error: Standard deviations must be non-negative")
    
    # calculate difference between means
    mean_diff = sample_mean_1 - sample_mean_2

    # calculate the standard error of the difference of means
    standard_error = np.sqrt((sample_std_1**2 / sample_size_1) + (sample_std_2**2 / sample_size_2))

    # calculate the t-statistic
    t_statistic = mean_diff / standard_error

    # calculate the degrees of freedom
    df = ((sample_std_1**2 / sample_size_1 + sample_std_2**2 / sample_size_2)**2) / ((sample_std_1**2 / sample_size_1)**2 / (sample_size_1 - 1) + (sample_std_2**2 / sample_size_2)**2 / (sample_size_2 - 1))

    # calculate the p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
    return p_value
