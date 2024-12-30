from typing import Union
import numpy as np
import scipy.stats as stats
import math

# compare two proportions
def prop_sig_test(
        sample_size_1: int,
        sample_size_2: int,
        response_rate_1: float,
        response_rate_2: float
    ) -> float:
    """
    Performs a significance test for comparing two proportions.

    Args:
        sample_size_1 (int): The size of the first sample.
        sample_size_2 (int): The size of the second sample.
        response_rate_1 (float): The response rate of the first sample.
        response_rate_2 (float): The response rate of the second sample.

    Returns:
        float: The p-value of the significance test.

    References:
    """

    sd1 = np.sqrt(sample_size_1 * response_rate_1 * (1 - response_rate_1))
    sd2 = np.sqrt(sample_size_2 * response_rate_2 * (1 - response_rate_2))
    
    min_sd = min(sd1, sd2)
    max_sd = max(sd1, sd2)
    sd_ratio = min_sd / max_sd

    if sd_ratio > 2:
        pooled = False
    else:
        pooled = True

    if pooled:
        p_hat = (sample_size_1 * response_rate_1 + sample_size_2 * response_rate_2) / (sample_size_1 + sample_size_2)
        z_score = (response_rate_1 - response_rate_2) / np.sqrt(p_hat * (1-p_hat) * (1/sample_size_1 + 1/sample_size_2))
        z_score_abs = abs(z_score)
        p_value = 2 * (1 - stats.norm.cdf(z_score_abs))
        return p_value
    else:
        z_score = (response_rate_1 - response_rate_2) / np.sqrt(response_rate_1 * (1 - response_rate_1) / sample_size_1 + response_rate_2*(1-response_rate_2)/sample_size_2)
        z_score_abs = abs(z_score)
        p_value = 2 * (1 - stats.norm.cdf(z_score_abs))
        return p_value

# find margin of error
def find_margin_of_error(
        sample_size: int,
        confidence_level: float = 0.95,
        response_rate: float = 0.5,
        population_size: int = 1000000
    ) -> float:
    """
    Finds the margin of error for a survey given a sample size. 

    Args:
        sample_size (int): The size of the sample.
        confidence_level (float): The desired confidence level (e.g. 0.95 for 95% confidence level)
        response_rate (float): The expected response rate (e.g. 0.5 for 50% response rate)
        population_size (int): The size of the population (e.g. 1000000 for 1 million people)

    Returns:
        float: The margin of error
    """

    # calculate z-score for the given confidence level
    z_score = abs(stats.norm.ppf((1 - confidence_level) / 2))

    # Calculate the margin of error
    d1 = z_score * z_score * response_rate * (1 - response_rate)
    d2 = d1 * (population_size - sample_size) / (sample_size * (population_size - 1))
    margin_of_error = math.sqrt(d2)

    return margin_of_error

# find optimal sample size
def find_optimal_sample_size(
        margin_of_error: float,
        confidence_level: float,
        response_rate: float = 0.5,
        population_size: int = 1000000
    ) -> int:
    """
    Finds the optimal sample size for a survey given a desired margin of error and confidence level.

    Args:
        margin_of_error (float): The desired margin of error (e.g. 0.05 for 5% margin of error)
        confidence_level (float): The desired confidence level (e.g. 0.95 for 95% confidence level)
        response_rate (float): The expected response rate (e.g. 0.5 for 50% response rate)
        population_size (int): The size of the population (e.g. 1000000 for 1 million people)

    Returns:
        int: The optimal sample size

    References:

    """
    # Input validation
    if not all(isinstance(x, (int, float)) for x in [margin_of_error, confidence_level, response_rate, population_size]):
        raise TypeError("All inputs must be numeric")
    if not 0 < margin_of_error < 1:
        raise ValueError("Margin of error must be between 0 and 1")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    if not 0 <= response_rate <= 1:
        raise ValueError("Response rate must be between 0 and 1")
    if population_size < 1:
        raise ValueError("Population size must be positive")

    # calculate z-score for the given confidence level
    z_score = abs(stats.norm.ppf((1 - confidence_level) / 2))

    # Calculate the sample size using Cochran's formula
    d1 = z_score * z_score * response_rate * (1 - response_rate)
    d2 = (population_size - 1) * (margin_of_error * margin_of_error) + d1
    rec_sample_size = math.ceil(population_size * d1 / d2)

    return rec_sample_size

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
