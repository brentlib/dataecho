import pandas as pd

# create a function that counts the values of a column - single column - include counts and percentages
def single_tabulate(df, column, sort_by_count=True, include_missing=False, include_total=False):
    """
    Counts the values of a column and returns a pandas data frame with counts and percentages.

    Args:
        df (pandas.DataFrame): The data frame to count the values of.
        column (str): The column to count the values of.
        sort_by_count (bool): Whether to sort the data frame by count. If True, the data frame will be sorted by count. If False, the data frame will be sorted by value.
        include_missing (bool): Whether to include missing values in the count. If True, missing values will be included in the count. If False, missing values will be excluded from the count.
        include_total (bool): Whether to include a total row in the data frame. If True, a total row will be included in the data frame. If False, a total row will not be included in the data frame.

    Returns:
        pandas.DataFrame: A data frame with counts and percentages.
    """
    # count the values of the column
    if include_missing:
        counts = df[column].value_counts(dropna=False)
    else:
        counts = df[column].value_counts(dropna=True)
    # convert to pandas data frame
    counts = counts.to_frame()
    # convert index to column
    counts['value'] = counts.index
    counts.reset_index(drop=True, inplace=True)
    counts.columns = ['freq', 'value']
    # add a column for the percentage
    counts['pct'] = counts['freq'] / counts['freq'].sum()
    # add a column for the variable
    counts['variable'] = column
    # rearrange columns - variable, value, freq, percentage
    counts = counts[['variable', 'value', 'freq', 'pct']]
    # sort by count if sort_by_count is True
    if sort_by_count:
        counts.sort_values(by='freq', ascending=False, inplace=True)
    else:
        counts.sort_values(by='value', ascending=True, inplace=True)
    # add a row at the bottom for the total
    if include_total:
        total_row = pd.DataFrame({'variable': [column], 'value': ['TOTAL'], 'freq': [counts['freq'].sum()], 'pct': [1]})
        counts = pd.concat([counts, total_row], ignore_index=True)
    return counts