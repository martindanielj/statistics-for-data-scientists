---
title: "Practical Statistics for Data Scientists"
author: "Dan Martin"
date: "2020-02-17"
...
# 1 - Exploratory Data Analysis

## Elements of Structured Data

### Key Terms for Data Types

- **_Continuous:_** Data that can take on any value in an interval
    - **Synonyms:** interval, float, numeric
- **_Discrete:_** Data that can take on only integer values, such as counts
    - **Synonyms:** integer, count
- **_Categorical:_** Data that can take on only a specific set of values representing a set of possible categories
    - **Synonyms:** enums, enumerated, factors, nominal, polychotomous
- **_Binary:_** A special case of categorical data with just two categories of values (0/1, true/false)
    - **Synonyms:** dichotomous, logical, indicator, boolean
- **_Ordinal:_** Categorical Data that has an explicit ordering
    - **Synonyms:** ordered factor
    
### Key Ideas

- Data is typically classified in software by type
- Data types include numeric (continuous, discrete) and categorical (binary, ordinal)
- Data typing in software acts s a signal to the software on how to process the data
    
## Rectangular Data

### Key Terms for Rectangular Data

- **_Dataframe:_** Rectangular data (like a spreadsheet) is the basic data structure for statistical and machine learning models
- **_Feature:_** A column in the table is commonly referred to as a _feature_
    - **Synonyms:** attribute, input, predictor, variable
- **_Outcome:_** Many data science projects involve predicting an _outcome_ - often a yes/no outcome (In table 1.1 it is 'auction was competitive or not').  The _features_ are sometimes used to predict the outcome in an experiment or study
    - **Synonyms:** dependent variable, response, target, output


|Category          |Currency|sellerRating|Duration|endDay|closePrice|openPrice|Competitive?|
|------------------|--------|------------|--------|------|----------|---------|------------|
|Music/ Movie/ Game|US      |3249        |5       |Mon   |0.01      |0.01     |0           |
|Music/ Movie/ Game|US      |3249        |5       |Mon   |0.01      |0.01     |0           |
|Automotive        |US      |3115        |7       |Tue   |0.01      |0.01     |0           |
|Automotive        |US      |3115        |7       |Tue   |0.01      |0.01     |0           |
|Automotive        |US      |3115        |7       |Tue   |0.01      |0.01     |0           |
|Automotive        |US      |3115        |7       |Tue   |0.01      |0.01     |0           |
|Automotive        |US      |3115        |7       |Tue   |0.01      |0.01     |1           |
|Automotive        |US      |3115        |7       |Tue   |0.01      |0.01     |1           |

Table 1.1: Sample Data

## Estimates of Location

Location can also be throught of as central tendency, where does the "typical" value lay?

### Key Terms for Estimates of Location

- **_Mean:_** The sum of all values divided by the number of values
    - **Synonyms:** average
- **_Weighted Mean:_** The sum of all values times a weight divided by the sum of the weights
    - **Synonyms:** weighted average
- **_Median:_** The value such that one half of the data lies above and below
    - **Synonyms:** 50th Percentile
- **_Percentile:_** The value such that P percent of the data lies below
    - **Synonyms:** quantile
- **_Weighted Median:_** The value such that one-half of the sum of the weights lies above and below the sorted data
- **_Trimmed Mean:_** The average of all values after dropping a fixed number of extreme values
    - **Synonyms:** truncated mean



### Mean

The mean represents the average - it's not a terribly robust estimate, however you can use a trimmed or weighted mean to improve resistance to outliers.

$$\overline{x} = \mu = \frac{\sum_{i}^{n}x_{i}}{n}$$

#### Trimmed Mean

$$\overline{x}_{t} = x = \frac{\sum_{i = p + 1}^{n - p}x_{i}}{n - 2p}$$

#### Weighted Mean

$$\overline{x}_{w} = \frac{\sum_{i = 1}^{n}w_{i}x_{i}}{\sum_{i}^{n}w_{i}}$$

### Robust Metrics

Median, trimmed and weighted mean are resistant to outliers - ...

### _Code Examples_


```python
import pandas as pd
import numpy as np
from scipy.stats import trim_mean
import wquantiles

state = pd.read_csv('../data/state.csv')

# Population

population_x_bar = np.mean(state['Population'])
population_x_bar_t = trim_mean(state['Population'], 0.1)
population_x_med = np.median(state['Population'])

# Murder Rate

murder_rate_x_bar = np.mean(state['Murder.Rate'])
murder_rate_x_bar_t = trim_mean(state['Murder.Rate'], 0.1)
murder_rate_x_bar_wt = np.average(state['Murder.Rate'], 
                                  weights=state['Population'])
murder_rate_x_med = np.median(state['Murder.Rate'])
murder_rate_x_med_w = wquantiles.median(state['Murder.Rate'], 
                                        weights=state['Population'])

# -----------------------------------
# Summary Output - Location Estimates
# -----------------------------------

print('Population Metrics')
print('-'*30,'\n')
print(f'Mean: {population_x_bar:,.0f}')
print(f'Trimmed Mean: {population_x_bar_t:,.0f}')
print(f'Median: {population_x_med:,.0f}\n')

print('Murder Rate Metrics')
print('-'*30,'\n')
print(f'Mean: {murder_rate_x_bar:,.2f}')
print(f'Trimmed Mean: {murder_rate_x_bar_wt:,.3f}')
print(f'Weighted Mean: {murder_rate_x_bar_wt:,.5f}')
print(f'Median: {murder_rate_x_med:,.3f}')
print(f'Weighted Median: {murder_rate_x_med_w:,.3f}')
```

    Population Metrics
    ------------------------------ 
    
    Mean: 6,162,876
    Trimmed Mean: 4,783,697
    Median: 4,436,370
    
    Murder Rate Metrics
    ------------------------------ 
    
    Mean: 4.07
    Trimmed Mean: 4.446
    Weighted Mean: 4.44583
    Median: 4.000
    Weighted Median: 4.400


### Key Ideas

- The basic metric for location is the mean, but it can be sensitive to extreme values (outliers)
- Other metrics (median, trimmed mean) are less sensitive to outliers and unusual distributions, hence more robust

### Further Reading

[Wikipedia: Central Tendency](https://en.wikipedia.org/wiki/Central_tendency)  
John Tukey - _Exploratory Data Analysis_ 1977 Pearson

## Estimates of Variability

Variability measures disperion around the middle (central tendency).  Part of statistics is understanding random vs. real variability.

### Key Terms for Variability Metrics
- **_Deviations:_** The difference between the observed value and the estimate of location
    - **Synonyms:** errors, residuals
- **_Variance:_** The sum of squared deviations from the mean divided by $n - 1$, where $n$ is the number of observations
    - **Synonyms:** mean-squared-error
- **_Standard Deviation:_** The square root of the variance
    - **Synonyms:** l1-norm, Euclidean norm
- **_Mean Absolute Deviation:_** The mean of the absolute deviations from the mean
    - **Synonyms:** l2-norm, Manhattan norm
- **_Median Absolute Deviation from the Median:_** The median of the absolute values of the deviations from the median
- **_Range:_** The difference between the largest and the smallest values in a dataset
- **_Order Statistics:_** Metrics based on the data values ordered from smallest to largest
    - **Synonyms:** ranks
- **_Percentile:_** The value such that P-percent take on this value or less and (100 - P) take on this value or more
    - **Synonyms:** quantile
- **_Interquartile Range:_** The difference between the 75th and 25th percentile
    - **Synonyms:** IQR
    
### Standard Deviation and Related Estimates

Since the sum of all deviations is zero, there are corrections to the data that need to be made to get meaningful measures of dispersion.

#### Mean Absolute Deviation

Using the absolute deviations eliminates the offset of the positive and negative deviations:

$$\text{Mean Absolution Deviation} = \frac{\sum_{i=1}^{n}\left|x_i - \overline{x}\right|}{n}$$

#### Variance and Standard Deviation

These are two of the most widely used measures of variability - squaring the deviations eliminates the positive/negative offset:

$$\text{Variance} = s^2 = \frac{\sum_{i=1}^{n} (x_i - \overline{x})^2}{n-1}$$  
$$\text{Standard Deviation} = s = \sqrt{\text{Variance}}$$  
The standard deviation is easy to work with since it's on the same scale as the measure in question

> **Degrees of Freedom - $n$ vs. $n - 1$:**
> The use of $n - 1$ vs $n$ in calulcating standard deviation comes from the concept of degrees of freedom.  The estimate of variance without using degrees of freedom results in what's called a biased estimate.  This is because there is a constraint placed on the calulation of the estimate of variance - namely, you have already estimated the mean.  This means that you have one fewer degree of freedom in the system because of this constraint.  Hence, dividing by $n - 1$ yields what we call the _unbiased estimator_.

**Variance**, **Standard Deviation**, and **Mean Absolute Deviation** are not robust estimates - they are all sensitive to outliers, especially variance and SD since they use squared deviations.



```python
# -------------------------------------
# Variance Measures
# -------------------------------------

# Calculations

population_sd = np.std(state['Population'])

# Output

print('Population - Variability')
print('-'*30,'\n')
print(f'Standard Deviation: {population_sd:,.3f}')

```

    Population - Variability
    ------------------------------ 
    
    Standard Deviation: 6,779,407.115


## Exploring the Data Distribution
