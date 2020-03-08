---
title: "Practical Statistics for Data Scientists"
author: "Dan Martin"
date: "2020-02-17"
...
# 1 - Exploratory Data Analysis
---

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
    
    
**table 1.1**

|Category        |Currency|sellerRating|Duration|endDay|closePrice|openPrice|Competitive?|
|----------------|--------|------------|--------|------|----------|---------|------------|
|Music/Movie/Game|US      |3249        |5       |Mon   |0.01      |0.01     |0           |
|Music/Movie/Game|US      |3249        |5       |Mon   |0.01      |0.01     |0           |
|Automotive      |US      |3115        |7       |Tue   |0.01      |0.01     |0           |
|Automotive      |US      |3115        |7       |Tue   |0.01      |0.01     |0           |
|Automotive      |US      |3115        |7       |Tue   |0.01      |0.01     |0           |
|Automotive      |US      |3115        |7       |Tue   |0.01      |0.01     |0           |
|Automotive      |US      |3115        |7       |Tue   |0.01      |0.01     |1           |
|Automotive      |US      |3115        |7       |Tue   |0.01      |0.01     |1           |

Table: Sample Data

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

$$\overline{x} = \mu = \frac{\sum_{i}^{n}x_{i}}{n}$$

### Trimmed Mean

$$\overline{x}_{t} = x = \frac{\sum_{i = p + 1}^{n - p}x_{i}}{n - 2p}$$

### Weighted Mean

$$\overline{x}_{w} = \frac{\sum_{i = 1}^{n}w_{i}x_{i}}{\sum_{i}^{n}w_{i}}$$

### _Code Examples_


```python
import pandas as pd
import numpy as np
from scipy.stats import trim_mean

state = pd.read_csv('../data/state.csv')

x_bar = np.mean(state['Population'])
x_bar_t = trim_mean(state['Population'], 0.1)
x_med = np.median(state['Population'])

# output
print(f'Mean: {x_bar}')
print(f'Trimmed Mean: {x_bar_t}')
print(f'Median: {x_med}')
```

    Mean: 6162876.3
    Trimmed Mean: 4783697.125
    Median: 4436369.5



```python
x_bar_wt = np.average(state['Murder.Rate'], weights=state['Population'])

# output
print(f'Weighted Mean: {x_bar_wt}')
```

    Weighted Mean: 4.445833981123393


### Estimates of Variability
