import pandas as pd
import numpy as np
from scipy import stats

scores = pd.Series([72, 65, 88, np.nan, 54, 91, 76, np.nan, 83, 69])

# (a) describe
print("Descriptive statistics:")
print(scores.describe())

# (b) skewness and kurtosis (drop NaN first)
clean = scores.dropna()
print("\nSkewness:", stats.skew(clean))
print("Kurtosis:", stats.kurtosis(clean))

# (c) missing count
print("\nMissing values:", scores.isnull().sum())

# (d) fill with mean
filled = scores.fillna(scores.mean())
print("\nFilled scores:")
print(filled)