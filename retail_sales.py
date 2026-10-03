import pandas as pd
import numpy as np
from scipy.stats import zscore

df = pd.DataFrame({
    "Day": range(1, 11),
    "Revenue": [45, 52, np.nan, 48, 390, 55, np.nan, 50, 47, 53]
})

# (a) missing values + median fill
print("Missing values in Revenue:", df["Revenue"].isnull().sum())
df["Revenue"] = df["Revenue"].fillna(df["Revenue"].median())
print("\nAfter median fill:")
print(df)

# (b) normalization
rev = df["Revenue"]
df["Revenue_MinMax"] = (rev - rev.min()) / (rev.max() - rev.min())
df["Revenue_Zscore"] = zscore(rev)

# (c) IQR outliers
Q1 = rev.quantile(0.25)
Q3 = rev.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df["Outlier"] = (rev < lower) | (rev > upper)

print("\nIQR bounds:", lower, "-", upper)
print("IQR outliers detected on day(s):", df.loc[df["Outlier"], "Day"].tolist())

# (d) final table
print("\nFinal table:")
print(df.round(4))