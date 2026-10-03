import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

df = pd.DataFrame({
    "Hour": range(1, 16),
    "Temperature": [22, 23, np.nan, np.nan, np.nan, 25, 26, 58,
                    24, 23, 22, 21, 23, 24, 25]
})

# (a) stats BEFORE cleaning
raw = df["Temperature"].copy()
print("Stats BEFORE cleaning:")
print(raw.describe())
print("Skewness:", stats.skew(raw.dropna()))
print("Kurtosis:", stats.kurtosis(raw.dropna()))
mean_before, std_before = raw.mean(), raw.std()

# (b) linear interpolation
df["Temp_clean"] = df["Temperature"].interpolate(method="linear")
print("\nAfter interpolation, missing values:", df["Temp_clean"].isnull().sum())

# (c) Z-score outlier detection (threshold = 2), replace only flagged value
z = zscore(df["Temp_clean"])
mask = np.abs(z) > 2
median_val = df["Temp_clean"].median()      # computed BEFORE replacement
for _, row in df[mask].iterrows():
    print(f"Z-score(>2) outlier detected at Hour: {int(row['Hour'])} (value: {row['Temp_clean']})")
print("Replaced with column median:", median_val)
df.loc[mask, "Temp_clean"] = median_val

# (d) standardize + comparison
df["Temp_Zscore"] = zscore(df["Temp_clean"])
comparison = df[["Hour", "Temperature", "Temp_clean", "Temp_Zscore"]]
comparison = comparison.rename(columns={"Temperature": "Temp_raw"})
print("\nBefore/After comparison:")
print(comparison.round(4))

print("\nStats AFTER cleaning:")
print("Mean:", df["Temp_clean"].mean(), "Std:", df["Temp_clean"].std())
print(f"(compare to BEFORE: Mean {mean_before:.2f}, Std {std_before:.2f})")