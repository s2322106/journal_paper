# import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("Social_Vulnerability_Index_2018_-_United_States__tract_20250119.csv")

# Check the data overview
print("Column names:")
print(df.columns)

print("\nSummary of 'RPL_THEMES' before preprocessing:")
print(df["RPL_THEMES"].describe())

# Keep and copy data where 'RPL_THEMES' is between 0 and 1 
df1 = df[df["RPL_THEMES"].between(0, 1)].copy()

print("\nSummary of 'RPL_THEMES' (after filtering):")
print(df1["RPL_THEMES"].describe())

# Divide 'RPL_THEMES' into three classes (low, medium, high) using tertiles 
df1['RPL_THEMES_BIN'] = pd.qcut(df1['RPL_THEMES'], q=3, labels=[0, 1, 2])

# Check the distribution of the new classes
print("\nDistribution of 'RPL_THEMES_BIN':")
print(df1['RPL_THEMES_BIN'].value_counts())

# Define target and feature variables 
target_variable = 'RPL_THEMES_BIN'
feature_variables = ['EP_POV', 'EP_UNEMP', 'EP_NOHSDP', 'EP_MINRTY', 'EP_AGE65']
X = df1[feature_variables]
y = df1[target_variable].astype(int)  # Convert categorical to integer

# Check the first few rows of feature variables
print("\nFirst few rows of features:")
print(X.head())

# Standardize the data
X = (X - X.mean()) / X.std()

print("\nFirst few rows of target variable:")
print(y.head())

# Check for missing values
print("\nNumber of missing values:")
print(X.isnull().sum())

# Fill missing values with the median  
X = X.fillna(X.median())

# Check missing values again 
print("\nNumber of missing values (after imputation):")
print(X.isnull().sum())

# Set float display to 8 decimal places  
pd.options.display.float_format = '{:0.15f}'.format

# Apply KBinsDiscretizer
kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
X_binned = kbd.fit_transform(X)
X_binned = pd.DataFrame(X_binned, columns=feature_variables)

print("\nFirst few rows of data after binning into 3 bins:")
print(X_binned.head())

# Perform Chi-square test 
chi2_stat, p_values = chi2(X_binned, y)

# Create a DataFrame for statistics and p-values
chi2_results = pd.DataFrame({
    'Feature': feature_variables,
    'Chi2 Statistic': chi2_stat,
    'p-value': p_values
})

# Sort by Chi-square statistic and get the top 5 
chi2_top5 = chi2_results.sort_values(by='Chi2 Statistic', ascending=False)

print("\nTop 5 important features based on Chi-square test:") 
print(chi2_top5[['Feature', 'Chi2 Statistic', 'p-value']])

# Calculate feature importance using Spearman correlation
spearman_results = []
for feature in feature_variables:
    corr, p = spearmanr(X[feature], y)
    spearman_results.append({
        'Feature': feature,
        'Spearman Correlation': corr,  # Signed correlation 
        'p-value': round(p, 8)
    })

spearman_df = pd.DataFrame(spearman_results)

# Sort by absolute correlation and get the top 5
spearman_top5 = spearman_df.reindex(
    spearman_df['Spearman Correlation'].abs().sort_values(ascending=False).index
)

print("\nTop 5 important features based on Spearman correlation:")  
print(spearman_top5[['Feature', 'Spearman Correlation', 'p-value']])

# Calculate feature importance using Kendall's Tau 
from scipy.stats import kendalltau

kendall_results = []

for feature in feature_variables:

    corr, p = kendalltau(X[feature], y)
    kendall_results.append({
        'Feature': feature,
        'Kendall Tau': corr,
        'p-value': round(p, 8)
    })

kendall_df = pd.DataFrame(kendall_results)

# Sort by absolute value and get the top 5
kendall_top5 = kendall_df.reindex(
    kendall_df['Kendall Tau'].abs().sort_values(ascending=False).index
)

print("\nTop 5 important features based on Kendall's Tau:") 
print(kendall_top5[['Feature', 'Kendall Tau', 'p-value']])

# Build Random Forest model 
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Get feature importances 
rf_importances = pd.Series(rf.feature_importances_, index=feature_variables)
rf_top5 = rf_importances.sort_values(ascending=False)

print("\nTop 5 important features based on Random Forest:") 
print(rf_top5)

from xgboost import XGBClassifier

# Build and train XGBoost model
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X, y)

# Get feature importances  
xgb_importances = pd.Series(xgb.feature_importances_, index=feature_variables)
xgb_top5 = xgb_importances.sort_values(ascending=False)

print("\nTop 5 important features based on XGBoost:")  
print(xgb_top5)

# Build SVM model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create pipeline
svc = make_pipeline(StandardScaler(), SVC(kernel='linear', random_state=42))
svc.fit(X, y)

# Get feature weights
svc_weights = pd.Series(svc.named_steps['svc'].coef_[0], index=feature_variables)
svc_top5 = svc_weights.abs().sort_values(ascending=False)

print("\nTop 5 important features based on SVM:")
print(svc_top5)

# Build Lasso regression model 
from sklearn.linear_model import LogisticRegression

# Create pipeline 
lasso = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', random_state=42))
lasso.fit(X, y)

# Get feature weights
lasso_weights = pd.Series(lasso.named_steps['logisticregression'].coef_[0], index=feature_variables)
lasso_top5 = lasso_weights.abs().sort_values(ascending=False)
print("\nTop 5 important features based on Lasso regression:")
print(lasso_top5)

# Build Naive Bayes model
from sklearn.naive_bayes import GaussianNB

# Create pipeline
nb = make_pipeline(StandardScaler(), GaussianNB())
nb.fit(X, y)

# Get feature weights
nb_weights = pd.Series(nb.named_steps['gaussiannb'].theta_[0], index=feature_variables)
nb_top5 = nb_weights.abs().sort_values(ascending=False)

print("\nTop 5 important features based on Naive Bayes:") 
print(nb_top5)


# Create a rank list from 1 to 5 
rank = [1, 2, 3, 4, 5]

# Get ranked features for each method 
features_Kendall_ranked = kendall_top5['Feature'].tolist()
features_chi2_ranked = chi2_top5['Feature'].tolist()
features_spearman_ranked = spearman_top5['Feature'].tolist()

# Create a combined DataFrame 
rank_table = pd.DataFrame({
    'Rank': rank,
    'Chi-Squared Statistic': features_chi2_ranked,
    'Spearman Correlation': features_spearman_ranked,
    'Kendall Tau': features_Kendall_ranked
})

# Set index to Rank
rank_table = rank_table.set_index('Rank')

print("\nCombined top 5 features from each method:")  
print(rank_table)

# Get top 5 features for each method in ranking order
top_features_rf = rf_importances.sort_values(ascending=False).index.tolist()
top_features_xgb = xgb_importances.sort_values(ascending=False).index.tolist()
top_features_svc = svc_weights.abs().sort_values(ascending=False).index.tolist()
top_features_lasso = lasso_weights.abs().sort_values(ascending=False).index.tolist()
top_features_nb = nb_weights.abs().sort_values(ascending=False).index.tolist()

# Create a rank list from 1 to 5  
rank = list(range(1, 6))

# Create a combined DataFrame  
df_rank = pd.DataFrame({
    'Rank': rank,
    'Random Forest': top_features_rf,
    'XGBoost': top_features_xgb,
    'Lasso': top_features_lasso,
    'SVM': top_features_svc,
    'Naive Bayes': top_features_nb
})

print("\nFeature ranking by each method:")  
print(df_rank)

from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF calculation function 
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values(by='VIF', ascending=False)

# Initial VIF calculation 
vif_data = calculate_vif(X)
print("\nInitial VIF:")
print(vif_data)
