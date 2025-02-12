# ライブラリのインポート
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
df = pd.read_csv("Social_Vulnerability_Index_2018_-_United_States__tract_20250119.csv")

# データの概要を確認
print("データの列名:")
print(df.columns)

print("\n'RPL_THEMES'の概要前処理前:")
print(df["RPL_THEMES"].describe())

# 'RPL_THEMES'が0から1の範囲内にあるデータを保持し、コピー
df1 = df[df["RPL_THEMES"].between(0, 1)].copy()

print("\n'RPL_THEMES'の概要（フィルタ後）:")
print(df1["RPL_THEMES"].describe())

# 'RPL_THEMES' を3クラスに分割（低、中、高） using tertiles
df1['RPL_THEMES_BIN'] = pd.qcut(df1['RPL_THEMES'], q=3, labels=[0, 1, 2])

# 新しいクラスの分布を確認
print("\n'RPL_THEMES_BIN'の分布:")
print(df1['RPL_THEMES_BIN'].value_counts())

# 目的変数と説明変数の設定
target_variable = 'RPL_THEMES_BIN'
feature_variables = ['EP_POV', 'EP_UNEMP', 'EP_NOHSDP', 'EP_MINRTY', 'EP_AGE65']
X = df1[feature_variables]
y = df1[target_variable].astype(int)  # カテゴリ型から整数型に変換


# データの先頭を確認
print("\n特徴量の先頭:")
print(X.head())

# データの標準化
X = (X - X.mean()) / X.std()

print("\n目的変数の先頭:")
print(y.head())

# 欠損値の確認
print("\n欠損値の数:")
print(X.isnull().sum())

# 欠損値を中央値で補完
X = X.fillna(X.median())

# 再度欠損値を確認
print("\n欠損値の数（補完後）:")
print(X.isnull().sum())

import pandas as pd

# 小数点第8位まで表示する設定
pd.options.display.float_format = '{:0.15f}'.format

# そのままコードを実行
kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
X_binned = kbd.fit_transform(X)
X_binned = pd.DataFrame(X_binned, columns=feature_variables)

print("\n連続変数を3ビンに分割したデータの先頭:")
print(X_binned.head())

# カイ二乗検定の実施
chi2_stat, p_values = chi2(X_binned, y)

# 統計量とp値をデータフレームにまとめる
chi2_results = pd.DataFrame({
    'Feature': feature_variables,
    'Chi2 Statistic': chi2_stat,
    'p-value': p_values
})

# 統計量でソートしてトップ5を取得
chi2_top5 = chi2_results.sort_values(by='Chi2 Statistic', ascending=False)

print("\nカイ二乗検定による特徴量重要度（トップ5）:")
print(chi2_top5[['Feature', 'Chi2 Statistic', 'p-value']])

# スピアマン相関の計算
spearman_results = []
for feature in feature_variables:
    corr, p = spearmanr(X[feature], y)
    spearman_results.append({
        'Feature': feature,
        'Spearman Correlation': corr,  # 符号付き
        'p-value': round(p, 8)
    })

spearman_df = pd.DataFrame(spearman_results)

# 絶対値でソートしてトップ5を取得
spearman_top5 = spearman_df.reindex(
    spearman_df['Spearman Correlation'].abs().sort_values(ascending=False).index
).head(5)

print("\nスピアマン相関による特徴量重要度（トップ5）:")
print(spearman_top5[['Feature', 'Spearman Correlation', 'p-value']])

### Kendalls Tauによる特徴量重要度の計算
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

# 絶対値でソートしてトップ5を取得
kendall_top5 = kendall_df.reindex(
    kendall_df['Kendall Tau'].abs().sort_values(ascending=False).index
)

print("\nKendall's Tauによる特徴量重要度（トップ5）:")
print(kendall_top5[['Feature', 'Kendall Tau', 'p-value']])

# Random Forest モデルの構築
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# 特徴量重要度の取得
rf_importances = pd.Series(rf.feature_importances_, index=feature_variables)
rf_top5 = rf_importances.sort_values(ascending=False)

print("\nRandom Forest による特徴量重要度（トップ5）:")
print(rf_top5)

from xgboost import XGBClassifier

# XGBoostモデルの構築と訓練
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X, y)

# 特徴量重要度の取得
xgb_importances = pd.Series(xgb.feature_importances_, index=feature_variables)
xgb_top5 = xgb_importances.sort_values(ascending=False)

print("\nXGBoost による特徴量重要度（トップ5）:")
print(xgb_top5)

# SVMのモデル構築
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# パイプラインの作成
svc = make_pipeline(StandardScaler(), SVC(kernel='linear', random_state=42))
svc.fit(X, y)

# 重みの取得
svc_weights = pd.Series(svc.named_steps['svc'].coef_[0], index=feature_variables)
svc_top5 = svc_weights.abs().sort_values(ascending=False)

print("\nSVM による特徴量重要度（トップ5）:")
print(svc_top5)

# Lasso回帰のモデル構築
from sklearn.linear_model import LogisticRegression

# パイプラインの作成
lasso = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', random_state=42))
lasso.fit(X, y)

# 重みの取得
lasso_weights = pd.Series(lasso.named_steps['logisticregression'].coef_[0], index=feature_variables)
lasso_top5 = lasso_weights.abs().sort_values(ascending=False)
print("\nLasso回帰による特徴量重要度（トップ5）:")
print(lasso_top5)

# Naive Bayesのモデル構築
from sklearn.naive_bayes import GaussianNB

# パイプラインの作成
nb = make_pipeline(StandardScaler(), GaussianNB())

# モデルの訓練
nb.fit(X, y)

# 重みの取得
nb_weights = pd.Series(nb.named_steps['gaussiannb'].theta_[0], index=feature_variables)
nb_top5 = nb_weights.abs().sort_values(ascending=False)

print("\nNaive Bayes による特徴量重要度（トップ5）:")
print(nb_top5)

# 各手法のトップ5をランキング順に取得
top_features_rf = rf_importances.sort_values(ascending=False).index.tolist()
top_features_chi2 = chi2_top5['Feature'].tolist()
top_features_spearman = spearman_top5['Feature'].tolist()

# ランキング1から5までのリストを作成
rank = [1, 2, 3, 4, 5]

# 各ランキングに対応する特徴量を取得
features_rf_ranked = rf_importances.sort_values(ascending=False).index.tolist()
features_chi2_ranked = chi2_top5['Feature'].tolist()
features_spearman_ranked = spearman_top5['Feature'].tolist()

# 統合データフレームの作成

rank_table = pd.DataFrame({
    'Rank': rank,
    'Random Forest Importance': features_rf_ranked,
    'Chi-Squared Statistic': features_chi2_ranked,
    'Spearman Correlation': features_spearman_ranked
})

# インデックスをRankに設定
rank_table = rank_table.set_index('Rank')

print("\n各手法のトップ5特徴量を統合した結果:")
print(rank_table)

# 各手法のトップ5をランキング順に取得
top_features_rf = rf_importances.sort_values(ascending=False).index.tolist()
top_features_xgb = xgb_importances.sort_values(ascending=False).index.tolist()
top_features_svc = svc_weights.abs().sort_values(ascending=False).index.tolist()
top_features_lasso = lasso_weights.abs().sort_values(ascending=False).index.tolist()
top_features_nb = nb_weights.abs().sort_values(ascending=False).index.tolist()

# ランキング1から5までのリストを作成
rank = list(range(1, 6))

# 統合データフレームの作成
df_rank = pd.DataFrame({
    'Rank': rank,
    'Random Forest': top_features_rf,
    'XGBoost': top_features_xgb,
    'Lasso': top_features_lasso,
    'SVM': top_features_svc,
    'Naive Bayes': top_features_nb
})

print("\n各手法による特徴量ランキング:")
print(df_rank)

from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIFの計算関数
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values(by='VIF', ascending=False)

# 最初のVIF計算
vif_data = calculate_vif(X)
print("\n初回のVIF:")
print(vif_data)

