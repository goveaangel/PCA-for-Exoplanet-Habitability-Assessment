import pandas as pd
import numpy as np

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocessing and modeling
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import zscore
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_excel("data/Exoplanets2.xlsx")

df.describe()
df.info()

exo_pl_name = df["pl_name"]
df = df.drop(columns=["pl_name", "Unnamed: 0"])

sns.heatmap(df.corr(), annot = True, cmap = "coolwarm",annot_kws = {"size": 5})
plt.title("Mapa de calor de correlaciones entre variables")
plt.show()

(df.duplicated()).sum()

for column in df.columns:
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribución de {column}")
    plt.show()

z_scores = zscore(df)
outliers = (z_scores > 3).sum(axis=0)
outliers

selector = VarianceThreshold(threshold=0.01)
df_reduced = selector.fit_transform(df)
df_reduced

results_pca = pd.read_excel("data/results_exoplanets.xlsx")

# modelo 1 (datos originales)
# -----------------------------------------------------------------------------------

X = df
y = results_pca["habitability_score"]

X_train, X_test, y_train, y_test = train_test_split(X,y.values.reshape(-1,1),
    train_size = 0.8,
    random_state = 512,
    shuffle = True
    )

X.columns
X.info()

exo_planets_train = pd.DataFrame(np.hstack((X_train, y_train)),columns=['sy_snum', 'sy_pnum', 'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmasse', 'pl_orbeccen', 'ttv_flag', 'st_teff', 'st_rad', 'st_mass', 'st_logg', 'sy_vmag', 'sy_kmag', "habitability_score"])

modelf = smf.ols(formula = "habitability_score ~ sy_snum + sy_pnum + pl_orbper + pl_orbsmax + pl_rade + pl_bmasse + pl_orbeccen + ttv_flag + st_teff + st_rad + st_mass + st_logg + sy_vmag + sy_kmag", data = exo_planets_train)
modelf = modelf.fit()
modelf.summary()

modelf_noint = smf.ols(formula = "habitability_score ~ 0 + sy_snum + sy_pnum + pl_orbper + pl_orbsmax + pl_rade + pl_bmasse + pl_orbeccen + ttv_flag + st_teff + st_rad + st_mass + st_logg + sy_vmag + sy_kmag", data = exo_planets_train)
modelf_noint = modelf_noint.fit()
modelf_noint.summary()

modelf_noint = smf.ols(formula = "habitability_score ~ 0 +sy_snum + sy_pnum + pl_orbper + pl_orbsmax + pl_rade + pl_bmasse + pl_orbeccen + ttv_flag + st_teff + st_mass + st_logg + sy_vmag + sy_kmag", data = exo_planets_train)
modelf_noint = modelf_noint.fit()
modelf_noint.summary()

modelf_noint = smf.ols(formula = "habitability_score ~ 0 +sy_snum + sy_pnum + pl_orbsmax + pl_rade + pl_bmasse + pl_orbeccen + ttv_flag + st_teff + st_mass + st_logg + sy_vmag + sy_kmag", data = exo_planets_train)
modelf_noint = modelf_noint.fit()
modelf_noint.summary()

modelf_noint.conf_int(alpha=0.05)

# Análisis residual modelo 1

y_train = y_train.flatten()
prediction_train = modelf_noint.predict(exog = X_train)
residues_train   = prediction_train - y_train

residues_train.sum()
residues_train.mean()
residues_train.min(),residues_train.max()


fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_train, prediction_train, edgecolors=(0, 0, 0), alpha = 0.4)
ax.plot([y_train.min(), y_train.max()], [prediction_train.min(), prediction_train.max()],
                'r', color = 'red', lw=1)
ax.set_title('Predicted value vs real value', fontsize = 10, fontweight = "bold")
ax.set_xlabel('Real')
ax.set_ylabel('Prediction')
ax.tick_params(labelsize = 7)

fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(
    data    = residues_train,
    stat    = "density",
    kde     = True,
    line_kws= {'linewidth': 1},
    color   = "firebrick",
    alpha   = 0.3,
)
ax.set_title('Distribution of residues', fontsize = 10,
                     fontweight = "bold")
ax.set_xlabel("Residue")
ax.tick_params(labelsize = 7)

fig, ax = plt.subplots(figsize=(6, 6))
sm.qqplot(
    residues_train,
    fit   = True,
    line  = 'q', 
    color = 'firebrick',
    alpha = 0.4,
    lw    = 2
)
ax.set_title('Q-Q residues', fontsize = 10, fontweight = "bold")
ax.tick_params(labelsize = 7)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(prediction_train, residues_train,
                   edgecolors=(0, 0, 0), alpha = 0.4)
ax.axhline(y = 0, color = 'red', lw=2)
ax.set_title('Residues vs prediction', fontsize = 10, fontweight = "bold")
ax.set_xlabel('Prediction')
ax.set_ylabel('Residue')
ax.tick_params(labelsize = 7)

shapiro_test = stats.shapiro(residues_train)
shapiro_test

k2, p_value = stats.normaltest(residues_train)
print(f"Statistic= {k2}, p-value = {p_value}")

predictions = modelf_noint.get_prediction(exog = X_train).summary_frame(alpha=0.05)
predictions.head(4)

predictions = modelf_noint.predict(exog = X_test[list(X_train.columns)])
mse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predictions
       )
print("")
rmse = np.sqrt(mse)
print(f"The test error (rmse) is: {rmse}")

y.min(),y.mean(),y.max()

X.describe()

# Modelo 2
# ----------------------------------------------------------------------------------

results_pca
X2 = results_pca.drop(columns = ["Unnamed: 0","habitability_score","habitability_class"])

X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y.values.reshape(-1,1),
    train_size = 0.8,
    random_state = 512,
    shuffle = True
    )

exo_planets2_train = pd.DataFrame(np.hstack((X2_train, y2_train)),columns=["PC1_contribution","PC2_contribution","PC3_contribution","PC4_contribution","PC5_contribution","PC6_contribution","PC7_contribution","PC8_contribution","PC9_contribution", "habitability_score"])

modelf2 = smf.ols(formula = "habitability_score ~ PC1_contribution + PC2_contribution + PC3_contribution + PC4_contribution + PC5_contribution + PC6_contribution + PC7_contribution + PC8_contribution + PC9_contribution", data = exo_planets2_train)
modelf2 = modelf2.fit()
modelf2.summary()
