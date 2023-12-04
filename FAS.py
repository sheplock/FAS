import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# read in data as pandas dataframe
df = pd.read_csv('fas_2024.csv')

# run multiple linear regression to fit model
# for more information on how these specific variables were chosen
# please see the Jupyter notebook in this repository
model = sm.OLS.from_formula('three_pct_season ~ three_non_cnr_pct_oct_nov + three_cnr_pct_oct_nov + ft_pct_oct_nov', data = df).fit()
print(model.summary())
fitted = model.predict(df)

# make plots to see accuracy of model
# and residual histogram
plt.scatter(fitted, df.three_pct_season,label='Prediction results')
plt.xlim(0.25,0.45)
plt.plot(np.arange(0.25,0.45,0.01), np.arange(0.25,0.45,0.01), label='y=x')
plt.xlabel('Predicted 3 point FG%')
plt.ylabel('Actual 3 point FG%')
plt.legend()
plt.show()
plt.savefig("pred_vs_actual.png",format="png")
plt.clf()

residuals = df.three_pct_season - fitted
plt.hist(residuals)
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()
plt.savefig("residuals.png",format="png")

print("RMS Error = {}".format(np.sqrt(np.mean(np.square(df.three_pct_season - fitted)))))
