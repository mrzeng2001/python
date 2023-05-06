import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.tsa.stattools as st

df = pd.read_csv('C:\\Users\\Administrator\\Desktop\\data.csv')  # 导入数据

print(df)

data = df['CHNCPIALLMINMEI'].values
print(data)
year = df['DATE'].values
print(year)
ts = pd.DataFrame(data, index=year, columns=['CHNCPIALLMINMEI'])
print(ts)
dp = pd.Series(data, index=year)

res = sm.tsa.arma_order_select_ic(
    ts,
    max_ar=2,
    max_ma=2, ic=["aic", "bic"]
)  # 定阶
print(res)
mod = ARIMA(ts, order=(2, 0, 2))
res = mod.fit()
print(res.summary())
fig = plt.figure(figsize=(16, 9))
fig = res.plot_diagnostics(fig=fig, lags=30)
plt.savefig("tsplot2.png", dpi=600, bbox_inches = 'tight')
plt.show()

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
fig = sm.graphics.tsa.plot_acf(dp.values, lags=10, ax=ax1)
plt.savefig("acf.png", dpi=600, bbox_inches='tight')
plt.show()  # acf图

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
fig = sm.graphics.tsa.plot_pacf(dp.values, lags=10, ax=ax1)
plt.savefig("pacf.png", dpi=600, bbox_inches='tight')
plt.show()  # pacf图

diff = dp.diff(1)
diff.dropna(inplace=True)
diff.plot(figsize=(12, 8), marker='o', color='black')
plt.show()   # 一阶差分
