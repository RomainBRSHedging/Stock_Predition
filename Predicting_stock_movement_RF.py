#Predicting_stock_movement_RF.py

#Classic librairies
import yfinance as yf
import datetime as dt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff

#ML libriaries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Market Data
stocks_list = ['AAPL']#, 'MSFT', 'AMZN', 'META', 'DIS']
end = dt.datetime.now()
start = '2020-11-01'
df = yf.download(stocks_list, start=start, end=end)
df['P Change'] = df['Close'].diff()
del df['Adj Close']

#Preprocess
up_df, down_df = df['P Change'].copy(), df['P Change'].copy()
up_df.loc['P Change'] = up_df.loc[  up_df < 0 ] = 0
down_df.loc['P Change'] = down_df.loc[  down_df> 0  ] = 0
down_df = down_df.abs()



######Indicators########
n = 14

    # 1) Relative Strength Index/Exponential Weighted Moving Average
ewma_up = up_df.transform((lambda x: x.ewm(span = n).mean()))
ewma_down = down_df.transform((lambda x: x.ewm(span = n).mean()))
RS = ewma_up/ewma_down
RSI = 100.0 - (100.0 / (1.0 + RS) )

df['Down_days'] = down_df
df['Up_days'] = up_df
df['RSI'] = RSI

    # 2) Stochastique Oscillator
low_14, high_14 = df['Low'].copy(), df['High'].copy()
low_14 = low_14.transform(lambda x: x.rolling(window = n).min())
high_14 = high_14.transform(lambda x: x.rolling(window = n).max())
Sto_O = 100.0*( (df['Close'] - low_14) /(high_14 - low_14))

df['Low_14'] = low_14
df['High_14'] = high_14
df['Sto_O'] = Sto_O

    # 3) William % R 
R = (high_14 - df['Close'])/(high_14 - low_14) *-100.0
df['W_R'] = R

    # 4) Moving Average Convergence Divergence  
ema_26 = df['Close'].transform(lambda x: x.ewm(span = 26).mean())
ema_12 = df['Close'].transform(lambda x: x.ewm(span = 12).mean())
macd = ema_12 - ema_26
ema_9_macd = macd.ewm(span = 9).mean()

df['MACD'] = macd
df['MACD_EMA'] = ema_9_macd

    # 5) Price Rate Of Change 
n_ = 9
df['PROC'] = df['Close'].transform(lambda x: x.pct_change(periods = n))



#####Model ###### 

    #Prediciton column
close_groups = df['Close'].transform(lambda x: np.sign(x.diff()))
df['Prediction'] = close_groups
df.loc[df['Prediction']== 0] = 1
df = df.dropna()

    #Build the Model: Random Forest
X_col = df[['RSI', 'Sto_O', 'W_R', 'MACD', 'MACD_EMA', 'PROC']]
Y_col = df['Prediction']
# X_col = X_col.reset_index(drop=True)
# Y_col = Y_col.reset_index(drop=True)
X_train, X_test, Y_train, Y_test = train_test_split(X_col, Y_col ,random_state=40)

    #Model
model = RandomForestClassifier(n_estimators=200, oob_score=True, criterion='gini', random_state=40)
model.fit(X_train, Y_train)

     #Prediction
Y_pred = model.predict(X_test)

# print(accuracy_score(Y_test, Y_pred, normalize=True)*100.0)

    #Evaluation
target_names = ['Up day', 'Down day']
report = classification_report(y_true=Y_test, y_pred=Y_pred,target_names=target_names , output_dict=True)
report = pd.DataFrame(report).transpose()

cm = confusion_matrix(Y_test, Y_pred)

print(report)
fig = ff.create_annotated_heatmap(z=cm, x=['D Day', 'U Day'], y=['Up Day', 'Down Day'])
fig.update_layout(title_text='Confusion Matrix')
fig.show()
