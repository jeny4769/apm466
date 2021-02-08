#import the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

#read the data
df = pd.read_excel("466.xlsx")
df.set_index("Date", inplace = True)

coup = np.asarray([0.75,0.75,0.5,1.75,2.75,1.5,1.25,0.5,2.75,9])
months_to_mature = np.asarray([2,8,14,26,38,44,50,56,17,53]) 

#the coupons are paid each 6 months
terms_to_mature = months_to_mature // 6 + 1
months_to_next_payment = months_to_mature % 6
months_from_last_payment = 6 - months_to_next_payment

#gives the dirty price
for i in range(10):
    df[df.columns[i]] += coup * months_from_last_payment / 12
    

#compute yield by ridder
def half_year_yield_rate(yield_rate, months_to_next_payment , terms_to_mature, coupon, current_value):
    return yield_rate**(months_to_next_payment/6) * (coupon * (yield_rate**(terms_to_mature + 1) - 1)/(yield_rate - 1) + 100 * (yield_rate ** terms_to_mature)) - current_value


#new df to obtain yeild rate 
df_yield = df.iloc[:,:-1].copy()
for i in range(10):
    for j in range(10):
        df_yield.iloc[i,j] = (optimize.ridder(half_year_yield_rate, 
                                                  0.5, 0.999, #by observation, the rate should be in between [1,2]
                                                  args = (months_to_next_payment[i], terms_to_mature[i] ,coup[i]/2, df.iloc[i,j ]))
                                                     )**(-2)


df_yield["years_to_mature"] = months_to_mature/12
df_yield.sort_values("years_to_mature",inplace= True)
df_yield.set_index("years_to_mature", inplace = True)


pt = (df_yield - 1).plot(grid = True,figsize = (20,10),title = "yield curve")
pt.set_xlabel("time to maturity (yr)")
pt.set_ylabel("yield rate")



terms_to_mature


df_spot


df_spot = df_yield.copy()
for i in range(1,10):
    for j in range(10):
        months_to_mature_num = df_spot.index[i] * 12
        month_list = - np.arange(months_to_mature_num%6, months_to_mature_num, 6)/12
        start_point = i - len(month_list)
        append_spot_rate=False
        if start_point < 0:
            start_point = 0
            append_spot_rate = True
        spot_rate = df_spot.iloc[start_point :i,j]
        if append_spot_rate:
            spot_rate = np.append(spot_rate, df_spot.iloc[i - 1,j])
        pre = df.iloc[i,j] - coup[i]*(spot_rate**month_list).sum()/2
        df_spot.iloc[i,j] = np.power((100 + coup[i]/2)/pre , 1/df_spot.index[i])


pt = (df_spot - 1).plot(grid = True,figsize = (20,10),title = "spot curve")
pt.set_xlabel("time to maturity (yr)")
pt.set_ylabel("spot rate")



# compute the forward
yd = df_yield.values
forward = np.zeros((4,10))
for i in range(4):
    forward[i] = ((yd[i+1])**(i+2) / yd[0])**(1/(i+1))


df_forward = pd.DataFrame(forward)
df_forward.index = range(1,5)
df_forward.columns = df_yield.columns
pt2 = (df_forward - 1).plot(grid = True,figsize = (10,5),title = "forward curve")
pt2.set_xlabel("1 yr to i yr")
pt2.set_ylabel("foward rate")




# log return of yield
X = np.zeros((5,9))
for i in range(5):
    X[i] = np.log((np.asarray(df_yield.iloc[i,1:]) - 1)/(np.array(df_yield.iloc[i,:9]) - 1))
cov_X = np.cov(X)
print('cov for log return')
print(cov_X)


# log return of forward
log_forward = np.zeros((4,9))
for i in range(4):
    log_forward[i] = np.log((np.asarray(df_forward.iloc[i,1:]) - 1)/(np.array(df_forward.iloc[i,:9]) - 1))
cov_for = np.cov(log_forward)
print('cov for log forward')
print(cov_for)



#eigenvalue and eigenvectors
w_f, v_f = np.linalg.eig(cov_for)
w_X, v_X = np.linalg.eig(cov_X)



print ("eigenvalue and vectors for cov X")
print(w_X)
print(v_X)


print ("eigenvalue and vectors for cov forward")
print(w_f)
print(v_f)

