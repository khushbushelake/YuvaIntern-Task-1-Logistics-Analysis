import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df=pd.read_csv("data/logistics_orders.csv")
df["Capacity_Utilization_Pct"]=df["Load_Weight_KG"]/df["Vehicle_Capacity_KG"]*100
zone_kpis=df.groupby("Delivery_Zone").agg(Orders=("Order_ID","count"),On_Time_Rate=("On_Time","mean"),Avg_Delivery_Time=("Delivery_Time_Hours","mean"),Avg_Cost=("Delivery_Cost_INR","mean")).reset_index()
X=pd.get_dummies(df[["Distance_KM","Load_Weight_KG","Traffic_Level","Weather","Capacity_Utilization_Pct"]],drop_first=True)
y=df["Delivery_Time_Hours"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42)
model=LinearRegression().fit(X_train,y_train)
pred=model.predict(X_test)
print("MAE:",mean_absolute_error(y_test,pred),"R2:",r2_score(y_test,pred))
print(zone_kpis)
