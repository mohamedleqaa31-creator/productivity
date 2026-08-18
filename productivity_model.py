import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
sleep_hours =np.array([4,5,6,7,8,9,5,6,7,8])
study_hours =np.array([2,3,4,5,6,7,2,3,5,6])
screen_time =np.array([8,7,6,4,3,2,9,6,4,3])
breaks =np.array([2,2,3,4,4,5,1,3,4,4])
productivity =np.array([45,50,60,72,82,90,42,58,75,84])

df =pd.DataFrame({
  "sleep_hours":sleep_hours,
"study_hours":study_hours,
"screen_time":screen_time,
"breaks":breaks,
"productivity":productivity })
print(df) #هنا عشان اطبع كل االداتا فريم
print(df.head()) #بردو عشان اطبع عدد معين من الصفوف بس من الاول
print(df.shape) # هنا عشان اعرف شكل المصفوفه كام فى كان من حيث عدد الاعمده و الصفوف
df.info() # بتدينا معلومات عن كل الداتا فرين 
print(df.describe()) # ودى بقا احصاءيات للداتا اللى معايا زى المين و الميدن 
#هنا استخدمت مكتبه لاماتبلوت عشان الرن البيانى
plt.scatter(df["sleep_hours"] , df["productivity"])
plt.plot(sleep_hours, productivity, color='green', linestyle='--')
plt.xlabel("sleep hour") 
plt.ylabel("productivity")
plt.title("sleep hour vs productivity")
plt.show()
#هنا بدات استخدم مكتبه السي بورن وجواها ماتبلوت عشان اعمل الهيت ماب
sns.heatmap(df.corr(), annot=True ,cmap="Blues")
plt.title("Correlation Between Variables")
plt.show()
#هنا هنعمل فصل لاللداتا اللى الموديل هيتعلم منها و اللى هيتوقعها
X=df[["study_hours","sleep_hours","screen_time","breaks"]]
Y=df[["productivity"]]

#هنا نبدا نعمل تريننج ةتيست
X_train,X_test,Y_train,Y_test = train_test_split(X,
                                                Y,
                                                test_size=0.2,
                                                random_state=42)
print(X_train.shape)
print(X_test.shape)
#هنبدا نستخدم ال تروش هنستخدم منها التينسور
X_train_tensor=torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor=torch.tensor(X_test.values, dtype=torch.float32)
Y_train_tensor=torch.tensor(Y_train.values, dtype=torch.float32)
Y_test_tensor=torch.tensor(Y_test.values, dtype=torch.float32)
# هنبدا نستعى الحاجات اللى هنسخدمها ف المودل زى الnn
class ProductivityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)

model = ProductivityModel()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5000):
    y_pred = model(X_train_tensor)

    loss = criterion(y_pred, Y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(epoch, loss.item())

with torch.no_grad():
    predictions = model(X_test_tensor)

print("Predictions:")
print(predictions)

print("Actual:")
print(Y_test_tensor)


y_mean = torch.mean(Y_test_tensor)

ss_total = torch.sum((Y_test_tensor - y_mean) ** 2)

ss_res = torch.sum((Y_test_tensor - predictions) ** 2)

r2 = 1 - (ss_res / ss_total)

print("R²:", r2.item())

new_data = torch.tensor([[6, 8,10,]], dtype=torch.float32)

with torch.no_grad():
    new_prediction = model(new_data)

print("Predicted Productivity:", new_prediction.item())
         
 