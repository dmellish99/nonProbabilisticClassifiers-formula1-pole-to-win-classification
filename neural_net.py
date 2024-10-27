

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler



df=pd.read_csv("cleaned.csv")

## remove rows containing null values

df=df.replace(np.nan,None)
df=df[~df.isnull().all(1)]


x=df.iloc[:,:-1]

x_cols=x.columns

## for x select all columns except the last
X=df.iloc[:,:-1].to_numpy()



## for y select the last

y=df.iloc[:,-1].to_numpy()



scaler=StandardScaler()

scaler.fit(X)


X_scaled=scaler.transform(X)




### Neural network


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20)


model = Sequential()

powers_of_two=[2,4,16,32,64]

# Input layer with 2 hidden layers

model.add(Dense(units=17, activation='relu', input_dim=17))

model.add(Dense(units=powers_of_two[2], activation='relu'))

model.add(Dense(units=powers_of_two[1], activation='relu'))

model.add(Dense(units=1, activation='relu'))

model.compile(optimizer=Adam(learning_rate=.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(
    x=X_train,
    y=y_train,
    batch_size=4,
    epochs=100,
)

# Summary of the model
model.summary()


predictions=model.predict(X_test)

## evaluate how the model performs

from sklearn.metrics import confusion_matrix
confusion_matrix(predictions,y_test)



