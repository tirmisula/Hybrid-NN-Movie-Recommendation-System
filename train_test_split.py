import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data = pd.read_csv("ratings.csv")
    x,y = data.iloc[:,0:4], data.iloc[:,0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print(len(x_train))
    print(len(x_test))
    x_train.to_csv('trainset.csv',index=False)
    x_test.to_csv('testset.csv',index=False)
