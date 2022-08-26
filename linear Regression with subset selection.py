import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import sns as sns
from pandas.compat import numpy
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import  random as rm

# for both the subset selection methods, we have taken a confidence level of 95%.
def forward(datatest, target, significance_level=0.05):
      data=pd.DataFrame(datatest)
      initial_features = data.columns.tolist()
      best_features = []
      while (len(initial_features) > 0):
            remaining_features = list(set(initial_features) - set(best_features))
            new_pval = pd.Series(index=remaining_features)
            #This loop takes each feature at a time and add it to the previous best features if any and given as input to the
            #Ordiary Least Squares(OLS()) fucntion to get the p-values.
            for new_column in remaining_features:
                  model = sm.OLS(target, sm.add_constant(data[best_features + [new_column]])).fit()
                  new_pval[new_column] = model.pvalues[new_column]
            min_p_value = new_pval.min()
            # the feature with the minimum p-value is conisders as the best feature and is added to the list.
            if (min_p_value < significance_level):
                  best_features.append(new_pval.idxmin())
            else:
                  break
            #the loop continues until the min_p_value is no more less than the significance_value.
          # the following code is used to select substes of (1,2,3,...) from the best features and MSE and R2 is calculated.
      t=0;
      MSE_fw={}
      r2_fw={}
      while t < len(best_features):
            if t==0:
                  random_x = (pd.DataFrame(datatest).iloc[:,t])
            else:
                  random_x = pd.DataFrame(datatest).iloc[:,best_features[0]]
                  uh = pd.DataFrame()
                  for i in range(1,t+1):
                        uh = pd.DataFrame(datatest).iloc[:,best_features[i]]
                        random_x = pd.DataFrame(random_x).join(uh)

            if t==0:
                  x_train, x_test, y_train, y_test = train_test_split(random_x, target, test_size=0.2, random_state=0)
                  regressor = LinearRegression()
                  regressor.fit(x_train.values.reshape(-1,1), y_train)
                  y_predict = regressor.predict(x_test.values.reshape(-1,1))
                  MSE1, r21 = metrics(y_test, y_predict)
                  MSE_fw[t]=MSE1
                  r2_fw[t]=r21
            else:
                  x_train, x_test, y_train, y_test = train_test_split(random_x, target, test_size=0.2, random_state=0)
                  regressor = LinearRegression()
                  regressor.fit(x_train, y_train)
                  y_predict = regressor.predict(x_test)
                  MSE1, r21 = metrics(y_test, y_predict)
                  MSE_fw[t] = MSE1
                  r2_fw[t] = r21


            t+=1
      #the best features and MSE and r2 values for each subset it returned to plot the graph.
      return best_features,MSE_fw,r2_fw
      pass

def backward(datatest, target,significance_level = 0.05):
      data = pd.DataFrame(datatest)
      features = data.columns.tolist()
      #the while loop takes all the fatures of the dataset and gives it as the input to the OLS() to get the p_values.
      #the value with the p-vlaue higher than the significance_level is removed.
      while (len(features) > 0):
            features_with_constant = sm.add_constant(data[features])
            p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
            max_p_value = p_values.max()
            if (max_p_value >= significance_level):
                  excluded_feature = p_values.idxmax()
                  features.remove(excluded_feature)
            else:
                  break
      t = 0;
      MSE_bw = {}
      r2_bw = {}
      # the following code is used to select substes of (1,2,3,...) from the best features and MSE and R2 is calculated.
      while t < len(features):
            if t == 0:
                  random_x = (pd.DataFrame(datatest).iloc[:, t])
            else:
                  random_x = pd.DataFrame(datatest).iloc[:, features[0]]
                  uh = pd.DataFrame()
                  for i in range(1, t + 1):
                        uh = pd.DataFrame(datatest).iloc[:,features[i]]
                        random_x = pd.DataFrame(random_x).join(uh)

            if t == 0:
                  x_train, x_test, y_train, y_test = train_test_split(random_x, target, test_size=0.2, random_state=0)
                  regressor = LinearRegression()
                  regressor.fit(x_train.values.reshape(-1, 1), y_train)
                  y_predict = regressor.predict(x_test.values.reshape(-1, 1))
                  MSE1, r21 = metrics(y_test, y_predict)
                  MSE_bw[t] = MSE1
                  r2_bw[t] = r21

            else:
                  x_train, x_test, y_train, y_test = train_test_split(random_x, target, test_size=0.2, random_state=0)
                  regressor = LinearRegression()
                  regressor.fit(x_train, y_train)
                  y_predict = regressor.predict(x_test)
                  MSE1, r21 = metrics(y_test, y_predict)
                  MSE_bw[t] = MSE1
                  r2_bw[t] = r21


            t += 1
      return features,MSE_bw,r2_bw

      pass
#In the following function 5 features are selcted randomly from the dataset and a model is generated.
def random():
      diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
      columns=[]
      #the for loop is used to pick some random features.
      for col in pd.DataFrame(diabetes_X).columns:
            columns.append(col)
      random_coloums=rm.sample(columns,5)
      random_x=pd.DataFrame(diabetes_X).iloc[:,random_coloums[0]]
      uh = pd.DataFrame()
      for i in range(1,5):
            uh=pd.DataFrame(diabetes_X).iloc[:, random_coloums[i]]
            random_x =pd.DataFrame(random_x).join(uh)
      x_train, x_test, y_train, y_test = train_test_split(random_x, diabetes_y, test_size=0.2, random_state=0)
      regressor = LinearRegression()
      regressor.fit(x_train, y_train)
      y_predict = regressor.predict(x_train)
      MSE_rn,r2_rn=metrics(y_train,y_predict)
      return  MSE_rn,r2_rn
      pass

def train(x_train,y_train): #training the algorithm.
      regressor = LinearRegression()
      regressor.fit(x_train, y_train)
      return  regressor

def process_data(diabetes_X, diabetes_y): #spliting the dataset into train and test set
      x_train, x_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.2,random_state=0)
      return x_train, x_test, y_train, y_test
      pass

def plot(MSE_FW,R2_FW,MSE_BW,R2_BW): #ploting the MSE with respect to the subsets of features.
      plt.plot(MSE_FW.keys(), MSE_FW.values(), color='red', label="forword")
      plt.plot(MSE_BW.keys(),MSE_BW.values(), color='blue',label="forword")
      plt.scatter(MSE_FW.keys(), MSE_FW.values(), color='red')
      plt.scatter(MSE_BW.keys(), MSE_BW.values(), color='blue')
      plt.xlabel("subsets")
      plt.ylabel("MSE")
      plt.legend()
      plt.show()
      # plt2.plot(R2_FW.keys(), R2_FW.values(), color='red')
      # plt2.plot(R2_BW.keys(), R2_BW.values(), color='blue')
      # plt2.scatter(R2_FW.keys(), R2_FW.values(), color='red')
      # plt2.scatter(R2_BW.keys(), R2_BW.values(), color='blue')
      # plt2.show()
      pass

def metrics(y_test,y_predict): #calculating the MSE and r2.
      MSE= mean_squared_error(y_test,y_predict)
      r2 = r2_score(y_test,y_predict)
      return MSE,r2

      pass

def main():
      diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
      x_train, x_test, y_train, y_test= process_data(diabetes_X, diabetes_y)
      regressor = train(x_train, y_train)
      y_predict = regressor.predict(x_train)
      MSE, r2 = metrics(y_train, y_predict)
      fw_bestfeaturs,MSE_FW,R2_FW = forward(x_train,y_train)
      print("the best featurs for forword subset selection method",fw_bestfeaturs)
      bw_bestfeaturs,MSE_BW,R2_BW= backward(x_train,y_train)
      print("the best featurs for backword subset selection method", bw_bestfeaturs)
      plot(MSE_FW,R2_FW,MSE_BW,R2_BW)
      plt3.plot(x_train[:,2], y_predict, 'o')
      m, b = np.polyfit(x_train[:,2], y_train, 1)
      plt3.plot(x_train[:,2], m * x_train[:,2] + b)
      plt3.show()
      plt2.plot(x_train[:, 3], y_predict, 'o')
      m, b = np.polyfit(x_train[:, 3], y_train, 1)
      plt2.plot(x_train[:, 3], m * x_train[:, 3] + b)
      plt2.show()
      MSE_rn,r2_rn=random()
      print("the MSE for random algorithm is ",MSE_rn)
      print("the r2 for random algorithm is ", r2_rn)
      pass


if __name__ == '__main__':
      main()

