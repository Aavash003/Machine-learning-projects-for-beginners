
##DIGIT PREDICTIONS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

dig=load_digits()

##digits
plt.gray()
for i in range(4):
    plt.matshow(dig.images[i])
print(i)


print(dig.keys())
print(dig.data)
print(dig.target)
print('features_names',dig.feature_names)
print(dig.feature_names[0])
print(dig.target_names)
print(dig.DESCR)
print('\n')

df=pd.DataFrame(dig.data,columns=dig.feature_names)
print(df.head())
print(df.shape)
print()

df['target']=dig.target
x=df.values
y=df.target.values
print(df)
trainx,testx,trainy,testy=train_test_split(x,y,test_size=.2)
svm=SVC(C=100,gamma=15,kernel='poly')#kernel=linear,rbf,poly
svm.fit(trainx,trainy)
predy=svm.predict(trainx)
print(predy)
print(trainy)

mat=confusion_matrix(trainy,predy)
print(mat)
print(len(trainy))
mse=mean_squared_error(trainy,predy)
print("Accuracy=",(1-mse)*100,"%")

sum=0
for i in range(len(trainy)):
    diff=float((trainy[i]-predy[i])**2)
    sum+=diff
    error=float(sum/len(trainy))
print('Error=',error*100,'%')



