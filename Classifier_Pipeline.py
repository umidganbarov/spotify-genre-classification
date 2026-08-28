import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#!MODELS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df_original=pd.read_csv('dataset.csv')#spotify dataset
df=df_original.drop(columns=['n','track_id','artists','album_name','track_name'])
df.explicit=df.explicit.astype('int')


df=df[df.track_genre.isin(['acoustic','classical','electronic','electro','guitar','k-pop','study','turkish'])]
df
x=df.iloc[:,:15]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=12)
#more imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
pipe1=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7))
pipe2=make_pipeline(StandardScaler(),LogisticRegression(random_state=12))
pipe3=make_pipeline(StandardScaler(),DecisionTreeClassifier(max_depth=5,random_state=12))
pipe4=make_pipeline(StandardScaler(),RandomForestClassifier(random_state=12))
pipelines={
    'knn':pipe1,
    'logr':pipe2,
    'dtree':pipe3,
    'rfc':pipe4
}
for name,pipe in pipelines.items():
    print(f"{name} with accuracy: {(pipe.fit(x_train,y_train).score(x_test,y_test)*100):.2f}%, CVScore: {(cross_val_score(pipe,x,y,cv=10).mean()*100):.2f}%")

