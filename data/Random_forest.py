from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np 

np.concatenate() #doe hier je moleculen en eiwitten
X_train, X_test, y_train, y_test = train_test_split(test_size=0.2, random_state=42)


rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=None
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

