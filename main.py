import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
 
def simulate_wifi_fingerprints(n_aps=6, n_locs=10, n_samples=200):
    np.random.seed(42)
    X,y=[],[]
    loc_profiles=np.random.randint(-90,-40,(n_locs,n_aps))
    for loc in range(n_locs):
        for _ in range(n_samples//n_locs):
            rssi=loc_profiles[loc]+np.random.randint(-5,6,n_aps)
            X.append(rssi); y.append(loc)
    return np.array(X),np.array(y)
 
def wknn(X_train, y_train, query, k=3):
    dists=np.linalg.norm(X_train-query, axis=1)
    idx=np.argsort(dists)[:k]
    weights=1/(dists[idx]+1e-8)
      from collections import Counter
    vote=Counter()
    for i,w in zip(idx,weights): vote[y_train[i]]+=w
    return vote.most_common(1)[0][0]
 
X,y=simulate_wifi_fingerprints()
scaler=StandardScaler(); X_s=scaler.fit_transform(X)
knn=KNeighborsClassifier(n_neighbors=3,weights='distance')
rf=RandomForestClassifier(n_estimators=100,random_state=42)
for name,model in [("WKNN",knn),("Random Forest",rf)]:
    scores=cross_val_score(model,X_s,y,cv=5)
    print(f"{name:15s}: Accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
