from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

digits = load_digits()
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
print(X[0, :])
