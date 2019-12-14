from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


digits = load_digits()
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
y = digits.target   # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4) 
print(X[0, :])
