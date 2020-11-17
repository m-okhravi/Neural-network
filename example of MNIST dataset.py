from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# this code will print image of a digit in 8*8 matrix pixel
digits = load_digits()
print(digits.data.shape)
plt.gray()
plt.matshow(digits.images[1])
plt.show()
