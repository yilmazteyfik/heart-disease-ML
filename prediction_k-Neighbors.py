import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Veri dosyasını okuma
heart_dataset = pd.read_csv("heart.csv")

# İlk 5 veriyi gösterme
print(heart_dataset.head())

# Hedef değişkenin dağılımını kontrol etme
print(heart_dataset["target"].value_counts())

# Özellikler ve hedef değişkeni ayırma
X = heart_dataset.drop(columns='target', axis=1)
Y = heart_dataset['target']

# Eğitim ve test verilerini ayırma
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2)

# k-NN modelini eğitme
k = 3  # k değeri
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, Y_train)

# Eğitim verisinde doğruluk
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(train_predictions, Y_train)
print('Eğitim verisinde doğruluk: ', train_accuracy)

# Test verisinde doğruluk
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(test_predictions, Y_test)
print('Test verisinde doğruluk: ', test_accuracy)

# Tahmin sistemi oluşturma
input_data = np.array([63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]).reshape(1, -1)
prediction = model.predict(input_data)
print(prediction)

if prediction[0] == 0:
    print("Kişide kalp hastalığı yok.")
else:
    print("Kişide kalp hastalığı var.")
