import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('krediVeriseti.csv')

dataset['krediMiktari'].fillna(0, inplace=True)

dataset['yas'].fillna(dataset['yas'].mean(), inplace=True)

X = dataset.iloc[:, :5]

#Ev durumunu sadeleştirme
def ev_durumunu_sadelestir(param):
    degistir = {'evsahibi':1,'kiraci':2}
    return degistir[param]

X['evDurumu'] = X['evDurumu'].apply(lambda x : ev_durumunu_sadelestir(x))

#Telefon durumunu sadeleştirme
def tel_durumunu_sadelestir(param):
    degistir = {'var':1,'yok':2}
    return degistir[param]

X['telefonDurumu'] = X['telefonDurumu'].apply(lambda x : tel_durumunu_sadelestir(x))

#Kredi durumunu sadeleştirme
def kredi_durumunu_sadelestir(param):
    degistir = {'krediver':1,'verme':2}
    return degistir[param]



y = dataset['KrediDurumu']

y = y.apply(lambda x : kredi_durumunu_sadelestir(x))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
knn=classifier.fit(X_train, y_train)

print("Eğitim Sonucu: {:.2f}".format(knn.score(X_train, y_train)))
print("Test sonucu: {:.7f}".format(knn.score(X_test, y_test)))


pickle.dump(classifier, open('modelM4A.pkl','wb'))

# Sonucları alıyoruz
model = pickle.load(open('modelM4A.pkl','rb'))
print(model.predict([[12500,62,2,0,2]]))