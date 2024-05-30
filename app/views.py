import pickle
from flask import Flask,render_template,request
import numpy as np


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
rf_model = pickle.load(open('model.pkl', 'rb'))
knn = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def AnaSayfa():
    return render_template('index.html')



@app.route('/tahmin', methods=['GET', 'POST'])
def Tahmin():
    if request.method == 'POST':

     yas = int(request.form.get('yas'))

     cinsiyet = int(request.form.get('cinsiyet'))

     agriTipi = int(request.form.get('agriTipi'))

     kanBasinci = int(request.form.get('kanBasinci'))

     kolestrol = int(request.form.get('kolestrol'))

     kanSekeri = int(request.form.get('kanSekeri'))

     ekg = int(request.form.get('ekg'))

     thalach = int(request.form.get('thalach'))

     angina = int(request.form.get('angina'))

     eskiSt = float(request.form.get('eskiSt'))

     yeniSt = int(request.form.get('yeniSt'))

     floroskopi = int(request.form.get('floroskopi'))

     thal = int(request.form.get('thal'))

     
     dizi = np.array([[yas, cinsiyet, agriTipi, kanBasinci, kolestrol, kanSekeri, ekg, thalach, angina, eskiSt, yeniSt, floroskopi, thal ]])
     durum = model.predict(dizi)
     durum2 = rf_model.predict(dizi)
     durum3 = knn.predict(dizi)

     if durum == 0:
        sonuc = "lojistik regresyon modeline göre kalp hastalığı ihtimali düşük, sağlıklısınız."
        durum=0
        logistic_dizi = np.array([[yas, cinsiyet, agriTipi, kanBasinci, kolestrol, kanSekeri, ekg, thalach, angina, eskiSt, yeniSt, floroskopi, thal, durum ]])

     else:
        sonuc = "lojistik regresyon modeline göre kalp hastalığı ihtimali yüksektir, sağlığınıza dikkat etmelisiniz."
        durum=1
        logistic_dizi = np.array([[yas, cinsiyet, agriTipi, kanBasinci, kolestrol, kanSekeri, ekg, thalach, angina, eskiSt, yeniSt, floroskopi, thal, durum ]])

     if durum2 == 0:
        sonuc2 = "random forest modeline göre kalp hastalığı ihtimali düşük, sağlıklısınız."
        durum2=0
        Forest_dizi = np.array([[yas, cinsiyet, agriTipi, kanBasinci, kolestrol, kanSekeri, ekg, thalach, angina, eskiSt, yeniSt, floroskopi, thal, durum2 ]])

     else:
        sonuc2 = "random forest modeline göre kalp hastalığı ihtimali yüksektir, sağlığınıza dikkat etmelisiniz."
        durum2=1
        Forest_dizi = np.array([[yas, cinsiyet, agriTipi, kanBasinci, kolestrol, kanSekeri, ekg, thalach, angina, eskiSt, yeniSt, floroskopi, thal, durum2 ]])

     if durum3 == 0:
        sonuc3 = "k-nearest neighbors modeline göre kalp hastalığı ihtimali düşük, sağlıklısınız."
        
     else:
        sonuc3 = "k-nearest neighbors modeline göre kalp hastalığı ihtimali yüksektir, sağlığınıza dikkat etmelisiniz."
        

    with open('LogisticDataSet.csv', 'a') as f:np.savetxt(f, logistic_dizi, fmt='%d', delimiter=',') 
    with open('ForestDataSet.csv', 'a') as f:np.savetxt(f, Forest_dizi, fmt='%d', delimiter=',') 

    return render_template('tahmin.html', prediction_text='Kullanıcının {}'.format(sonuc),
                                           prediction_text2='Kullanıcının {}'.format(sonuc2),
                                             prediction_text3='Kullanıcının {}'.format(sonuc3))
    
        
if __name__=='__main__':
    app.run(debug=True)
