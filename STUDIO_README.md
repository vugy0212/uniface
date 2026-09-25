# 👤 UniFace Studio - Sustav za prepoznavanje i analizu lica

UniFace Studio je kompletna aplikacija s modernim grafičkim web sučeljem (**Gradio Web UI**) koja radi **100% lokalno** na vašem računalu, koristeći **UniFace v4.0.0**, **RetinaFace**, **ArcFace**, **FairFace** i **SQLite** bazu lica.

---

## 🚀 Brzo pokretanje

### Opcija 1: Windows (Dvoklik)
Dovoljno je pokrenuti:
* `pokreni.bat` ili `run_studio.bat`

### Opcija 2: Pokretanje putem terminala
```bash
# 1. Instalacija ovisnosti za Studio
pip install -r requirements_studio.txt

# 2. Pokretanje aplikacije
python run_studio.py
```
Aplikacija se automatski otvara u web pregledniku na adresi:  
👉 **`http://127.0.0.1:7860`**

---

## 🌟 Ključne mogućnosti

### 1. 👥 Baza Osoba (Unos, uređivanje i pretraga)
* **Unos pojedinačnih i grupnih fotografija:** Učitajte sliku s jednom ili više osoba. Sustav automatski detektira sva lica, numerira ih i omogućuje odabir točno onog lica koje želite spremiti u bazu.
* **Pretraga osoba:** Integrirana tražilica za brzo filtriranje osoba u bazi po imenu ili bilješci u stvarnom vremenu.
* **Brzi odabir iz tablice:** Klikom na osobu u tablici automatski se otvara njena galerija spremljenih uzoraka lica te se popunjava obrazac za brzo dodavanje novih fotografija.
* **Galerija i brisanje uzoraka:** Pregledajte sva registrirana lica po osobi i uklonite pojedinačne slike ili cijelu osobu jednim klikom.

### 2. 🔍 Prepoznavanje i analiza lica
* **Višestruko prepoznavanje:** Detektira i uspoređuje sva lica s fotografije prema biometrijskim vektorima iz baze (512-dimenzionalni ArcFace embedding).
* **Pregledni vizualni rezultati:**
  * 🟩 **Zeleni okvir:** Prepoznata osoba s imenom, postotkom točnosti i sigurnosnom marginom u odnosu na drugog najizglednijeg kandidata.
  * 🟥 **Crveni okvir:** Nepoznata osoba s postotkom maksimalne sličnosti.
* **Finotuning parametara:**
  * Klizač za prag prepoznavanja (preporučeno `0.55 - 0.65`).
  * Opcija automatskog zamućenja (cenzure) nepoznatih lica i prolaznika.
* **Registracija nepoznatih direktno iz rezultata:** Ako sustav detektira lice koje nije u bazi, možete ga direktno iz rezultata analize spremiti kao novu osobu.

### 3. 🛡️ Lokalna privatnost i sigurnost
* Nijedna fotografija niti vektor ne napuštaju vaše računalo.
* Radi potpuno samostalno bez internetske veze (*offline*).
* Baza podataka pohranjena je lokalno u SQLite datoteci (`data/database.db`).

---

## 📁 Struktura koda

```text
├── src/
│   ├── app.py             # Glavno Gradio web sučelje i kontroler
│   ├── db.py              # SQLite upravljanje osobama, slikama i vektorima
│   ├── face_engine.py     # Integracija RetinaFace detekcije i ArcFace prepoznavanja
│   └── image_utils.py     # Obrada slika s podrškom za Unicode/dijakritičke putanje
├── pokreni.bat            # Windows brzi pokretač (HR)
├── run_studio.bat         # Windows launcher (EN)
├── run_studio.py          # Glavni Python pokretač
├── requirements_studio.txt# Python ovisnosti za Studio
└── STUDIO_README.md       # Ova dokumentacija
```
