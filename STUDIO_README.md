# 👤 UniFace Studio - Sustav za prepoznavanje i analizu lica

UniFace Studio je kompletna aplikacija s modernim grafičkim web sučeljem (**Gradio Web UI**) koja radi **100% lokalno** na vašem računalu, koristeći **UniFace v4.0.0**, **RetinaFace**, **ArcFace**, **FairFace** i **SQLite** bazu lica.

---

## 🚀 Brzo pokretanje

### Opcija 1: Windows (Dvoklik)
Dovoljno je pokrenuti:
* `pokreni.bat` ili `run_studio.bat` - Pokretanje kompletnog Web UI studija
* `live_kamera.bat` - Izravno pokretanje live prepoznavanja kamere
* `mreza_kamera.bat` - Izravno pokretanje 2×2 nadzorne mreže kamera
* `nvr_snimanje.bat` - Izravno pokretanje 24/7 NVR pozadinskog snimanja

### Opcija 2: Pokretanje putem terminala
```bash
# 1. Instalacija ovisnosti za Studio
pip install -r requirements_studio.txt

# 2. Pokretanje aplikacije (automatski se otvara u samostalnom radnom prozoru)
python run_studio.py

# Ako želite prisiliti otvaranje u običnom web pregledniku:
python run_studio.py --browser
```
Aplikacija se otvara kao samostalna **desktop aplikacija** (nativni prozor bez adresne trake i tabova preglednika). U pozadini se servis vrti na:  
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
  * Klizač za prag prepoznavanja (preporučeno `0.45 - 0.55`).
  * Opcija automatskog zamućenja (cenzure) nepoznatih lica i prolaznika.
* **Registracija nepoznatih direktno iz rezultata:** Ako sustav detektira lice koje nije u bazi, možete ga direktno iz rezultata analize spremiti kao novu osobu.

### 3. 🛡️ Lokalna privatnost i sigurnost
* Nijedna fotografija niti vektor ne napuštaju vaše računalo.
* Radi potpuno samostalno bez internetske veze (*offline*).
* Baza podataka pohranjena je lokalno u SQLite datoteci (`data/database.db`).

### 4. 🎛️ Mreža Više Kamera (Multi-Camera 2×2 Grid)
* **Paralelno procesiranje do 4 kamere istovremeno:** Podržava kombinaciju lokalnih USB web kamera i mrežnih IP/RTSP nadzornih kamera (npr. Denver IPC-1030MK2 na `192.168.50.236`).
* **Interaktivne prečice u prozoru nadzora:**
  * `[1]` do `[4]`: Solo prikaz preko cijelog ekrana za odabranu kameru.
  * `[0]` ili `[ESC]`: Povratak u 2×2 mrežu.
  * `[R]`: Trenutno uključivanje / isključivanje NVR snimanja.
  * `[E]`: Uključivanje / isključivanje automatske evidencije prolazaka.
  * `[S]`: Spremanje kadra visoke rezolucije u arhivu.
  * `[Q]`: Sigurno zaustavljanje i izlaz.

### 5. 📹 24/7 NVR Snimanje i Biometrijski Video Markeri
* **Kontinuirano snimanje u rotirajuće MP4 segmente:** Podesivo trajanje segmenata (npr. 5 minuta po datoteci).
* **Automatsko FIFO čišćenje diska:** Održava zauzeće unutar zadane kvote (npr. 20 GB) automatskim brisanjem najstarijih segmenata.
* **Biometrijski video markeri (Bookmarks):** Svaki prepoznati prolazak u bazi automatski bilježi točnu video datoteku i vremensku sekundu detekcije (`video_offset_sec`).
* **Integrirani video player i arhiva:**
  * U **Dnevniku prolazaka (Tab 4)** klikom na bilo koji prolazak automatski se otvara video snimka na točnom trenutku detekcije.
  * Podsekcija **NVR Video Arhiva** omogućuje izravan pregled, reprodukciju i preuzimanje svih snimljenih MP4 segmenata.
* **Pozadinski servis:** Pokretanje bez grafičkog sučelja putem `nvr_snimanje.bat`.

---

## 📁 Struktura koda

```text
├── src/
│   ├── app.py             # Glavno Gradio web sučelje i kontroler
│   ├── db.py              # SQLite upravljanje osobama, slikama, vektorima i NVR evidencijom
│   ├── face_engine.py     # Integracija RetinaFace detekcije i ArcFace prepoznavanja
│   ├── image_utils.py     # Obrada slika s podrškom za Unicode/dijakritičke putanje
│   ├── live_cam.py        # Modul za prikaz uživo, detekciju i NVR snimanje jedne kamere
│   ├── multicam.py        # Optimizirani 2×2 Multi-Cam Grid sustav dretvi i renderinga
│   └── nvr_recorder.py    # 24/7 NVR MP4 segmentator, FIFO disk cleaner i bookmarking
├── run_studio.py          # Glavni pokretač Gradio web sučelja
├── run_live_cam.py        # Pokretač live prikaza pojedinačne kamere
├── run_multicam.py        # Pokretač 2×2 mreže kamera
├── run_nvr.py             # Pokretač pozadinskog 24/7 NVR servisa snimanja
├── pokreni.bat            # Windows brzi pokretač Web UI (HR)
├── live_kamera.bat        # Windows brzi pokretač live kamere
├── mreza_kamera.bat       # Windows brzi pokretač 2×2 mreže kamera
├── nvr_snimanje.bat       # Windows brzi pokretač 24/7 NVR snimanja
├── requirements_studio.txt# Python ovisnosti za Studio
└── STUDIO_README.md       # Ova dokumentacija
```
