# 👤 ArgusFace Studio - Sustav za prepoznavanje i analizu lica

ArgusFace Studio je kompletna aplikacija s modernim grafičkim web sučeljem (**Gradio Web UI**) koja radi **100% lokalno** na vašem računalu, koristeći **RetinaFace** (MIT), **EdgeFace BASE** (BSD-3-Clause) i **SQLite** bazu lica uz 100% komercijalnu licencu.

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

### 1. 👥 Baza Osoba (Unos, uređivanje, centroid i FAISS indeks)
* **Unos pojedinačnih i grupnih fotografija:** Učitajte sliku s jednom ili više osoba. Sustav automatski detektira sva lica, numerira ih i omogućuje odabir točno onog lica koje želite spremiti u bazu.
* **Uređivanje profila i bilješki (Novo):** Klikom na bilo koju osobu u tablici možete izravno urediti ime, prezime ili administrativne bilješke te spremiti izmjene u bazu jednim klikom.
* **Centroid Multi-Sample tehnologija:** Spajanje višestrukih fotografija u jedinstveni biometrijski centroid podiže točnost prepoznavanja na preko 95-100%.
* **⚡ FAISS Vektorski Indeks (Novo):** Ugrađeni C++ AVX2/HNSW vektorski indeks omogućuje sub-milisekundno pretraživanje (0.1 ms) čak i pri bazama s desecima tisuća lica, bez CPU zagušenja.
* **Pretraga i brzi odabir:** Tražilica u stvarnom vremenu po imenu ili bilješci, pregled galerije uzoraka i jednostavno brisanje.

### 2. 🔍 Prepoznavanje i analiza lica
* **Višestruko prepoznavanje:** Detektira i uspoređuje sva lica s fotografije prema biometrijskim vektorima iz baze (512-dimenzionalni EdgeFace BASE embedding).
* **Pregledni vizualni rezultati:**
  * 🟩 **Zeleni okvir:** Prepoznata osoba s imenom, postotkom točnosti i sigurnosnom marginom u odnosu na drugog najizglednijeg kandidata.
  * 🟥 **Crveni okvir:** Nepoznata osoba s postotkom maksimalne sličnosti.
* **Finotuning parametara:**
  * Klizač za prag prepoznavanja (preporučeno `0.45 - 0.55`).
  * Opcija automatskog zamućenja (cenzure) nepoznatih lica i prolaznika.
  * Pametni prikaz kartica detekcija (zadano samo prepoznati, uz mogućnost prikaza svih lica).
* **Registracija nepoznatih direktno iz rezultata:** Ako sustav detektira lice koje nije u bazi, možete ga direktno iz rezultata analize spremiti kao novu osobu.

### 3. 🛡️ Lokalna privatnost, GDPR i Sigurnost
* **100% lokalna obrada (On-Premise):** Nijedna fotografija niti vektor nikada ne napuštaju vaše računalo. Radi potpuno samostalno bez interneta (*offline*).
* **🛡️ GDPR Automatska Rotacija Podataka (Novo):** Usklađeno s Člankom 5(1)(e) GDPR-a (*Storage limitation*). U Tabu 6 možete odabrati rok rotacije (15 dana preporučeno za video nadzor / AZOP, 30, 60, 90 dana ili trajno). Sustav pri svakom startu automatski uklanja stare događaje, kadrove i NVR MP4 snimke te oslobađa prostor na disku.
* **Pravo na zaborav (Članak 17. GDPR-a):** Trenutno trajno brisanje profila osobe i svih povezanih biometrijskih vektora.

### 4. ⚡ Univerzalno Hardversko Ubrzanje (DirectML & OpenVINO) (Novo)
* **DirectX 12 GPU & NPU akceleracija:** Podrška za **Intel Iris Xe, Intel Arc, AMD Radeon, NVIDIA GeForce i NPU procesore** (Intel Core Ultra, AMD Ryzen AI) bez potrebe za instaliranjem vanjskog CUDA SDK-a.
* **Automatska detekcija (`device="AUTO"`):** Sustav automatski prepoznaje najbolji dostupni grafički hardver za video nadzor uživo, multi-cam mrežu i sortiranje slika.
* **Cyber statusni bedž (Tab 6):** Prikazuje aktivni mehanizam ubrzanja, detektirane grafičke kartice i status vektorske baze u stvarnom vremenu.

### 5. 🎛️ Mreža Više Kamera (Multi-Camera 2×2 Grid)
* **Paralelno procesiranje do 4 kamere istovremeno:** Podržava kombinaciju lokalnih USB web kamera i mrežnih IP/RTSP nadzornih kamera (npr. Denver IPC-1030MK2).
* **Interaktivne prečice u prozoru nadzora:**
  * `[1]` do `[4]`: Solo prikaz preko cijelog ekrana za odabranu kameru.
  * `[0]` ili `[ESC]`: Povratak u 2×2 mrežu.
  * `[R]`: Trenutno uključivanje / isključivanje NVR snimanja.
  * `[E]`: Uključivanje / isključivanje automatske evidencije prolazaka.
  * `[S]`: Spremanje kadra visoke rezolucije u arhivu.
  * `[Q]`: Sigurno zaustavljanje i izlaz.

### 6. 📹 24/7 NVR Snimanje i Biometrijski Video Markeri
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
