# UniFace - Arhitektura Licenciranja i Komercijalizacije

Ovaj dokument definira cjelokupni model licenciranja, zaštite i aktivacije aplikacije UniFace (hibridni pristup: **Online** i **Offline / Air-Gapped**).

---

## 1. Modeli ponude i razine funkcionalnosti

| Značajka | Besplatni Demo (Trial / Free) | PRO / Enterprise Licenca |
| :--- | :--- | :--- |
| **Baza lica (zaposlenici)** | Ograničeno: **max 5 osoba** | **Neograničeno** (ili definirano ugovorom) |
| **Broj kamera** | 1 lokalna web/RTSP kamera | Više kamera paralelno |
| **Anti-Spoofing (Liveness Guard)** | Isključeno (samo osnovno prepoznavanje) | **Uključeno (MiniFASNet V2)**: blokira mobitele i slike |
| **Izvoz podataka** | Blokiran izvoz (ili samo zadnja 3 zapisa) | Puni Excel / CSV / PDF izvoz |
| **Email / Webhook alarmi** | Isključeno | Uključeno |
| **Trajanje rada** | Neograničeno (klijent može testirati u miru) | Trajna licenca ili godišnja pretplata |

---

## 2. Hibridni model aktivacije (Online + Offline)

Klijentska aplikacija u sebi sadrži **samo Javni ključ (Public Key)** kojim matematički provjerava autentičnost licence. Tvoj poslužitelj (ili tvoj admin alat na računalu) posjeduje **Privatni ključ (Private Key)** kojim se licenca potpisuje.

```
                         ┌────────────────────────┐
                         │   UniFace Aplikacija   │
                         │ (Generira Machine HWID)│
                         └───────────┬────────────┘
                                     │
                ┌────────────────────┴────────────────────┐
                ▼                                         ▼
       [ A. Online Aktivacija ]                  [ B. Offline Aktivacija ]
  (Korisnici s web stranice)                (Zatvorene mreže / Air-Gapped)
                │                                         │
1. Unos ključa (npr. UNIFACE-XXXX)         1. Aplikacija prikaže "Machine ID"
2. UniFace šalje API-ju:                      (npr. MCH-98A1-BC44)
   { license_key, machine_id }             2. Korisnik pošalje Machine ID emailom
3. Tvoj poslužitelj validira i potpisuje      ili ga unese na web portal
   licencu Privatnim ključem               3. Poslužitelj/Ti izgeneriraš licencu:
4. UniFace prima i sprema "license.dat"       "uniface.lic"
                │                          4. Korisnik klikne "Učitaj licencu" (.lic)
                │                                         │
                └────────────────────┬────────────────────┘
                                     ▼
                   [ Lokalna validacija u UniFace ]
                   - Provjera potpisa Javnim ključem
                   - Provjera poklapanja Machine ID-a
                   - Provjera datuma isteka (ako postoji)
                                     │
                             [ STATUS: AKTIVIRANO ]
```

---

## 3. Tehnički detalji implementacije

### 3.1. Generiranje Hardverskog Otiska (Machine ID / HWID)
Machine ID veže licencu za konkretno računalo kako se ista licenca ne bi mogla kopirati na druga računala.

Na Windows operativnom sustavu preporučuje se kombinirati:
* UUID matične ploče (`wmic csproduct get uuid`)
* Serijski broj sistemskog diska (`wmic diskdrive get serialnumber`)
* CPU ID (`wmic cpu get processorid`)

Hashira se u kratki format: npr. `SHA256(...)[:16]` -> `MCH-A1B2-C3D4-E5F6`.

### 3.2. Sadržaj licencnog certifikata (JSON Payload)
Podaci koje potpisuje tvoj privatni ključ (koristeći Ed25519 ili RSA-2048):

```json
{
  "license_key": "UNIFACE-2026-X812B",
  "machine_id": "MCH-A1B2-C3D4-E5F6",
  "tier": "enterprise",
  "max_faces": -1,
  "features": ["multicam", "export_reports", "webhook_alerts"],
  "valid_until": "2028-12-31",
  "issued_to": "Tvrtka d.o.o."
}
```

### 3.3. Lokalna provjera u aplikaciji
Aplikacija učitava certifikat i njegov kriptografski potpis:
1. `verify_signature(payload, signature, PUBLIC_KEY)` $\rightarrow$ ako je ijedan bajt izmijenjen, provjera pada.
2. `get_current_machine_id() == payload["machine_id"]` $\rightarrow$ ako je prebačeno na drugi PC, provjera pada.
3. `current_date <= payload["valid_until"]` $\rightarrow$ provjera valjanosti.

---

## 4. Zaštita Python izvornog koda od zaobilaženja (Reverse Engineering)

Budući da se Python `exe` (kreiran npr. kroz PyInstaller) može relativno lako dekompajlirati, primjenjuje se višeslojna zaštita:

1. **PyArmor (`pyarmor gen`)**:
   - Šifrira Python bytecode, obfuskira varijable i funkcije te onemogućuje jednostavne dekompajlere (poput `pycdc` ili `uncompyle6`).
   - PyArmor sam po sebi podržava i vezanje za serijski broj hardvera (`pyarmor gen -b`).
2. **Kompajliranje u C/C++ binarije (Cython)**:
   - Modul `license_validator.py` i ključni dijelovi poslovne logike mogu se kroz Cython pretvoriti u C kod i kompajlirati u `.pyd` (Windows DLL format). Dekompajliranje `.pyd` datoteke u Python kod je praktički nemoguće.
3. **Raspršene provjere (Defense in Depth)**:
   - Umjesto samo jednog `if is_licensed:` na početku programa, provjere se diskretno umeću u kritične funkcije:
     - Kod dodavanja novog lica u bazu (`db.py` / `face_engine.py`) provjerava se limit lica.
     - Kod izvoza izvještaja provjerava se potpis licence.

---

## 5. Korisničko sučelje (UI prozor za aktivaciju)

Unutar prozora postavki u UniFace dodati dijalog **"Upravljanje Licencom"**:

* **Kartica 1: Online aktivacija**
  * Polje: `Unesite licencni ključ: [________________]`
  * Gumb: `[Aktiviraj putem interneta]` (povezuje se na npr. `https://api.uniface.com/v1/activate`)

* **Kartica 2: Offline aktivacija (Zaštićeni sustavi)**
  * Polje: `Vaš Machine ID:` `[ MCH-98A1-BC44-22F1 ]` (s gumbom za kopiranje)
  * Upute: *"Pošaljite ovaj kod na licenca@uniface.com ili ga unesite u portal za preuzimanje .lic datoteke."*
  * Gumb: `[Učitaj licencnu datoteku (.lic)]`
