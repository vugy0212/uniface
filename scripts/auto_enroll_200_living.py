"""
Automatski unos 200 isključivo ŽIVUĆIH javnih osoba (Hrvatska, Srbija, BiH, Crna Gora).
Kategorije:
1. Glazbena scena (40)
2. Politika & Diplomacija (35)
3. Vrhunski sport & Treneri (40)
4. Religija & Duhovni život (30)
5. Film, kazalište & TV (30)
6. Znanost, poduzetništvo, olimpijci & mediji (25)

Preuzima portrete visoke kvalitete putem Wikimedia Commons / Wikipedia API-ja,
vrši biometrijsku detekciju lica, sprema 3-4 uzorka po osobi i rekalkulira FAISS indeks.
"""

import os
import sys
import time
import json
import uuid
import urllib.request
import urllib.parse
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import db
import face_engine
from image_utils import imwrite_unicode
from app import save_image_dedup, UPLOADS_DIR, CROPS_DIR

HEADERS = {
    "User-Agent": "ArgusFaceBot/2.0 (research-scalability@argusface.local; python-urllib)"
}

PEOPLE = [
    # =========================================================================
    # 1. 🎵 GLAZBENA SCENA - ŽIVUĆI IZVOĐAČI (40 osoba)
    # =========================================================================
    {"name": "Zlatan Stipišić Gibonni", "notes": "Glazba - Kantautor / Split", "terms": ["Gibonni", "Zlatan Stipisic Gibonni", "Zlatan Stipišić"]},
    {"name": "Marko Perković Thompson", "notes": "Glazba - Pjevač i kantautor / Čavoglave", "terms": ["Marko Perkovic Thompson", "Marko Perković Thompson", "Thompson pjevac"]},
    {"name": "Miroslav Škoro", "notes": "Glazba - Glazbenik i kantautor", "terms": ["Miroslav Skoro", "Miroslav Škoro"]},
    {"name": "Petar Grašo", "notes": "Glazba - Pop pjevač i kantautor / Split", "terms": ["Petar Graso", "Petar Grašo"]},
    {"name": "Tony Cetinski", "notes": "Glazba - Pop-rock pjevač / Rovinj", "terms": ["Tony Cetinski", "Toni Cetinski"]},
    {"name": "Severina Vučković", "notes": "Glazba - Pop pjevačica / Split", "terms": ["Severina Vuckovic", "Severina Vučković", "Severina"]},
    {"name": "Nina Badrić", "notes": "Glazba - Pop pjevačica / Zagreb", "terms": ["Nina Badric", "Nina Badrić"]},
    {"name": "Josipa Lisac", "notes": "Glazba - Glazbena diva / Zagreb", "terms": ["Josipa Lisac"]},
    {"name": "Gabi Novak", "notes": "Glazba - Jazz i zabavna diva", "terms": ["Gabi Novak"]},
    {"name": "Tereza Kesovija", "notes": "Glazba - Glazbena diva / Dubrovnik", "terms": ["Tereza Kesovija"]},
    {"name": "Mišo Kovač", "notes": "Glazba - Pjevačka legenda / Dalmacija", "terms": ["Miso Kovac", "Mišo Kovač"]},
    {"name": "Mladen Grdović", "notes": "Glazba - Pjevač zabavne glazbe / Zadar", "terms": ["Mladen Grdovic", "Mladen Grdović"]},
    {"name": "Goran Karan", "notes": "Glazba - Pjevač i kantautor / Split", "terms": ["Goran Karan"]},
    {"name": "Davor Gobac", "notes": "Glazba - Frontmen Psihomodo Pop", "terms": ["Davor Gobac", "Psihomodo Pop Gobac"]},
    {"name": "Neno Belan", "notes": "Glazba - Pjevač i kantautor / Đavoli", "terms": ["Neno Belan"]},
    {"name": "Boris Novković", "notes": "Glazba - Pop-rock kantautor", "terms": ["Boris Novkovic", "Boris Novković"]},
    {"name": "Maksim Mrvica", "notes": "Glazba - Virtuoz na klaviru / Šibenik", "terms": ["Maksim Mrvica"]},
    {"name": "Stjepan Hauser", "notes": "Glazba - Svjetski violončelist (2Cellos)", "terms": ["Stjepan Hauser", "Hauser cellist"]},
    {"name": "Luka Šulić", "notes": "Glazba - Violončelist (2Cellos)", "terms": ["Luka Sulic", "Luka Šulić"]},
    {"name": "Zdravko Čolić", "notes": "Glazba - Legendarni pop pjevač", "terms": ["Zdravko Colic", "Zdravko Čolić"]},
    {"name": "Dino Merlin", "notes": "Glazba - Kantautor / Sarajevo", "terms": ["Dino Merlin", "Edin Dervishalidovic"]},
    {"name": "Halid Bešlić", "notes": "Glazba - Pjevačka legenda / Sarajevo", "terms": ["Halid Beslic", "Halid Bešlić"]},
    {"name": "Goran Bregović", "notes": "Glazba - Skladatelj i glazbenik (Bijelo dugme)", "terms": ["Goran Bregovic", "Goran Bregović"]},
    {"name": "Momčilo Bajagić Bajaga", "notes": "Glazba - Rock autor i frontmen Bajaga i Instruktori", "terms": ["Momcilo Bajagic Bajaga", "Momčilo Bajagić", "Bajaga"]},
    {"name": "Željko Joksimović", "notes": "Glazba - Glazbenik i skladatelj", "terms": ["Zeljko Joksimovic", "Željko Joksimović"]},
    {"name": "Vlado Georgiev", "notes": "Glazba - Kantautor i producent", "terms": ["Vlado Georgiev"]},
    {"name": "Željko Samardžić", "notes": "Glazba - Pop pjevač", "terms": ["Zeljko Samardzic", "Željko Samardžić"]},
    {"name": "Miroslav Ilić", "notes": "Glazba - Pjevač narodne glazbe", "terms": ["Miroslav Ilic", "Miroslav Ilić"]},
    {"name": "Lepa Brena", "notes": "Glazba - Glazbena diva (Fahreta Jahić)", "terms": ["Lepa Brena", "Fahreta Jahic"]},
    {"name": "Haris Džinović", "notes": "Glazba - Pjevač i kantautor", "terms": ["Haris Dzinovic", "Haris Džinović"]},
    {"name": "Hari Varešanović", "notes": "Glazba - Frontmen Hari Mata Hari", "terms": ["Hari Varesanovic", "Hari Varešanović", "Hari Mata Hari"]},
    {"name": "Saša Lošić", "notes": "Glazba - Frontmen Plavi orkestar (Loša)", "terms": ["Sasa Losic", "Saša Lošić", "Sasa Losic Losa"]},
    {"name": "Željko Bebek", "notes": "Glazba - Rock pjevač / Bivši vokal Bijelog dugmeta", "terms": ["Zeljko Bebek", "Željko Bebek"]},
    {"name": "Alen Islamović", "notes": "Glazba - Rock pjevač (Divlje jagode / Bijelo dugme)", "terms": ["Alen Islamovic", "Alen Islamović"]},
    {"name": "Mladen Vojičić Tifa", "notes": "Glazba - Rock pjevač / Sarajevo", "terms": ["Mladen Vojicic Tifa", "Mladen Vojičić Tifa", "Tifa"]},
    {"name": "Jelena Rozga", "notes": "Glazba - Pop pjevačica / Split", "terms": ["Jelena Rozga"]},
    {"name": "Maja Šuput", "notes": "Glazba - Pjevačica i TV voditeljica", "terms": ["Maja Suput", "Maja Šuput"]},
    {"name": "Vesna Pisarović", "notes": "Glazba - Pop i jazz pjevačica", "terms": ["Vesna Pisarovic", "Vesna Pisarović"]},
    {"name": "Ana Rucner", "notes": "Glazba - Violončelistica", "terms": ["Ana Rucner"]},
    {"name": "Damir Urban", "notes": "Glazba - Rock kantautor / Rijeka", "terms": ["Damir Urban", "Urban singer"]},

    # =========================================================================
    # 2. 🏛️ POLITIKA & DIPLOMACIJA - ŽIVUĆI POLITIČARI (35 osoba)
    # =========================================================================
    {"name": "Jakov Milatović", "notes": "Politika - Predsjednik Crne Gore", "terms": ["Jakov Milatovic", "Jakov Milatović"]},
    {"name": "Milojko Spajić", "notes": "Politika - Predsjednik Vlade Crne Gore", "terms": ["Milojko Spajic", "Milojko Spajić"]},
    {"name": "Milo Đukanović", "notes": "Politika - Bivši predsjednik Crne Gore", "terms": ["Milo Djukanovic", "Milo Đukanović"]},
    {"name": "Dritan Abazović", "notes": "Politika - Bivši premijer Crne Gore", "terms": ["Dritan Abazovic", "Dritan Abazović"]},
    {"name": "Aleksa Bečić", "notes": "Politika - Potpredsjednik Vlade Crne Gore", "terms": ["Aleksa Becic", "Aleksa Bečić"]},
    {"name": "Andrija Mandić", "notes": "Politika - Predsjednik Skupštine Crne Gore", "terms": ["Andrija Mandic", "Andrija Mandić"]},
    {"name": "Milan Knežević", "notes": "Politika - Zastupnik u Skupštini Crne Gore (DNP)", "terms": ["Milan Knezevic", "Milan Knežević"]},
    {"name": "Zdravko Krivokapić", "notes": "Politika - Bivši premijer Crne Gore", "terms": ["Zdravko Krivokapic", "Zdravko Krivokapić"]},
    {"name": "Filip Vujanović", "notes": "Politika - Bivši predsjednik Crne Gore", "terms": ["Filip Vujanovic", "Filip Vujanović"]},
    {"name": "Vojislav Koštunica", "notes": "Politika - Bivši predsjednik SRJ i premijer Srbije", "terms": ["Vojislav Kostunica", "Vojislav Koštunica"]},
    {"name": "Vuk Drašković", "notes": "Politika - Pisac i političar (SPO)", "terms": ["Vuk Draskovic", "Vuk Drašković"]},
    {"name": "Rasim Ljajić", "notes": "Politika - Predsjednik SDPS-a / Novi Pazar", "terms": ["Rasim Ljajic", "Rasim Ljajić"]},
    {"name": "Usame Zukorlić", "notes": "Politika - Ministar u Vladi Srbije (SPP)", "terms": ["Usame Zukorlic", "Usame Zukorlić"]},
    {"name": "Nenad Čanak", "notes": "Politika - Političar i bivši predsjednik LSV", "terms": ["Nenad Canak", "Nenad Čanak"]},
    {"name": "Boško Obradović", "notes": "Politika - Bivši predsjednik Dveri", "terms": ["Bosko Obradovic", "Boško Obradović"]},
    {"name": "Miloš Jovanović", "notes": "Politika - Predsjednik Novog DSS-a", "terms": ["Milos Jovanovic", "Miloš Jovanović DSS"]},
    {"name": "Aleksandar Jovanović Ćuta", "notes": "Politika - Ekološki aktivist i zastupnik", "terms": ["Aleksandar Jovanovic Cuta", "Aleksandar Jovanović Ćuta"]},
    {"name": "Radomir Lazović", "notes": "Politika - Saborski zastupnik / Zeleno-levi front", "terms": ["Radomir Lazovic", "Radomir Lazović"]},
    {"name": "Marinika Tepić", "notes": "Politika - Zastupnica u Skupštini Srbije (SSP)", "terms": ["Marinika Tepic", "Marinika Tepić"]},
    {"name": "Miroslav Aleksić", "notes": "Politika - Predsjednik Narodnog pokreta Srbije", "terms": ["Miroslav Aleksic", "Miroslav Aleksić"]},
    {"name": "Srđan Milivojević", "notes": "Politika - Zastupnik Demokratske stranke", "terms": ["Srdjan Milivojevic", "Srđan Milivojević"]},
    {"name": "Saša Janković", "notes": "Politika - Bivši zaštitnik građana Srbije", "terms": ["Sasa Jankovic", "Saša Janković"]},
    {"name": "Tomislav Karamarko", "notes": "Politika - Bivši potpredsjednik Vlade RH i ministar", "terms": ["Tomislav Karamarko"]},
    {"name": "Zlatko Hasanbegović", "notes": "Politika - Povjesničar i bivši ministar kulture RH", "terms": ["Zlatko Hasanbegovic", "Zlatko Hasanbegović"]},
    {"name": "Davor Ivo Stier", "notes": "Politika - Saborski zastupnik i bivši ministar vanjskih poslova RH", "terms": ["Davor Ivo Stier"]},
    {"name": "Tonino Picula", "notes": "Politika - Zastupnik u Europskom parlamentu", "terms": ["Tonino Picula"]},
    {"name": "Biljana Borzan", "notes": "Politika - Zastupnica u Europskom parlamentu", "terms": ["Biljana Borzan"]},
    {"name": "Karolina Vidović Krišto", "notes": "Politika - Političarka i bivša saborska zastupnica", "terms": ["Karolina Vidovic Kristo", "Karolina Vidović Krišto"]},
    {"name": "Mislav Kolakušić", "notes": "Politika - Bivši sudac i europarlamentarac", "terms": ["Mislav Kolakusic", "Mislav Kolakušić"]},
    {"name": "Ivan Vilibor Sinčić", "notes": "Politika - Bivši europarlamentarac / Živi zid", "terms": ["Ivan Vilibor Sincic", "Ivan Vilibor Sinčić"]},
    {"name": "Željko Kerum", "notes": "Politika - Bivši gradonačelnik Splita i poduzetnik", "terms": ["Zeljko Kerum", "Željko Kerum"]},
    {"name": "Vesna Pusić", "notes": "Politika - Bivša ministrica vanjskih poslova RH", "terms": ["Vesna Pusic", "Vesna Pusić"]},
    {"name": "Zlatko Mateša", "notes": "Politika - Predsjednik HOO-a / Bivši premijer RH", "terms": ["Zlatko Matesa", "Zlatko Mateša"]},
    {"name": "Franjo Gregurić", "notes": "Politika - Bivši predsjednik Vlade demokratskog jedinstva RH", "terms": ["Franjo Greguric", "Franjo Gregurić"]},
    {"name": "Dražen Budiša", "notes": "Politika - Političar i disident / HSLS", "terms": ["Drazen Budisa", "Dražen Budiša"]},

    # =========================================================================
    # 3. ⚽🎾🏀 SPORT & TRENERI - ŽIVUĆI (40 osoba)
    # =========================================================================
    {"name": "Ivan Rakitić", "notes": "Sport - Srebrni vatreni SP 2018 / Hajduk", "terms": ["Ivan Rakitic", "Ivan Rakitić"]},
    {"name": "Vedran Ćorluka", "notes": "Sport - Nogometna legenda i pomoćni trener reprezentacije", "terms": ["Vedran Corluka", "Vedran Ćorluka"]},
    {"name": "Ivica Olić", "notes": "Sport - Nogometna legenda i izbornik U-21 reprezentacije", "terms": ["Ivica Olic", "Ivica Olić"]},
    {"name": "Danijel Pranjić", "notes": "Sport - Bivši nogometni reprezentativac i trener", "terms": ["Danijel Pranjic", "Danijel Pranjić"]},
    {"name": "Mladen Petrić", "notes": "Sport - Bivši nogometni reprezentativac", "terms": ["Mladen Petric", "Mladen Petrić"]},
    {"name": "Nemanja Vidić", "notes": "Sport - Nogometna legenda Manchester Uniteda", "terms": ["Nemanja Vidic", "Nemanja Vidić"]},
    {"name": "Dejan Stanković", "notes": "Sport - Nogometna legenda Intera i trener", "terms": ["Dejan Stankovic", "Dejan Stanković"]},
    {"name": "Branislav Ivanović", "notes": "Sport - Nogometna legenda Chelseaja", "terms": ["Branislav Ivanovic", "Branislav Ivanović"]},
    {"name": "Aleksandar Kolarov", "notes": "Sport - Bivši kapetan reprezentacije Srbije (Man City)", "terms": ["Aleksandar Kolarov"]},
    {"name": "Dejan Savićević", "notes": "Sport - Genije / Predsjednik FSCG i legenda Milana", "terms": ["Dejan Savicevic", "Dejan Savićević"]},
    {"name": "Predrag Mijatović", "notes": "Sport - Nogometna legenda Real Madrida i Partizana", "terms": ["Predrag Mijatovic", "Predrag Mijatović"]},
    {"name": "Stevan Jovetić", "notes": "Sport - Kapetan crnogorske reprezentacije", "terms": ["Stevan Jovetic", "Stevan Jovetić"]},
    {"name": "Stefan Savić", "notes": "Sport - Nogometaš reprezentacije Crne Gore", "terms": ["Stefan Savic", "Stefan Savić"]},
    {"name": "Dado Pršo", "notes": "Sport - Nogometna legenda (Monaco / Rangers / Hrvatska)", "terms": ["Dado Prso", "Dado Pršo"]},
    {"name": "Alen Bokšić", "notes": "Sport - Nogometna legenda (Marseille / Lazio / Juventus)", "terms": ["Alen Boksic", "Alen Bokšić"]},
    {"name": "Igor Štimac", "notes": "Sport - Brončani vatreni 1998 i nogometni trener", "terms": ["Igor Stimac", "Igor Štimac"]},
    {"name": "Stipe Pletikosa", "notes": "Sport - Legendarni vratar reprezentacije (HNS)", "terms": ["Stipe Pletikosa"]},
    {"name": "Niko Kranjčar", "notes": "Sport - Nogometna legenda (Tottenham / Dinamo / Hajduk)", "terms": ["Niko Kranjcar", "Niko Kranjčar"]},
    {"name": "Ante Rebić", "notes": "Sport - Srebrni vatreni SP 2018 (Lecce / Milan)", "terms": ["Ante Rebic", "Ante Rebić"]},
    {"name": "Nikola Vlašić", "notes": "Sport - Nogometni reprezentativac (Torino)", "terms": ["Nikola Vlasic", "Nikola Vlašić"]},
    {"name": "Lovro Majer", "notes": "Sport - Nogometni reprezentativac (Wolfsburg)", "terms": ["Lovro Majer"]},
    {"name": "Luka Sučić", "notes": "Sport - Nogometni reprezentativac (Real Sociedad)", "terms": ["Luka Sucic", "Luka Sučić"]},
    {"name": "Martin Baturina", "notes": "Sport - Nogometaš Dinama i reprezentacije", "terms": ["Martin Baturina"]},
    {"name": "Marko Pjaca", "notes": "Sport - Nogometaš Dinama i reprezentacije", "terms": ["Marko Pjaca"]},
    {"name": "Mario Pašalić", "notes": "Sport - Nogometni reprezentativac (Atalanta)", "terms": ["Mario Pasalic", "Mario Pašalić"]},
    {"name": "Josip Juranović", "notes": "Sport - Nogometni reprezentativac (Union Berlin)", "terms": ["Josip Juranovic", "Josip Juranović"]},
    {"name": "Luka Jović", "notes": "Sport - Nogometni reprezentativac Srbije (Milan)", "terms": ["Luka Jovic", "Luka Jović"]},
    {"name": "Strahinja Pavlović", "notes": "Sport - Nogometni reprezentativac Srbije (Milan)", "terms": ["Strahinja Pavlovic", "Strahinja Pavlović"]},
    {"name": "Lazar Samardžić", "notes": "Sport - Nogometni reprezentativac Srbije (Atalanta)", "terms": ["Lazar Samardzic", "Lazar Samardžić"]},
    {"name": "Luka Dončić", "notes": "Sport - NBA superstar (Dallas Mavericks / Slovenija)", "terms": ["Luka Doncic", "Luka Dončić"]},
    {"name": "Goran Dragić", "notes": "Sport - Bivši NBA All-Star košarkaš", "terms": ["Goran Dragic", "Goran Dragić"]},
    {"name": "Nikola Vučević", "notes": "Sport - NBA All-Star košarkaš (Chicago Bulls)", "terms": ["Nikola Vucevic", "Nikola Vučević"]},
    {"name": "Vlade Divac", "notes": "Sport - NBA legenda i Hall of Fame centar", "terms": ["Vlade Divac"]},
    {"name": "Predrag Stojaković", "notes": "Sport - Peđa Stojaković / NBA šampion (Sacramento)", "terms": ["Predrag Stojakovic", "Pedja Stojakovic", "Predrag Stojaković"]},
    {"name": "Dejan Bodiroga", "notes": "Sport - Košarkaška legenda i predsjednik Eurolige", "terms": ["Dejan Bodiroga"]},
    {"name": "Aleksandar Saša Đorđević", "notes": "Sport - Legendarni košarkaš i izbornik", "terms": ["Aleksandar Djordjevic", "Sasa Djordjevic", "Aleksandar Đorđević"]},
    {"name": "Predrag Danilović", "notes": "Sport - Košarkaška legenda i predsjednik KSS", "terms": ["Predrag Danilovic", "Predrag Danilović"]},
    {"name": "Miloš Teodosić", "notes": "Sport - Košarkaški maestro (Crvena zvezda / Virtus)", "terms": ["Milos Teodosic", "Miloš Teodosić"]},
    {"name": "Vasilije Micić", "notes": "Sport - MVP Eurolige / NBA košarkaš", "terms": ["Vasilije Micic", "Vasilije Micić"]},
    {"name": "Nikola Kalinić", "notes": "Sport - Košarkaš Crvene zvezde i reprezentacije", "terms": ["Nikola Kalinic", "Nikola Kalinić basketball"]},

    # =========================================================================
    # 4. ⛪ RELIGIJA & DUHOVNI ŽIVOT - ŽIVUĆI CRKVENI VELIKODOSTOJNICI (30 osoba)
    # =========================================================================
    {"name": "Zdenko Križić", "notes": "Religija - Nadbiskup splitsko-makarski", "terms": ["Zdenko Krizic", "Zdenko Križić", "Zdenko Krizic nadbiskup"]},
    {"name": "Milan Zgrablić", "notes": "Religija - Zadarski nadbiskup", "terms": ["Milan Zgrablic", "Milan Zgrablić"]},
    {"name": "Milan Stipić", "notes": "Religija - Vladika križevački (Grkokatolička crkva)", "terms": ["Milan Stipic", "Milan Stipić vladika"]},
    {"name": "Antun Škvorčević", "notes": "Religija - Umirovljeni požeški biskup", "terms": ["Antun Skvorcevic", "Antun Škvorčević"]},
    {"name": "Juraj Jezerinac", "notes": "Religija - Umirovljeni vojni ordinarij RH", "terms": ["Juraj Jezerinac"]},
    {"name": "Slobodan Štambuk", "notes": "Religija - Umirovljeni hvarski biskup", "terms": ["Slobodan Stambuk", "Slobodan Štambuk"]},
    {"name": "Ante Ivas", "notes": "Religija - Umirovljeni šibenski biskup", "terms": ["Ante Ivas", "Ante Ivas biskup"]},
    {"name": "Valter Župan", "notes": "Religija - Umirovljeni krčki biskup", "terms": ["Valter Zupan", "Valter Župan"]},
    {"name": "Ivo Martinović", "notes": "Religija - Požeški biskup", "terms": ["Ivo Martinovic biskup", "Ivo Martinović"]},
    {"name": "Jure Bogdan", "notes": "Religija - Vojni ordinarij u RH", "terms": ["Jure Bogdan", "Jure Bogdan biskup"]},
    {"name": "Vjekoslav Huzjak", "notes": "Religija - Bjelovarsko-križevački biskup", "terms": ["Vjekoslav Huzjak"]},
    {"name": "Tomislav Rogić", "notes": "Religija - Šibenski biskup", "terms": ["Tomislav Rogic", "Tomislav Rogić biskup"]},
    {"name": "Roko Glasnović", "notes": "Religija - Dubrovački biskup", "terms": ["Roko Glasnovic", "Roko Glasnović"]},
    {"name": "Ivan Štironja", "notes": "Religija - Porečki i pulski biskup", "terms": ["Ivan Stironja", "Ivan Štironja"]},
    {"name": "Rrok Gjonlleshaj", "notes": "Religija - Barski nadbiskup i apostolski upravitelj kotorski", "terms": ["Rrok Gjonlleshaj"]},
    {"name": "Velečasni Zlatko Sudac", "notes": "Religija - Svećenik i karizmatik", "terms": ["Zlatko Sudac"]},
    {"name": "Don Damir Stojić", "notes": "Religija - Studentski kapelan i svećenik salezijanac", "terms": ["Damir Stojic", "Damir Stojić", "Don Damir Stojic"]},
    {"name": "Episkop Vasilije Kačavenda", "notes": "Religija - Umirovljeni episkop zvorničko-tuzlanski", "terms": ["Vasilije Kacavenda", "Vasilije Kačavenda"]},
    {"name": "Episkop Georgije Đokić", "notes": "Religija - Umirovljeni episkop kanadski", "terms": ["Georgije Djokic", "Georgije Đokić"]},
    {"name": "Episkop Filaret Mićević", "notes": "Religija - Umirovljeni episkop mileševski", "terms": ["Filaret Micevic", "Filaret Mićević"]},
    {"name": "Episkop Konstantin Đokić", "notes": "Religija - Umirovljeni episkop srednjoevropski", "terms": ["Konstantin Djokic", "Konstantin Đokić"]},
    {"name": "Episkop Metodije Ostojić", "notes": "Religija - Episkop budimljansko-nikšićki", "terms": ["Metodije Ostojic", "Metodije Ostojić"]},
    {"name": "Episkop Dimitrije Rađenović", "notes": "Religija - Episkop zahumsko-hercegovački i primorski", "terms": ["Dimitrije Radjenovic", "Dimitrije Rađenović", "Episkop Dimitrije"]},
    {"name": "Episkop Isihije Rogić", "notes": "Religija - Episkop valjevski", "terms": ["Isihije Rogic", "Isihije Rogić", "Episkop Isihije"]},
    {"name": "Arhimandrit Metodije Marković", "notes": "Religija - Iguman manastira Hilandar (Sveta Gora)", "terms": ["Iguman Metodije Hilandar", "Metodije Markovic Hilandar"]},
    {"name": "Arhimandrit Rafailo Boljević", "notes": "Religija - Iguman manastira Podmaine (Budva)", "terms": ["Rafailo Boljevic", "Rafailo Boljević", "Otac Rafailo"]},
    {"name": "Otac Joil Bulatović", "notes": "Religija - Duhovnik manastira Ćirilovac", "terms": ["Otac Joil Bulatovic", "Otac Joil Cirilovac"]},
    {"name": "Otac Dionisije Pantelić", "notes": "Religija - Arhimandrit i duhovnik manastira Lipovac", "terms": ["Dionisije Pantelic", "Otac Dionisije Lipovac"]},
    {"name": "Nusret Abdibegović", "notes": "Religija - Muftija banjalučki", "terms": ["Nusret Abdibegovic", "Nusret ef. Abdibegovic"]},
    {"name": "Salem Dedović", "notes": "Religija - Muftija mostarski", "terms": ["Salem Dedovic", "Salem ef. Dedovic"]},

    # =========================================================================
    # 5. 🎭 FILM, KAZALIŠTE & TV - ŽIVUĆI GLUMCI I AUTORI (30 osoba)
    # =========================================================================
    {"name": "Slavko Štimac", "notes": "Kultura - Legendarni glumac (Vlak u snijegu / Ko to tamo peva)", "terms": ["Slavko Stimac", "Slavko Štimac"]},
    {"name": "Srđan Todorović", "notes": "Kultura - Glumac i glazbenik (Žika)", "terms": ["Srdjan Todorovic", "Srđan Todorović", "Zika Todorovic"]},
    {"name": "Dragan Mićanović", "notes": "Kultura - Glumac (Lajanje na zvezde / Kruna)", "terms": ["Dragan Micanovic", "Dragan Mićanović"]},
    {"name": "Sergej Trifunović", "notes": "Kultura - Filmski i kazališni glumac", "terms": ["Sergej Trifunovic", "Sergej Trifunović"]},
    {"name": "Branislav Lečić", "notes": "Kultura - Kazališni i filmski glumac", "terms": ["Branislav Lecic", "Branislav Lečić"]},
    {"name": "Vojin Ćetković", "notes": "Kultura - Glumac (Zona Zamfirova / Santa Maria della Salute)", "terms": ["Vojin Cetkovic", "Vojin Ćetković"]},
    {"name": "Nebojša Dugalić", "notes": "Kultura - Prvak drame i sveučilišni profesor", "terms": ["Nebojsa Dugalic", "Nebojša Dugalić"]},
    {"name": "Nenad Jezdić", "notes": "Kultura - Glumac", "terms": ["Nenad Jezdic", "Nenad Jezdić"]},
    {"name": "Vuk Kostić", "notes": "Kultura - Glumac (Ubice mog oca)", "terms": ["Vuk Kostic", "Vuk Kostić"]},
    {"name": "Milan Marić", "notes": "Kultura - Glumac (Toma)", "terms": ["Milan Maric", "Milan Marić glumac"]},
    {"name": "Miloš Timotijević", "notes": "Kultura - Glumac (Besa / Južni vetar)", "terms": ["Milos Timotijevic", "Miloš Timotijević"]},
    {"name": "Goran Bogdan", "notes": "Kultura - Glumac (Otac / Fargo)", "terms": ["Goran Bogdan"]},
    {"name": "Leon Lučev", "notes": "Kultura - Filmski i kazališni glumac", "terms": ["Leon Lucev", "Leon Lučev"]},
    {"name": "Krešimir Mikić", "notes": "Kultura - Glumac (Svećenikova djeca)", "terms": ["Kresimir Mikic", "Krešimir Mikić"]},
    {"name": "Ozren Grabarić", "notes": "Kultura - Prvak drame Gavelle", "terms": ["Ozren Grabaric", "Ozren Grabarić"]},
    {"name": "Goran Grgić", "notes": "Kultura - Glumac i predsjednik HDDU-a", "terms": ["Goran Grgic", "Goran Grgić"]},
    {"name": "Dušan Kovačević", "notes": "Kultura - Akademik, dramski pisac i scenarist (Maratonci)", "terms": ["Dusan Kovacevic", "Dušan Kovačević pisac"]},
    {"name": "Slobodan Šijan", "notes": "Kultura - Filmski redatelj (Ko to tamo peva)", "terms": ["Slobodan Sijan", "Slobodan Šijan"]},
    {"name": "Goran Marković", "notes": "Kultura - Filmski redatelj (Nacionalna klasa)", "terms": ["Goran Markovic reditelj", "Goran Marković"]},
    {"name": "Rajko Grlić", "notes": "Kultura - Filmski redatelj (U raljama života / Ustav RH)", "terms": ["Rajko Grlic", "Rajko Grlić"]},
    {"name": "Anica Dobra", "notes": "Kultura - Filmska i kazališna glumica", "terms": ["Anica Dobra"]},
    {"name": "Katarina Radivojević", "notes": "Kultura - Glumica (Zona Zamfirova)", "terms": ["Katarina Radivojevic", "Katarina Radivojević"]},
    {"name": "Nataša Ninković", "notes": "Kultura - Prvakinja drame Narodnog pozorišta", "terms": ["Natasa Ninkovic", "Nataša Ninković"]},
    {"name": "Hana Selimović", "notes": "Kultura - Kazališna i filmska glumica", "terms": ["Hana Selimovic", "Hana Selimović"]},
    {"name": "Sloboda Mićalović", "notes": "Kultura - Glumica (Ranjeni orao)", "terms": ["Sloboda Micalovic", "Sloboda Mićalović"]},
    {"name": "Mima Karadžić", "notes": "Kultura - Glumac i producent", "terms": ["Mima Karadzic", "Milutin Mima Karadzic"]},
    {"name": "Gordan Kičić", "notes": "Kultura - Glumac i redatelj", "terms": ["Gordan Kicic", "Gordan Kičić"]},
    {"name": "Bojan Dimitrijević", "notes": "Kultura - Kazališni i filmski glumac (Pikac)", "terms": ["Bojan Dimitrijevic glumac", "Bojan Dimitrijević"]},
    {"name": "Nikola Rakočević", "notes": "Kultura - Glumac", "terms": ["Nikola Rakocevic", "Nikola Rakočević"]},
    {"name": "Amar Bukvić", "notes": "Kultura - Kazališni i TV glumac", "terms": ["Amar Bukvic", "Amar Bukvić"]},

    # =========================================================================
    # 6. 🔬 ZNANOST, PODUZETNIŠTVO, OLIMPIJCI & MEDIJI - ŽIVUĆI (25 osoba)
    # =========================================================================
    {"name": "Barbara Matić", "notes": "Sport - Olimpijska pobjednica u judu (Pariz 2024)", "terms": ["Barbara Matic", "Barbara Matić judo"]},
    {"name": "Matea Jelić", "notes": "Sport - Olimpijska pobjednica u taekwondou (Tokio 2020)", "terms": ["Matea Jelic", "Matea Jelić taekwondo"]},
    {"name": "Damir Martin", "notes": "Sport - Trostruki olimpijski osvajač medalja u veslanju", "terms": ["Damir Martin", "Damir Martin rower"]},
    {"name": "Giovanni Cernogoraz", "notes": "Sport - Olimpijski pobjednik u streljaštvu (London 2012)", "terms": ["Giovanni Cernogoraz"]},
    {"name": "Josip Glasnović", "notes": "Sport - Olimpijski pobjednik u streljaštvu (Rio 2016)", "terms": ["Josip Glasnovic", "Josip Glasnović"]},
    {"name": "Zrinka Ljutić", "notes": "Sport - Vrhunska hrvatska skijašica (Svjetski kup)", "terms": ["Zrinka Ljutic", "Zrinka Ljutić"]},
    {"name": "Leona Popović", "notes": "Sport - Vrhunska hrvatska skijašica (Svjetski kup)", "terms": ["Leona Popovic", "Leona Popović"]},
    {"name": "Sara Kolak", "notes": "Sport - Olimpijska pobjednica u bacanju koplja (Rio 2016)", "terms": ["Sara Kolak"]},
    {"name": "Marijo Možnik", "notes": "Sport - Europski gimnastički prvak i predsjednik HGS-a", "terms": ["Marijo Moznik", "Marijo Možnik"]},
    {"name": "Željko Mavrović", "notes": "Sport - Šaka sa Srednjaka / Boksački prvak Europe", "terms": ["Zeljko Mavrovic", "Željko Mavrović"]},
    {"name": "Emil Tedeschi", "notes": "Poduzetništvo - Predsjednik uprave Atlantic Grupe", "terms": ["Emil Tedeschi"]},
    {"name": "Branko Roglić", "notes": "Poduzetništvo - Vlasnik Orbico Grupe", "terms": ["Branko Roglic", "Branko Roglić"]},
    {"name": "Davor Štern", "notes": "Poduzetništvo - Energetski stručnjak i bivši ministar gospodarstva", "terms": ["Davor Stern", "Davor Štern"]},
    {"name": "Darko Rundek", "notes": "Glazba - Kantautor i rock legenda (Haustor)", "terms": ["Darko Rundek"]},
    {"name": "Zoran Predin", "notes": "Glazba - Kantautor i frontmen Lačnog Franza", "terms": ["Zoran Predin"]},
    {"name": "Vlatko Stefanovski", "notes": "Glazba - Gitarski virtuoz i frontmen Leb i sol", "terms": ["Vlatko Stefanovski"]},
    {"name": "Dado Topić", "notes": "Glazba - Rock pjevač i basist (Time / Korni grupa)", "terms": ["Dado Topic", "Dado Topić"]},
    {"name": "Nele Karajlić", "notes": "Kultura - Glazbenik, scenarist i pisac (Zabranjeno pušenje)", "terms": ["Nele Karajlic", "Nele Karajlić", "Nenad Jankovic Nele"]},
    {"name": "Mihael Zmajlović", "notes": "Politika - Saborski zastupnik / Bivši ministar zaštite okoliša", "terms": ["Mihael Zmajlovic", "Mihael Zmajlović"]},
    {"name": "Mirela Holy", "notes": "Politika - Bivša ministrica zaštite okoliša i sveučilišna profesorica", "terms": ["Mirela Holy"]},
    {"name": "Maja Sever", "notes": "Mediji - Predsjednica Europske federacije novinara i TV novinarka", "terms": ["Maja Sever"]},
    {"name": "Mojmira Pastorčić", "notes": "Mediji - TV urednica i voditeljica RTL Direkta", "terms": ["Mojmira Pastorcic", "Mojmira Pastorčić"]},
    {"name": "Zoran Šprajc", "notes": "Mediji - TV urednik i novinar (Stanje nacije)", "terms": ["Zoran Sprajc", "Zoran Šprajc"]},
    {"name": "Andrija Jarak", "notes": "Mediji - Istraživački TV novinar", "terms": ["Andrija Jarak"]},
    {"name": "Jovan Memedović", "notes": "Mediji - TV autor Sasvim prirodno / Kviz Potera", "terms": ["Jovan Memedovic", "Jovan Memedović"]}
]

SKIP_KEYWORDS = [
    "flag", "coat", "signature", "logo", "map", "icon", "stemma", "herb",
    "grb", "zastava", "potpis", "badge", "diagram", "stadium", "building", "tomb"
]

def search_wikimedia(query, limit=15):
    """Pronalazi URL-ove slika na Wikimedia Commons za zadani upit."""
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrnamespace=6&gsrsearch={urllib.parse.quote(query)}"
        f"&gsrlimit={limit}&prop=imageinfo&iiprop=url|size|mime&format=json"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    urls = []
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, p in pages.items():
                for ii in p.get("imageinfo", []):
                    mime = ii.get("mime", "")
                    u = ii.get("url", "")
                    if mime in ["image/jpeg", "image/png"]:
                        u_lower = u.lower()
                        if not any(skip in u_lower for skip in SKIP_KEYWORDS):
                            if u not in urls:
                                urls.append(u)
    except Exception:
        pass
    return urls

def search_wikipedia_page_images(title, lang="hr"):
    """Dohvaća glavne slike sa stranice na Wikipediji kao alternativni izvor."""
    url = (
        f"https://{lang}.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}"
        f"&prop=pageimages|images&pithumbsize=1000&format=json"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    urls = []
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, p in pages.items():
                thumb = p.get("thumbnail", {}).get("source")
                if thumb and not any(skip in thumb.lower() for skip in SKIP_KEYWORDS):
                    urls.append(thumb)
    except Exception:
        pass
    return urls

def download_image(url):
    """Preuzima sliku s weba i dekodira je u OpenCV BGR format."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img_bgr
    except Exception:
        return None

def process_person(person_info, target_samples=3):
    """Preuzima slike, ekstrahira lica i sprema osobu u bazu."""
    name = person_info["name"]
    notes = person_info["notes"]
    terms = person_info["terms"]

    # 1. Prikupi kandidate za URL-ove
    candidate_urls = []
    for term in terms:
        urls = search_wikimedia(term, limit=12)
        for u in urls:
            if u not in candidate_urls:
                candidate_urls.append(u)
        if len(candidate_urls) >= 15:
            break

    # Dopuna sa Wikipedije ako fali slika
    if len(candidate_urls) < 6:
        for lang in ["hr", "sr", "bs", "en", "sh"]:
            wiki_urls = search_wikipedia_page_images(name, lang=lang)
            for wu in wiki_urls:
                if wu not in candidate_urls:
                    candidate_urls.append(wu)

    if not candidate_urls:
        return 0, "Nisu pronađene slike"

    # 2. Dohvati ili kreiraj osobu u bazi
    person_id = db.get_or_create_person(name, notes)

    existing_samples = db.get_person_samples(person_id)
    accepted_embeddings = [s["embedding"] for s in existing_samples if s.get("embedding") is not None]

    if len(accepted_embeddings) >= target_samples:
        return len(accepted_embeddings), "Već ima dovoljno uzoraka"

    saved_count = len(accepted_embeddings)

    for u in candidate_urls:
        if saved_count >= target_samples:
            break

        img_bgr = download_image(u)
        if img_bgr is None:
            continue

        h_img, w_img = img_bgr.shape[:2]
        if h_img < 100 or w_img < 100:
            continue

        faces = face_engine.extract_faces_from_image(img_bgr)
        if not faces:
            continue

        # Sortiraj lica po veličini bounding boxa (najveće prvo)
        faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
        best_face = faces[0]

        fw = best_face["bbox"][2] - best_face["bbox"][0]
        fh = best_face["bbox"][3] - best_face["bbox"][1]

        # Provjera minimalne rezolucije lica i pouzdanosti modela
        if fw < 50 or fh < 50 or best_face["confidence"] < 0.70:
            continue

        # Ako na slici ima više lica, prihvati samo ako je glavno lice dominantno
        if len(faces) > 1:
            second_area = (faces[1]["bbox"][2] - faces[1]["bbox"][0]) * (faces[1]["bbox"][3] - faces[1]["bbox"][1])
            best_area = fw * fh
            if best_area < 1.3 * second_area and best_face["confidence"] < 0.85:
                continue

        # Deduplikacija: izbjegni spremanje identičnih slika ili istih kadrova
        is_duplicate = False
        emb = best_face["embedding"]
        for prev_emb in accepted_embeddings:
            sim = float(np.dot(emb, prev_emb))
            if sim > 0.95:  # Gotovo identičan kadar
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # Spremi original i crop
        orig_path = save_image_dedup(img_bgr, UPLOADS_DIR, prefix="orig")
        crop_filename = f"crop_{person_id}_{uuid.uuid4().hex[:8]}.jpg"
        crop_path = os.path.join(CROPS_DIR, crop_filename)
        imwrite_unicode(crop_path, best_face["crop_bgr"])

        # Upiši u SQLite bazu
        db.add_face_sample(person_id, orig_path, crop_path, emb, best_face["confidence"])
        accepted_embeddings.append(emb)
        saved_count += 1
        time.sleep(0.10)

    return saved_count, "Uspješno"

def main():
    print("=" * 70)
    print("  ARGUSFACE STUDIO - MASOVNI UNOS 200 ŽIVUĆIH JAVNIH OSOBA")
    print("=" * 70)
    print(f"Ciljani broj osoba: {len(PEOPLE)} (isključivo živuće osobe)")
    print(f"Ciljano uzoraka po osobi: 3 do 4 kvalitetne profilne fotografije")
    print("-" * 70)

    start_time = time.time()
    success_count = 0
    total_images_added = 0

    for idx, p in enumerate(PEOPLE, 1):
        name = p["name"]
        print(f"[{idx:03d}/{len(PEOPLE)}] {name} ... ", end="", flush=True)

        try:
            samples_cnt, status = process_person(p, target_samples=3)
            if samples_cnt >= 1:
                badge = "Zlatni (3+ sl.)" if samples_cnt >= 3 else f"Profil ({samples_cnt} sl.)"
                print(f"✅ {samples_cnt} slika [{badge}]")
                success_count += 1
                total_images_added += samples_cnt
            else:
                print(f"⚠️ {status}")
        except Exception as e:
            print(f"❌ Greška: {e}")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Završena obrada u {elapsed:.1f} sekundi ({elapsed/60:.1f} min).")
    print(f"Uspješno unesenih osoba: {success_count}/{len(PEOPLE)}")
    print(f"Ukupno novih slika s biometrijom: {total_images_added}")
    print("-" * 70)

    # Rekalkulacija i ažuriranje FAISS vektorskog indeksa
    print("🔄 Obnavljanje FAISS vektorskog indeksa (FlatIP / HNSW) ...")
    try:
        from face_engine import get_face_index
        idx = get_face_index(force_refresh=True)
        print(f"✅ FAISS Vektorski indeks uspješno obnovljen! [{idx.backend}] Ukupno vektora u indeksu: {idx.total_samples}")
    except Exception as e:
        print(f"⚠️ Obnova indeksa: {e}")

    print("=" * 70)
    print("  MASOVNI UNOS 200 ŽIVUĆIH OSOBA JE USPJEŠNO ZAVRŠEN!")
    print("=" * 70)

if __name__ == "__main__":
    main()
