"""
Automatski unos 200 javnih osoba (Hrvatska, Srbija, BiH - politika, sport, religija, kultura, znanost).
Preuzima portrete visoke kvalitete putem Wikimedia Commons / Wikipedia API-ja,
vrši biometrijsku detekciju lica, sprema 3-4 različita uzorka po osobi i rekalkulira FAISS indeks.
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

# Ensure src is on path
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
    # 1. 🏛️ POLITIKA & JAVNI ŽIVOT - HRVATSKA (35 osoba)
    # =========================================================================
    {"name": "Zoran Milanović", "notes": "Politika - Predsjednik RH", "terms": ["Zoran Milanovic", "Zoran Milanović", "Milanovic portrait"]},
    {"name": "Andrej Plenković", "notes": "Politika - Predsjednik Vlade RH", "terms": ["Andrej Plenkovic", "Andrej Plenković", "Plenkovic portrait"]},
    {"name": "Gordan Jandroković", "notes": "Politika - Predsjednik Hrvatskog sabora", "terms": ["Gordan Jandrokovic", "Gordan Jandroković"]},
    {"name": "Tomislav Tomašević", "notes": "Politika - Gradonačelnik Zagreba", "terms": ["Tomislav Tomasevic", "Tomislav Tomašević"]},
    {"name": "Ivan Penava", "notes": "Politika - Gradonačelnik Vukovara / DP", "terms": ["Ivan Penava", "Penava portrait"]},
    {"name": "Božo Petrov", "notes": "Politika - Predsjednik Mosta", "terms": ["Bozo Petrov", "Božo Petrov"]},
    {"name": "Peđa Grbin", "notes": "Politika - Saborski zastupnik / SDP", "terms": ["Pedja Grbin", "Peđa Grbin"]},
    {"name": "Dalija Orešković", "notes": "Politika - Zastupnica u Saboru", "terms": ["Dalija Oreskovic", "Dalija Orešković"]},
    {"name": "Marija Selak Raspudić", "notes": "Politika - Zastupnica u Saboru", "terms": ["Marija Selak Raspudic", "Marija Selak"]},
    {"name": "Nino Raspudić", "notes": "Politika - Zastupnik u Saboru", "terms": ["Nino Raspudic", "Nino Raspudić"]},
    {"name": "Sandra Benčić", "notes": "Politika - Zastupnica u Saboru / Možemo", "terms": ["Sandra Bencic", "Sandra Benčić"]},
    {"name": "Ivana Kekin", "notes": "Politika - Zastupnica u Saboru / Možemo", "terms": ["Ivana Kekin"]},
    {"name": "Miro Bulj", "notes": "Politika - Gradonačelnik Sinja", "terms": ["Miro Bulj", "Bulj portrait"]},
    {"name": "Zvonimir Troskot", "notes": "Politika - Zastupnik u Saboru", "terms": ["Zvonimir Troskot"]},
    {"name": "Ivan Anušić", "notes": "Politika - Ministar obrane RH", "terms": ["Ivan Anusic", "Ivan Anušić"]},
    {"name": "Davor Božinović", "notes": "Politika - Ministar unutarnjih poslova RH", "terms": ["Davor Bozinovic", "Davor Božinović"]},
    {"name": "Oleg Butković", "notes": "Politika - Ministar mora, prometa i infrastrukture RH", "terms": ["Oleg Butkovic", "Oleg Butković"]},
    {"name": "Branko Bačić", "notes": "Politika - Ministar graditeljstva RH", "terms": ["Branko Bacic", "Branko Bačić"]},
    {"name": "Damir Habijan", "notes": "Politika - Ministar pravosuđa i uprave RH", "terms": ["Damir Habijan"]},
    {"name": "Marin Piletić", "notes": "Politika - Ministar rada i mirovinskoga sustava RH", "terms": ["Marin Piletic", "Marin Piletić"]},
    {"name": "Vili Beroš", "notes": "Liječnik / Bivši ministar zdravstva RH", "terms": ["Vili Beros", "Vili Beroš"]},
    {"name": "Radovan Fuchs", "notes": "Politika - Ministar znanosti i obrazovanja RH", "terms": ["Radovan Fuchs"]},
    {"name": "Nina Obuljen Koržinek", "notes": "Politika - Ministrica kulture i medija RH", "terms": ["Nina Obuljen Korzinek", "Nina Obuljen"]},
    {"name": "Šime Erlić", "notes": "Politika - Ministar regionalnoga razvoja RH", "terms": ["Sime Erlic", "Šime Erlić"]},
    {"name": "Marko Primorac", "notes": "Politika - Ministar financija RH", "terms": ["Marko Primorac"]},
    {"name": "Kolinda Grabar-Kitarović", "notes": "Politika - Bivša predsjednica RH", "terms": ["Kolinda Grabar-Kitarovic", "Kolinda Grabar-Kitarović"]},
    {"name": "Stjepan Mesić", "notes": "Politika - Bivši predsjednik RH", "terms": ["Stjepan Mesic", "Stjepan Mesić", "Stipe Mesic"]},
    {"name": "Ivo Josipović", "notes": "Politika - Bivši predsjednik RH", "terms": ["Ivo Josipovic", "Ivo Josipović"]},
    {"name": "Jadranka Kosor", "notes": "Politika - Bivša predsjednica Vlade RH", "terms": ["Jadranka Kosor"]},
    {"name": "Vladimir Šeks", "notes": "Politika - Bivši predsjednik Sabora", "terms": ["Vladimir Seks", "Vladimir Šeks"]},
    {"name": "Radimir Čačić", "notes": "Politika - Bivši potpredsjednik Vlade RH", "terms": ["Radimir Cacic", "Radimir Čačić"]},
    {"name": "Krešo Beljak", "notes": "Politika - Predsjednik HSS-a", "terms": ["Kreso Beljak", "Krešo Beljak"]},
    {"name": "Anka Mrak-Taritaš", "notes": "Politika - Predsjednica GLAS-a", "terms": ["Anka Mrak-Taritas", "Anka Mrak-Taritaš"]},
    {"name": "Stipo Mlinarić", "notes": "Politika - Zastupnik u Saboru / Vukovarski branitelj", "terms": ["Stipo Mlinaric", "Stipo Mlinarić Cipe"]},
    {"name": "Davor Bernardić", "notes": "Politika - Saborski zastupnik", "terms": ["Davor Bernardic", "Davor Bernardić"]},

    # =========================================================================
    # 2. ⚽🎾🏀 SPORT - HRVATSKA (35 osoba)
    # =========================================================================
    {"name": "Goran Ivanišević", "notes": "Sport - Wimbledon pobjednik / Tenis", "terms": ["Goran Ivanisevic", "Goran Ivanišević"]},
    {"name": "Marin Čilić", "notes": "Sport - US Open pobjednik / Tenis", "terms": ["Marin Cilic", "Marin Čilić"]},
    {"name": "Borna Ćorić", "notes": "Sport - Teniski reprezentativac", "terms": ["Borna Coric", "Borna Ćorić"]},
    {"name": "Donna Vekić", "notes": "Sport - Olimpijska viceprvakinja / Tenis", "terms": ["Donna Vekic", "Donna Vekić"]},
    {"name": "Blanka Vlašić", "notes": "Sport - Svjetska prvakinja u skoku u vis", "terms": ["Blanka Vlasic", "Blanka Vlašić"]},
    {"name": "Sandra Elkasević", "notes": "Sport - Olimpijska pobjednica u disku", "terms": ["Sandra Perkovic", "Sandra Elkasevic", "Sandra Perković"]},
    {"name": "Ivica Kostelić", "notes": "Sport - Četverostruki olimpijski viceprvak / Skijanje", "terms": ["Ivica Kostelic", "Ivica Kostelić"]},
    {"name": "Janica Kostelić", "notes": "Sport - Četverostruka olimpijska pobjednica / Skijanje", "terms": ["Janica Kostelic", "Janica Kostelić"]},
    {"name": "Zlatko Dalić", "notes": "Sport - Izbornik hrvatske nogometne reprezentacije", "terms": ["Zlatko Dalic", "Zlatko Dalić"]},
    {"name": "Miroslav Blažević", "notes": "Sport - Legendarni nogometni izbornik Ćiro", "terms": ["Miroslav Blazevic", "Miroslav Ciro Blazevic"]},
    {"name": "Robert Prosinečki", "notes": "Sport - Nogometna legenda", "terms": ["Robert Prosinecki", "Robert Prosinečki"]},
    {"name": "Zvonimir Boban", "notes": "Sport - Nogometna legenda", "terms": ["Zvonimir Boban"]},
    {"name": "Davor Šuker", "notes": "Sport - Zlatna kopačka SP 1998 / Nogomet", "terms": ["Davor Suker", "Davor Šuker"]},
    {"name": "Mario Mandžukić", "notes": "Sport - Nogometna legenda", "terms": ["Mario Mandzukic", "Mario Mandžukić"]},
    {"name": "Marcelo Brozović", "notes": "Sport - Nogometni reprezentativac", "terms": ["Marcelo Brozovic", "Marcelo Brozović"]},
    {"name": "Dominik Livaković", "notes": "Sport - Vratar hrvatske reprezentacije", "terms": ["Dominik Livakovic", "Dominik Livaković"]},
    {"name": "Bruno Petković", "notes": "Sport - Nogometaš Dinama i reprezentacije", "terms": ["Bruno Petkovic", "Bruno Petković"]},
    {"name": "Mislav Oršić", "notes": "Sport - Nogometni reprezentativac", "terms": ["Mislav Orsic", "Mislav Oršić"]},
    {"name": "Borna Sosa", "notes": "Sport - Nogometni reprezentativac", "terms": ["Borna Sosa"]},
    {"name": "Josip Šutalo", "notes": "Sport - Nogometni reprezentativac", "terms": ["Josip Sutalo", "Josip Šutalo"]},
    {"name": "Dejan Lovren", "notes": "Sport - Nogometna legenda", "terms": ["Dejan Lovren"]},
    {"name": "Domagoj Vida", "notes": "Sport - Nogometna legenda", "terms": ["Domagoj Vida"]},
    {"name": "Danijel Subašić", "notes": "Sport - Vratar SP 2018", "terms": ["Danijel Subasic", "Danijel Subašić"]},
    {"name": "Toni Kukoč", "notes": "Sport - NBA Hall of Fame košarkaš", "terms": ["Toni Kukoc", "Toni Kukoč"]},
    {"name": "Dino Rađa", "notes": "Sport - NBA Hall of Fame košarkaš", "terms": ["Dino Radja", "Dino Rađa"]},
    {"name": "Bojan Bogdanović", "notes": "Sport - NBA košarkaš (Hrvatska)", "terms": ["Bojan Bogdanovic", "Bojan Bogdanović NBA"]},
    {"name": "Dario Šarić", "notes": "Sport - NBA košarkaš", "terms": ["Dario Saric", "Dario Šarić"]},
    {"name": "Ivica Zubac", "notes": "Sport - NBA košarkaš LA Clippers", "terms": ["Ivica Zubac"]},
    {"name": "Tin Srbić", "notes": "Sport - Svjetski prvak u gimnastici", "terms": ["Tin Srbic", "Tin Srbić"]},
    {"name": "Martin Sinković", "notes": "Sport - Olimpijski pobjednik u veslanju", "terms": ["Martin Sinkovic", "Martin Sinković"]},
    {"name": "Valent Sinković", "notes": "Sport - Olimpijski pobjednik u veslanju", "terms": ["Valent Sinkovic", "Valent Sinković"]},
    {"name": "Filip Hrgović", "notes": "Sport - Teškaški boksački šampion", "terms": ["Filip Hrgovic", "Filip Hrgović"]},
    {"name": "Mirko Filipović", "notes": "Sport - Cro Cop / Legenda borilačkih sportova", "terms": ["Mirko Filipovic", "Mirko Cro Cop"]},
    {"name": "Ivano Balić", "notes": "Sport - Najbolji rukometaš svijeta svih vremena", "terms": ["Ivano Balic", "Ivano Balić"]},
    {"name": "Domagoj Duvnjak", "notes": "Sport - Kapetan hrvatske rukometne reprezentacije", "terms": ["Domagoj Duvnjak"]},

    # =========================================================================
    # 3. ⛪ RELIGIJA & DUHOVNI ŽIVOT - REGIONALNO (30 osoba)
    # =========================================================================
    {"name": "Dražen Kutleša", "notes": "Religija - Zagrebački nadbiskup i metropolit (HBK)", "terms": ["Drazen Kutlesa", "Dražen Kutleša"]},
    {"name": "Josip Bozanić", "notes": "Religija - Kardinal / Umirovljeni zagrebački nadbiskup", "terms": ["Josip Bozanic", "Josip Bozanić"]},
    {"name": "Mate Uzinić", "notes": "Religija - Riječki nadbiskup i metropolit", "terms": ["Mate Uzinic", "Mate Uzinić"]},
    {"name": "Đuro Hranić", "notes": "Religija - Đakovačko-osječki nadbiskup", "terms": ["Djuro Hranic", "Đuro Hranić"]},
    {"name": "Želimir Puljić", "notes": "Religija - Umirovljeni zadarski nadbiskup", "terms": ["Zelimir Puljic", "Želimir Puljić"]},
    {"name": "Vlado Košić", "notes": "Religija - Sisački biskup", "terms": ["Vlado Kosic", "Vlado Košić"]},
    {"name": "Ivan Šaško", "notes": "Religija - Pomoćni biskup zagrebački", "terms": ["Ivan Sasko", "Ivan Šaško"]},
    {"name": "Mijo Gorski", "notes": "Religija - Pomoćni biskup zagrebački", "terms": ["Mijo Gorski"]},
    {"name": "Petar Palić", "notes": "Religija - Mostarsko-duvanjski biskup", "terms": ["Petar Palic", "Petar Palić"]},
    {"name": "Tomo Vukšić", "notes": "Religija - Vrhbosanski nadbiskup i metropolit", "terms": ["Tomo Vuksic", "Tomo Vukšić"]},
    {"name": "Vinko Puljić", "notes": "Religija - Kardinal / Umirovljeni vrhbosanski nadbiskup", "terms": ["Vinko Puljic", "Vinko Puljić"]},
    {"name": "Franjo Komarica", "notes": "Religija - Umirovljeni banjolučki biskup", "terms": ["Franjo Komarica"]},
    {"name": "Željko Majić", "notes": "Religija - Banjolučki biskup", "terms": ["Zeljko Majic", "Željko Majić biskup"]},
    {"name": "Bože Radoš", "notes": "Religija - Varaždinski biskup", "terms": ["Boze Rados", "Bože Radoš"]},
    {"name": "Ranko Vidović", "notes": "Religija - Hvarski biskup", "terms": ["Ranko Vidovic", "Ranko Vidović"]},
    {"name": "Patrijarh Pavle", "notes": "Religija - Pokojni patrijarh srpski (1914-2009)", "terms": ["Patrijarh Pavle", "Patriarch Pavle"]},
    {"name": "Patrijarh Irinej", "notes": "Religija - Pokojni patrijarh srpski (1930-2020)", "terms": ["Patrijarh Irinej", "Patriarch Irinej"]},
    {"name": "Vladika Grigorije", "notes": "Religija - Episkop diseldorfski i cijele Njemačke (Durić)", "terms": ["Vladika Grigorije", "Grigorije Duric"]},
    {"name": "Mitropolit Joanikije", "notes": "Religija - Mitropolit crnogorsko-primorski (Mićović)", "terms": ["Mitropolit Joanikije", "Joanikije Micovic"]},
    {"name": "Episkop Irinej Bulović", "notes": "Religija - Episkop bački i profesor teologije", "terms": ["Irinej Bulovic", "Irinej Bulović"]},
    {"name": "Episkop Ignatije Midić", "notes": "Religija - Episkop braničevski i dekan PBF-a", "terms": ["Ignatije Midic", "Ignatije Midić"]},
    {"name": "Episkop David Perović", "notes": "Religija - Episkop kruševački", "terms": ["David Perovic", "David Perović"]},
    {"name": "Mitropolit Amfilohije", "notes": "Religija - Pokojni mitropolit crnogorsko-primorski (Radović)", "terms": ["Amfilohije Radovic", "Amfilohije Radović"]},
    {"name": "Episkop Atanasije Jevtić", "notes": "Religija - Pokojni episkop zahumsko-hercegovački i teolog", "terms": ["Atanasije Jevtic", "Atanasije Jevtić"]},
    {"name": "Mitropolit Hrizostom", "notes": "Religija - Mitropolit dabrobosanski (Jević)", "terms": ["Mitropolit Hrizostom", "Hrizostom Jevic"]},
    {"name": "Husein Kavazović", "notes": "Religija - Reis-ul-ulema Islamske zajednice u BiH", "terms": ["Husein Kavazovic", "Husein ef. Kavazovic"]},
    {"name": "Aziz Hasanović", "notes": "Religija - Muftija zagrebački i predsjednik Mešihata u RH", "terms": ["Aziz Hasanovic", "Aziz ef. Hasanovic"]},
    {"name": "Mustafa Cerić", "notes": "Religija - Bivši reis-ul-ulema Islamske zajednice", "terms": ["Mustafa Ceric", "Mustafa Cerić"]},
    {"name": "Mevlud Dudić", "notes": "Religija - Predsjednik Mešihata IZ u Srbiji", "terms": ["Mevlud Dudic", "Mevlud Dudić"]},
    {"name": "Nedžad Grabus", "notes": "Religija - Muftija sarajevski", "terms": ["Nedzad Grabus", "Nedžad Grabus"]},

    # =========================================================================
    # 4. 🎭 KULTURA, FILM & KAZALIŠTE - REGIJA (35 osoba)
    # =========================================================================
    {"name": "Goran Višnjić", "notes": "Kultura - Glumac (Hollywood / Hrvatska)", "terms": ["Goran Visnjic", "Goran Višnjić"]},
    {"name": "Rade Šerbedžija", "notes": "Kultura - Glumačka legenda i redatelj", "terms": ["Rade Serbedzija", "Rade Šerbedžija"]},
    {"name": "Rene Bitorajac", "notes": "Kultura - Glumac i voditelj", "terms": ["Rene Bitorajac"]},
    {"name": "Tarik Filipović", "notes": "Kultura - Glumac i TV voditelj", "terms": ["Tarik Filipovic", "Tarik Filipović"]},
    {"name": "Enis Bešlagić", "notes": "Kultura - Glumac i komičar", "terms": ["Enis Beslagic", "Enis Bešlagić"]},
    {"name": "Goran Navojec", "notes": "Kultura - Glumac", "terms": ["Goran Navojec"]},
    {"name": "Bojan Navojec", "notes": "Kultura - Glumac", "terms": ["Bojan Navojec"]},
    {"name": "Janko Popović Volarić", "notes": "Kultura - Glumac", "terms": ["Janko Popovic Volaric", "Janko Popović Volarić"]},
    {"name": "Zrinka Cvitešić", "notes": "Kultura - Glumica (London West End / Hrvatska)", "terms": ["Zrinka Cvitesic", "Zrinka Cvitešić"]},
    {"name": "Nataša Janjić", "notes": "Kultura - Kazališna i filmska glumica", "terms": ["Natasa Janjic", "Nataša Janjić"]},
    {"name": "Mustafa Nadarević", "notes": "Kultura - Glumačka legenda (Izet Fazlinović)", "terms": ["Mustafa Nadarevic", "Mustafa Nadarević"]},
    {"name": "Relja Bašić", "notes": "Kultura - Glumačka legenda (Gospon Fulir)", "terms": ["Relja Basic", "Relja Bašić"]},
    {"name": "Boris Dvornik", "notes": "Kultura - Glumačka legenda (Malo Misto / Velo Misto)", "terms": ["Boris Dvornik"]},
    {"name": "Fabijan Šovagović", "notes": "Kultura - Glumačka legenda", "terms": ["Fabijan Sovagovic", "Fabijan Šovagović"]},
    {"name": "Filip Šovagović", "notes": "Kultura - Glumac, dramatičar i redatelj", "terms": ["Filip Sovagovic", "Filip Šovagović"]},
    {"name": "Anja Šovagović Despot", "notes": "Kultura - Kazališna glumica", "terms": ["Anja Sovagovic", "Anja Šovagović"]},
    {"name": "Dragan Bjelogrlić", "notes": "Kultura - Glumac, redatelj i producent", "terms": ["Dragan Bjelogrlic", "Dragan Bjelogrlić"]},
    {"name": "Emir Kusturica", "notes": "Kultura - Filmski redatelj / Dvostruka Zlatna palma", "terms": ["Emir Kusturica"]},
    {"name": "Miloš Biković", "notes": "Kultura - Glumac", "terms": ["Milos Bikovic", "Miloš Biković"]},
    {"name": "Nikola Đuričko", "notes": "Kultura - Glumac (Stranger Things / Srbija)", "terms": ["Nikola Djuricko", "Nikola Đuričko"]},
    {"name": "Bogdan Diklić", "notes": "Kultura - Glumačka legenda", "terms": ["Bogdan Diklic", "Bogdan Diklić"]},
    {"name": "Miki Manojlović", "notes": "Kultura - Glumačka legenda", "terms": ["Miki Manojlovic", "Miki Manojlović", "Predrag Manojlovic"]},
    {"name": "Žarko Laušević", "notes": "Kultura - Glumačka legenda i književnik", "terms": ["Zarko Lausevic", "Žarko Laušević"]},
    {"name": "Nebojša Glogovac", "notes": "Kultura - Glumačka legenda", "terms": ["Nebojsa Glogovac", "Nebojša Glogovac"]},
    {"name": "Velimir Živojinović", "notes": "Kultura - Bata Živojinović / Legenda jugoslavenskog filma", "terms": ["Velimir Bata Zivojinovic", "Bata Zivojinovic"]},
    {"name": "Ljubiša Samardžić", "notes": "Kultura - Glumačka legenda (Smoki)", "terms": ["Ljubisa Samardzic", "Ljubiša Samardžić"]},
    {"name": "Milena Dravić", "notes": "Kultura - Glumačka diva", "terms": ["Milena Dravic", "Milena Dravić"]},
    {"name": "Dragan Nikolić", "notes": "Kultura - Glumačka legenda (Prle)", "terms": ["Dragan Nikolic", "Dragan Nikolić"]},
    {"name": "Mirjana Karanović", "notes": "Kultura - Glumica i redateljica", "terms": ["Mirjana Karanovic", "Mirjana Karanović"]},
    {"name": "Branka Katić", "notes": "Kultura - Glumica", "terms": ["Branka Katic", "Branka Katić"]},
    {"name": "Andrija Milošević", "notes": "Kultura - Glumac i komičar", "terms": ["Andrija Milosevic", "Andrija Milošević"]},
    {"name": "Danis Tanović", "notes": "Kultura - Filmski redatelj / Dobitnik Oscara", "terms": ["Danis Tanovic", "Danis Tanović"]},
    {"name": "Jasmila Žbanić", "notes": "Kultura - Filmska redateljica (Zlatni medvjed)", "terms": ["Jasmila Zbanic", "Jasmila Žbanić"]},
    {"name": "Milan Štrljić", "notes": "Kultura - Glumac", "terms": ["Milan Strljic", "Milan Štrljić"]},
    {"name": "Ksenija Pajić", "notes": "Kultura - Glumica", "terms": ["Ksenija Pajic", "Ksenija Pajić"]},

    # =========================================================================
    # 5. ⛪ RELIGIJA I DUHOVNI ŽIVOT SRBIJE & SPC (35 osoba)
    # =========================================================================
    {"name": "Sveti Nikolaj Velimirović", "notes": "Religija - Episkop ohridski i žički / Teolog i svetitelj", "terms": ["Nikolaj Velimirovic", "Nikolaj Velimirović", "Bishop Nikolaj"]},
    {"name": "Sveti Justin Popović", "notes": "Religija - Otac Justin Ćelijski / Dogmatičar i svetitelj", "terms": ["Justin Popovic", "Otac Justin Celijski", "Justin Popović"]},
    {"name": "Otac Tadej Štrbulović", "notes": "Religija - Arhimandrit Tadej Vitovnički / Duhovnik", "terms": ["Otac Tadej", "Tadej Vitovnicki", "Tadej Strbulovic"]},
    {"name": "Otac Kleopa Stefanović", "notes": "Religija - Iguman manastira Vavedenje", "terms": ["Otac Kleopa", "Kleopa Stefanovic"]},
    {"name": "Arhimandrit Tihon Rakićević", "notes": "Religija - Iguman carskog manastira Studenica", "terms": ["Tihon Rakicevic", "Tihon Rakićević", "Iguman Tihon"]},
    {"name": "Patrijarh German", "notes": "Religija - Patrijarh srpski (Hranislav Đorić, 1958-1990)", "terms": ["Patrijarh German", "Patriarch German"]},
    {"name": "Patrijarh Vikentije", "notes": "Religija - Patrijarh srpski (Vitomir Prodanov, 1950-1958)", "terms": ["Patrijarh Vikentije", "Vikentije Prodanov"]},
    {"name": "Patrijarh Gavrilo Dožić", "notes": "Religija - Patrijarh srpski (1938-1950)", "terms": ["Patrijarh Gavrilo", "Gavrilo Dozic"]},
    {"name": "Patrijarh Varnava", "notes": "Religija - Patrijarh srpski (Petar Rosić, 1930-1937)", "terms": ["Patrijarh Varnava", "Varnava Rosic"]},
    {"name": "Episkop Teodosije", "notes": "Religija - Episkop raško-prizrenski i kosovsko-metohijski (Šibalić)", "terms": ["Teodosije Sibalic", "Episkop Teodosije", "Teodosije Šibalić"]},
    {"name": "Episkop Justin Stefanović", "notes": "Religija - Episkop žički", "terms": ["Justin Stefanovic", "Episkop Justin zicki"]},
    {"name": "Episkop Jovan Mladenović", "notes": "Religija - Episkop šumadijski", "terms": ["Jovan Mladenovic", "Episkop Jovan sumadijski"]},
    {"name": "Episkop Arsenije Glavčić", "notes": "Religija - Episkop niški", "terms": ["Arsenije Glavcic", "Episkop Arsenije"]},
    {"name": "Episkop Vasilije Vadić", "notes": "Religija - Episkop sremski", "terms": ["Vasilije Vadic", "Episkop Vasilije sremski"]},
    {"name": "Episkop Lavrentije", "notes": "Religija - Dugogodišnji episkop šabački i valjevski (Trifunović)", "terms": ["Lavrentije Trifunovic", "Episkop Lavrentije"]},
    {"name": "Episkop Jerotej Petrović", "notes": "Religija - Episkop šabački", "terms": ["Jerotej Petrovic", "Episkop Jerotej"]},
    {"name": "Episkop Pahomije Gačić", "notes": "Religija - Episkop vranjski", "terms": ["Pahomije Gacic", "Episkop Pahomije"]},
    {"name": "Episkop Ilarion Lupulović", "notes": "Religija - Episkop novobrdski / Rastko Lupulović", "terms": ["Ilarion Lupulovic", "Rastko Lupulovic", "Episkop Ilarion"]},
    {"name": "Episkop Stefan Šarić", "notes": "Religija - Episkop remezijanski / Starješina Hrama Sv. Save", "terms": ["Stefan Saric", "Episkop Stefan remezijanski"]},
    {"name": "Episkop Aleksej Bogićević", "notes": "Religija - Episkop hvostanski / Iguman manastira Sv. Luke", "terms": ["Aleksej Bogicevic", "Episkop Aleksej"]},
    {"name": "Episkop Damaskin Grabež", "notes": "Religija - Episkop mohački", "terms": ["Damaskin Grabez", "Episkop Damaskin"]},
    {"name": "Episkop Sava Bundalo", "notes": "Religija - Episkop marčanski", "terms": ["Sava Bundalo", "Episkop Sava marcanski"]},
    {"name": "Episkop Petar Bogdanović", "notes": "Religija - Episkop toplički", "terms": ["Petar Bogdanovic", "Episkop Petar toplicki"]},
    {"name": "Episkop Atanasije Rakita", "notes": "Religija - Episkop mileševski", "terms": ["Atanasije Rakita", "Episkop Atanasije milesevski"]},
    {"name": "Episkop Jovan Ćulibrk", "notes": "Religija - Episkop pakračko-slavonski / Padobranac i povjesničar", "terms": ["Jovan Culibrk", "Episkop Jovan Culibrk"]},
    {"name": "Episkop Maksim Vasiljević", "notes": "Religija - Episkop zapadnoamerički i teolog", "terms": ["Maksim Vasiljevic", "Episkop Maksim"]},
    {"name": "Episkop Longin Krčo", "notes": "Religija - Episkop novogračaničko-srednjozapadnoamerički", "terms": ["Longin Krco", "Episkop Longin"]},
    {"name": "Episkop Mitrofan Kodić", "notes": "Religija - Episkop kanadski", "terms": ["Mitrofan Kodic", "Episkop Mitrofan"]},
    {"name": "Episkop Andrej Ćilerdžić", "notes": "Religija - Episkop austrijsko-švicarski", "terms": ["Andrej Cilerdzic", "Episkop Andrej"]},
    {"name": "Episkop Dositej Motika", "notes": "Religija - Episkop britansko-skandinavski", "terms": ["Dositej Motika", "Episkop Dositej"]},
    {"name": "Episkop Kirilo Bojović", "notes": "Religija - Episkop buenosajreski i južno-centralnoamerički", "terms": ["Kirilo Bojovic", "Episkop Kirilo"]},
    {"name": "Episkop Siluan Mrakić", "notes": "Religija - Episkop australijsko-novozelandski", "terms": ["Siluan Mrakic", "Episkop Siluan"]},
    {"name": "Episkop Justin Jeremić", "notes": "Religija - Episkop zapadnoevropski", "terms": ["Justin Jeremic", "Episkop Justin pariz"]},
    {"name": "Episkop Lukijan Pantelić", "notes": "Religija - Episkop budimski i administrator temišvarski", "terms": ["Lukijan Pantelic", "Episkop Lukijan"]},
    {"name": "Episkop Gerasim Popović", "notes": "Religija - Episkop gornjokarlovački", "terms": ["Gerasim Popovic", "Episkop Gerasim"]},

    # =========================================================================
    # 6. 🔬 ZNANOST, PODUZETNIŠTVO, MEDIJI & TRENERI (30 osoba)
    # =========================================================================
    {"name": "Mate Rimac", "notes": "Poduzetništvo - Inovator / Rimac Group & Bugatti", "terms": ["Mate Rimac"]},
    {"name": "Silvio Kutić", "notes": "Poduzetništvo - Suosnivač prvog hrvatskog jednoroga Infobip", "terms": ["Silvio Kutic", "Silvio Kutić"]},
    {"name": "Nenad Bakić", "notes": "Poduzetništvo - Poduzetnik, investitor i matematičar", "terms": ["Nenad Bakic", "Nenad Bakić"]},
    {"name": "Ivan Đikić", "notes": "Znanost - Molekularni biolog i znanstvenik", "terms": ["Ivan Djikic", "Ivan Đikić"]},
    {"name": "Korado Korlević", "notes": "Znanost - Astronom / Zvjezdarnica Višnjan", "terms": ["Korado Korlevic", "Korado Korlević"]},
    {"name": "Gordan Lauc", "notes": "Znanost - Profesor biokemije i genetičar", "terms": ["Gordan Lauc"]},
    {"name": "Boris Jokić", "notes": "Znanost - Znanstvenik i pedagog", "terms": ["Boris Jokic", "Boris Jokić"]},
    {"name": "Robert Knjaz", "notes": "Mediji - Legendarni TV autor i redatelj", "terms": ["Robert Knjaz"]},
    {"name": "Aleksandar Stanković", "notes": "Mediji - Urednik i voditelj Nedjeljom u 2", "terms": ["Aleksandar Stankovic", "Aleksandar Stanković"]},
    {"name": "Zoran Šprajc", "notes": "Mediji - TV urednik i novinar (Stanje nacije)", "terms": ["Zoran Sprajc", "Zoran Šprajc"]},
    {"name": "Andrija Jarak", "notes": "Mediji - Istraživački TV novinar", "terms": ["Andrija Jarak"]},
    {"name": "Goran Milić", "notes": "Mediji - Legendarni televizijski novinar i putopisac", "terms": ["Goran Milic", "Goran Milić"]},
    {"name": "Jovan Memedović", "notes": "Mediji - TV autor Sasvim prirodno / Kviz Potera", "terms": ["Jovan Memedovic", "Jovan Memedović"]},
    {"name": "Ivan Ivanović", "notes": "Mediji - TV voditelj i autor", "terms": ["Ivan Ivanovic", "Ivan Ivanović"]},
    {"name": "Zoran Kesić", "notes": "Mediji - TV autor i satiričar (24 minuta)", "terms": ["Zoran Kesic", "Zoran Kesić"]},
    {"name": "Senad Hadžifejzović", "notes": "Mediji - TV urednik i voditelj Centralnog dnevnika", "terms": ["Senad Hadzifejzovic", "Senad Hadžifejzović"]},
    {"name": "Lino Červar", "notes": "Sport - Najtrofejniji hrvatski rukometni izbornik", "terms": ["Lino Cervar", "Lino Červar"]},
    {"name": "Slavko Goluža", "notes": "Sport - Rukometna legenda i trener", "terms": ["Slavko Goluza", "Slavko Goluža"]},
    {"name": "Mirza Delibašić", "notes": "Sport - Legendarni košarkaš / Kinđe", "terms": ["Mirza Delibasic", "Mirza Delibašić"]},
    {"name": "Dražen Petrović", "notes": "Sport - Košarkaški Mozart / NBA Hall of Fame", "terms": ["Drazen Petrovic", "Dražen Petrović"]},
    {"name": "Krešimir Ćosić", "notes": "Sport - Košarkaška legenda / NBA Hall of Fame", "terms": ["Kresimir Cosic", "Krešimir Ćosić"]},
    {"name": "Velimir Perasović", "notes": "Sport - Košarkaški trener i reprezentativac", "terms": ["Velimir Perasovic", "Velimir Perasović"]},
    {"name": "Neven Spahija", "notes": "Sport - Košarkaški trener", "terms": ["Neven Spahija"]},
    {"name": "Aleksandar Petrović", "notes": "Sport - Aco Petrović / Košarkaški izbornik", "terms": ["Aleksandar Petrovic", "Aco Petrovic"]},
    {"name": "Nenad Bjelica", "notes": "Sport - Nogometni trener Dinama", "terms": ["Nenad Bjelica"]},
    {"name": "Igor Bišćan", "notes": "Sport - Nogometni trener i bivši igrač Liverpoola", "terms": ["Igor Biscan", "Igor Bišćan"]},
    {"name": "Niko Kovač", "notes": "Sport - Nogometni trener (Bayern / Monaco / Hrvatska)", "terms": ["Niko Kovac", "Niko Kovač"]},
    {"name": "Robert Kovač", "notes": "Sport - Nogometni trener i legenda reprezentacije", "terms": ["Robert Kovac", "Robert Kovač"]},
    {"name": "Igor Tudor", "notes": "Sport - Nogometni trener (Juventus / Marseille / Lazio)", "terms": ["Igor Tudor"]},
    {"name": "Slaven Bilić", "notes": "Sport - Nogometni trener i izbornik", "terms": ["Slaven Bilic", "Slaven Bilić"]}
]

SKIP_KEYWORDS = [
    "flag", "coat", "signature", "logo", "map", "icon", "stemma", "herb",
    "grb", "zastava", "potpis", "badge", "diagram", "stadium", "building"
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
        for lang in ["hr", "sr", "bs", "en"]:
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
        time.sleep(0.12)

    return saved_count, "Uspješno"

def main():
    print("=" * 70)
    print("  ARGUSFACE STUDIO - MASOVNI UNOS 200 JAVNIH OSOBA (HR, SRB, BiH)")
    print("=" * 70)
    print(f"Ciljani broj osoba: {len(PEOPLE)}")
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
    print("  MASOVNI UNOS 200 OSOBA JE USPJEŠNO ZAVRŠEN!")
    print("=" * 70)

if __name__ == "__main__":
    main()
