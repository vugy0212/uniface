.venv\Scripts\python.exe -X utf8 -c "
import sys; sys.path.insert(0, 'c:/uniface-app/src')
import db

existing = set(p['name'].lower().strip() for p in db.get_all_persons())

candidates = [
    # --- 1. GLAZBENA SCENA - ŽIVUĆI IZVOĐAČI (40) ---
    'Zlatan Stipišić Gibonni', 'Marko Perković Thompson', 'Miroslav Škoro', 'Petar Grašo', 'Tony Cetinski',
    'Severina Vučković', 'Nina Badrić', 'Josipa Lisac', 'Gabi Novak', 'Tereza Kesovija',
    'Mišo Kovač', 'Mladen Grdović', 'Goran Karan', 'Davor Gobac', 'Neno Belan',
    'Boris Novković', 'Maksim Mrvica', 'Stjepan Hauser', 'Luka Šulić', 'Zdravko Čolić',
    'Dino Merlin', 'Halid Bešlić', 'Goran Bregović', 'Momčilo Bajagić Bajaga', 'Željko Joksimović',
    'Vlado Georgiev', 'Željko Samardžić', 'Miroslav Ilić', 'Lepa Brena', 'Haris Džinović',
    'Hari Varešanović', 'Saša Lošić', 'Željko Bebek', 'Alen Islamović', 'Mladen Vojičić Tifa',
    'Jelena Rozga', 'Maja Šuput', 'Vesna Pisarović', 'Ana Rucner', 'Damir Urban',

    # --- 2. POLITIKA & DIPLOMACIJA - ŽIVUĆI POLITIČARI (35) ---
    'Jakov Milatović', 'Milojko Spajić', 'Milo Đukanović', 'Dritan Abazović', 'Aleksa Bečić',
    'Andrija Mandić', 'Milan Knežević', 'Zdravko Krivokapić', 'Filip Vujanović', 'Vojislav Koštunica',
    'Vuk Drašković', 'Rasim Ljajić', 'Usame Zukorlić', 'Nenad Čanak', 'Boško Obradović',
    'Miloš Jovanović', 'Aleksandar Jovanović Ćuta', 'Radomir Lazović', 'Marinika Tepić', 'Miroslav Aleksić',
    'Srđan Milivojević', 'Saša Janković', 'Tomislav Karamarko', 'Zlatko Hasanbegović', 'Davor Ivo Stier',
    'Tonino Picula', 'Biljana Borzan', 'Karolina Vidović Krišto', 'Mislav Kolakušić', 'Ivan Vilibor Sinčić',
    'Željko Kerum', 'Vesna Pusić', 'Zlatko Mateša', 'Franjo Gregurić', 'Dražen Budiša',

    # --- 3. VRHUNSKI SPORT & TRENERI - ŽIVUĆI (40) ---
    'Ivan Rakitić', 'Vedran Ćorluka', 'Ivica Olić', 'Danijel Pranjić', 'Mladen Petrić',
    'Nemanja Vidić', 'Dejan Stanković', 'Branislav Ivanović', 'Aleksandar Kolarov', 'Dejan Savićević',
    'Predrag Mijatović', 'Stevan Jovetić', 'Stefan Savić', 'Dado Pršo', 'Alen Bokšić',
    'Igor Štimac', 'Stipe Pletikosa', 'Niko Kranjčar', 'Ante Rebić', 'Nikola Vlašić',
    'Lovro Majer', 'Luka Sučić', 'Martin Baturina', 'Marko Pjaca', 'Mario Pašalić',
    'Josip Juranović', 'Luka Jović', 'Strahinja Pavlović', 'Lazar Samardžić', 'Luka Dončić',
    'Goran Dragić', 'Nikola Vučević', 'Vlade Divac', 'Predrag Stojaković', 'Dejan Bodiroga',
    'Aleksandar Saša Đorđević', 'Predrag Danilović', 'Miloš Teodosić', 'Vasilije Micić', 'Nikola Kalinić',

    # --- 4. RELIGIJA & DUHOVNI ŽIVOT - ŽIVUĆI VELIKODOSTOJNICI (30) ---
    'Zdenko Križić', 'Milan Zgrablić', 'Milan Stipić', 'Antun Škvorčević', 'Juraj Jezerinac',
    'Slobodan Štambuk', 'Ante Ivas', 'Valter Župan', 'Ivo Martinović', 'Jure Bogdan',
    'Vjekoslav Huzjak', 'Tomislav Rogić', 'Roko Glasnović', 'Ivan Štironja', 'Rrok Gjonlleshaj',
    'Velečasni Zlatko Sudac', 'Don Damir Stojić', 'Episkop Vasilije Kačavenda', 'Episkop Georgije Đokić', 'Episkop Filaret Mićević',
    'Episkop Konstantin Đokić', 'Episkop Metodije Ostojić', 'Episkop Dimitrije Rađenović', 'Episkop Isihije Rogić', 'Arhimandrit Metodije Marković',
    'Arhimandrit Rafailo Boljević', 'Otac Joil Bulatović', 'Otac Dionisije Pantelić', 'Nusret Abdibegović', 'Salem Dedović',

    # --- 5. FILM, KAZALIŠTE & TV - ŽIVUĆI GLUMCI I AUTORI (30) ---
    'Slavko Štimac', 'Srđan Todorović', 'Dragan Mićanović', 'Sergej Trifunović', 'Branislav Lečić',
    'Vojin Ćetković', 'Nebojša Dugalić', 'Nenad Jezdić', 'Vuk Kostić', 'Milan Marić',
    'Miloš Timotijević', 'Goran Bogdan', 'Leon Lučev', 'Krešimir Mikić', 'Ozren Grabarić',
    'Goran Grgić', 'Dušan Kovačević', 'Slobodan Šijan', 'Goran Marković', 'Rajko Grlić',
    'Anica Dobra', 'Katarina Radivojević', 'Nataša Ninković', 'Hana Selimović', 'Sloboda Mićalović',
    'Mima Karadžić', 'Gordan Kičić', 'Bojan Dimitrijević', 'Nikola Rakočević', 'Amar Bukvić',

    # --- 6. ZNANOST, PODUZETNIŠTVO, OLIMPIJCI & MEDIJI - ŽIVUĆI (25) ---
    'Barbara Matić', 'Matea Jelić', 'Damir Martin', 'Giovanni Cernogoraz', 'Josip Glasnović',
    'Zrinka Ljutić', 'Leona Popović', 'Sara Kolak', 'Marijo Možnik', 'Željko Mavrović',
    'Emil Tedeschi', 'Branko Roglić', 'Davor Štern', 'Darko Rundek', 'Zoran Predin',
    'Vlatko Stefanovski', 'Dado Topić', 'Nele Karajlić', 'Mihael Zmajlović', 'Mirela Holy',
    'Maja Sever', 'Mojmira Pastorčić', 'Zoran Šprajc', 'Andrija Jarak', 'Jovan Memedović'
]

print(f'Total candidates: {len(candidates)}')
collisions = [c for c in candidates if c.lower().strip() in existing]
print(f'Collisions: {len(collisions)}', collisions)
duplicates = set([x for x in candidates if candidates.count(x) > 1])
print(f'Internal duplicates: {len(duplicates)}', duplicates)
"