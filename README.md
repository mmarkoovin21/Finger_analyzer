# Finger analyzer

Finger Analyzer je alat temeljen na Pythonu za obradu i analizu otisaka prstiju u svrhu biometrijske verifikacije. Obradom stvarnih i lažnih otisaka prstiju, alat trenira model te evaluira njegovu učinkovitost koristeći metrike poput točnosti (accuracy), stope lažnog odbijanja (FRR) i stope lažnog prihvaćanja (FAR).

## Preduvijeti
Za korištenje ovog projekta potrebno je sljedeće:
- Python (verzija 3.8 ili novija)
- Anaconda (za upravljanje okolinama)
- Visual Studio Code (za razvojni rad)

Potrebne Python biblioteke možete instalirati pomoću pip ili conda:
- numpy
- opencv-python
- scikit-learn
- shutil

## okolina i instalacija
1. Instalirajte Anacondu: Preuzmite i instalirajte [Anacondu](https://www.anaconda.com/).
2. Klonirajte repozitorij ili kopirajte datoteke projekta u svoj radni direktorij.
3. Ako je potrebno, instalirajte sve potrebne biblioteke s naredbom 
```
pip install numpy opencv-python scikit-learn
```
4. Postavite direktorije: Osigurajte sljedeću strukturu direktorija u glavnom direktoriju projekta:
```
./Real_Fingerprints_DB
./Fake_Fingerprints_DB
./Processed_Fingerprints
./Processed_Imposter_Fingerprints
```
5. Popunite Real_Fingerprints_DB i Fake_Fingerprints_DB stvarnim i lažnim otiscima prstiju u .tif formatu. Osobno korišteni Neurotechnology UareU (520 datoteka) za treniranje i dio FVC2000 DB1 B za testiranje (40 datoteka).

## Pokretanje
1.Pokrenite skriptu: Otvorite datoteku finger_analyzer.py u Visual Studio Codeu. Kliknite na gumb "Run Python File" ili koristite terminal za izvršavanje skripte:
```
python Finger_analyzer.py

```
2.Provjerite jesu li direktoriji `Real_Fingerprints_DB` i `Fake_Fingerprints_DB` popunjeni podacima o otiscima prstiju. Ako su prazni, program će vas obavijestiti da dodate potrebne datoteke.

