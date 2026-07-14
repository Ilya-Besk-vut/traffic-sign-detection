# Klasifikace dopravních značek v reálném čase

Tento projekt se zaměřuje na vytvoření a natrénování hluboké neuronové sítě (CNN) pro rozpoznávání vybraných dopravních značek v reálném čase pomocí webkamery. Projekt obsahuje celý proces od návrhu a trénování modelu v Jupyter Notebooku až po výsledný skript pro živou klasifikaci.

## Klíčové vlastnosti
- **Klasifikace v reálném čase:** Skript zachycuje videozáznam z webkamery, zpracovává jednotlivé snímky (preprocessing) a okamžitě zobrazuje název detekované značky.
- **Hluboké učení (Deep Learning):** Konvoluční neuronová síť (CNN) navržená a natrénovaná v Pythonu pomocí knihovny TensorFlow/Keras.
- **Zpracování obrazu:** Využití knihovny OpenCV pro plynulý odběr videa, úpravu velikosti snímků (resizing) a normalizaci dat před vstupem do modelu.

## Struktura projektu
- `traffic_sign_detection.ipynb` — Jupyter Notebook obsahující načtení dat, preprocessing, architekturu modelu, proces trénování a evaluaci.
- `CNNCamera.py` — Hlavní Python skript, který spouští webkameru a provádí predikce v reálném čase.
- `Best_Sign_Class.keras` — natrénovaný model (uložené váhy).

## Datová sada a klasifikované značky
Model byl trénován na datové sadě **GTSRB (German Traffic Sign Recognition Benchmark)**. Celou datovou sadu si můžete stáhnout zde (https://benchmark.ini.rub.de/gtsrb_news.html).

Vzhledem k velikosti datasetu nejsou zdrojová data součástí tohoto repozitáře. Model byl specificky natrénován na klasifikaci následujících **7 typů dopravních značek**:

| :--- | :--- |
| `(1, 0, 0, 0, 0, 0, 0)` | **Main road** |
| `(0, 1, 0, 0, 0, 0, 0)` | **Give way** |
| `(0, 0, 1, 0, 0, 0, 0)` | **Stop** |
| `(0, 0, 0, 1, 0, 0, 0)` | **Traffic is prohibited** |
| `(0, 0, 0, 0, 1, 0, 0)` | **Entry is forbidden** |
| `(0, 0, 0, 0, 0, 1, 0)` | **Rough road** |
| `(0, 0, 0, 0, 0, 0, 1)` | **Road work** |
