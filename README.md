# Transpileur\_ML\_EMB
transpileur Python --> C pour construire un fichier C de prédiction basé sur une régression linéaire entrainée en Python avec sklearn.

## Setup:

Ce tutoriel suppose que vous êtes sous linux.
Si jamais ce n'est pas le cas, cherchez comment lancer un environnement virtuel Python sur votre platforme,
ainsi que compiler et exécuter du code C avec GCC.

Sous Linux veuillez suivre ces commandes:

(sautez ce bloc dans le cas où vous ne voulez pas d'environnement virtuel.)
```sh
virtualenv transpi_env
source transpi_env/bin/activate
pip install -r requirements.txt
```

```sh
python3 transpile_simple_model.py
gcc -o predictor prediction.c
./predictor
```

## Fichiers créés:

* lin\_reg.sav: modèle de la régression linéaire sauvegardé grâce à Joblib.
* prediction.c: code C permettant la prédiction.
