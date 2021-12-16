# Transpileur\_ML\_EMB
transpileur Python --> C pour construire un fichier C de prédiction basé sur une régression linéaire entrainé en Python avec sklearn.

## Setup:

Ce tutoriel suppose que vous etes sous linux.
Si jamais ce n'est pas le cas, cherchez comment lancer un environnement virtuel Python sur votre platforme,
ainsi que compiler et executer du code C avec GCC.

Sous Linux veuillez suivre ces commandes:

(sautez ces etapes dans le cas ou vous ne voulez pas d'environnement virtuel.)
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

## Fichiers crees:

* lin\_reg.sav: modele de la regression lineaire sauvegarde grace a Joblib.
* prediction.c: code C permettant la prediction.
