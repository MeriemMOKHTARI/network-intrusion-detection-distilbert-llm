#  Détection d'Attaques Réseau à l'Aide d'un Modèle Fine-Tuné basé sur DistilBERT

## Introduction

Ce projet développe un modèle d'IA pour la détection binaire d'attaques réseau en utilisant des techniques de NLP appliquées aux données de trafic réseau. L'objectif est de transformer des logs réseau en texte structuré et de fine-tuner un modèle DistilBERT pour classer les flux comme normaux ou malveillants, dans le cadre de systèmes de détection d'intrusions (IDS).

## Architecture du Modèle

- **Modèle de base** : `distilbert-base-uncased`
- **Type** : Classification binaire (attaques vs normal)
- **Paramètres** : 66 millions
- **Tête de classification** : Couche linéaire pour 2 classes
- **Approche** : Fine-tuning sur données réseau structurées

## Dataset UNSW-NB15

### Caractéristiques
- **Source** : Université de New South Wales (2015)
- **Taille totale** : 257,673 exemples
- **Split** : Train (175,217) / Validation (30,921) / Test (51,535)
- **Distribution** : 68% attaques, 32% normal

### Colonnes Transformées en Texte
14 colonnes converties en format texte structuré :
proto=tcp | service=http | state=FIN | dur=0.5 | saddr=192.168.1.1 | ...


### Types d'Attaques
- Reconnaissance, Backdoors, DoS, Exploits
- Fuzzers, Generic, Shellcode, Worms
- **Classification binaire** : Toutes catégories vs normal

## Installation

```bash
# Cloner le repository
git clone https://github.com/MeriemMOKHTARI/network-intrusion-detection-distilbert-llm.git
cd detection-attaques-reseau

# Installer les dépendances
pip install -r scripts/requirements.txt


### Types d'Attaques
- Reconnaissance, Backdoors, DoS, Exploits
- Fuzzers, Generic, Shellcode, Worms
- **Classification binaire** : Toutes catégories vs normal

## Installation

```bash
# Cloner le repository
git clone https://github.com/MeriemMOKHTARI/network-intrusion-detection-distilbert-llm.git
cd detection-attaques-reseau

# Installer les dépendances
pip install -r scripts/requirements.txt

Licence
Ce projet utilise le dataset UNSW-NB15 libre de droits pour la recherche académique.

Auteurs
Mokhtari Hadjia Meriem