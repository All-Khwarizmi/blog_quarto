
<h1 align='center'>Education 2.0</h1>
<div align='center'>
<img src="./readme.png" widht='500px' height='400px'  >

</div>

## Introduction
Blog créé avec le framework [Quarto](https://quarto.org).

## Visualisation
Le blog est disponible à l'adresse suivante: [Education 2.0](https://all-khwarizmi.github.io/blog_quarto/)

## Installation
Pour installer le blog, il faut d'abord installer le framework Quarto.

### Quarto CLI
Pour cela, il faut suivre les instructions sur le site de Quarto.
lien: [Quarto Get Started](https://quarto.org/docs/getting-started/)

### Vscode extension
Il est conseillé d'installer l'extension vscode de Quarto. Pour cela, il faut l'installer depuis le marketplace de vscode.

Ensuite, il faut cloner le dépôt git du blog. Pour cela, il faut ouvrir un terminal et taper la commande suivante:
```bash
git clone All-Khwarizmi/blog_quarto
```

## Environnement
Il est conseillé d'utiliser un environnement virtuel pour installer les dépendances du blog. Pour cela, il faut ouvrir un terminal et taper les commandes suivantes:
```bash
python -m venv blog_quarto  
source blog_quarto/bin/activate
```
ou avec conda:
```bash
conda create -n blog_quarto python=3.10
conda activate blog_quarto
```
Il faut ensuite installer les dépendances du blog. Pour cela, il faut ouvrir un terminal et taper la commande suivante:
```bash
pip install -r requirements.txt
```

## Utilisation
Pour utiliser le blog, il faut ouvrir un terminal et taper la commande suivante:
```bash
quarto preview
```
## Contribution
Pour contribuer au blog, il faut ouvrir un terminal et taper la commande suivante:
```bash
git checkout -b <nom_de_la_branche>
```
Il faut ensuite faire les modifications souhaitées. Une fois les modifications terminées, il faut taper les commandes suivantes:
```bash
git add .
git commit -m "<message_de_commit>"
git push origin <nom_de_la_branche>
```
Il faut ensuite aller sur le dépôt git du blog et créer une pull request. Une fois la pull request validée, les modifications seront visibles sur le blog.

### Ajouter un article

Pour ajouter un article, il faut créer un dossier dans le dossier `posts`. Dans ce dossier, il faut créer un fichier `index.qmd`. Dans le fichier `index.qmd`, il faut mettre le contenu de l'article. Pour voir un exemple, il faut regarder le dossier `posts/road_map`.

Pour en savoir plus sur comment créer et éditer un article, il faut regarder la documentation de Quarto: [Quarto Documentation](https://quarto.org/docs/websites/website-blog.html)

