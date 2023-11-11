from GoogleNews import GoogleNews

# Initialisation
googlenews = GoogleNews()
googlenews.set_lang('fr')
googlenews.set_period('7d')

# Recherche
googlenews.search('finance')

# Récupération des résultats
result = googlenews.result()
for news in result:
    print("-" * 50)
    print("Titre:", news['title'])
    print("Date:", news['date'])
    print("Description:", news['desc'])
    print("Lien:", news['link'])

# Nettoyage
googlenews.clear()