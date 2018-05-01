---
layout: post
title: "Gotta Scrape 'em all"
featured-img: pokemon-by-pokedex-colour
---
# Meta
- See the converation on Reddit [here](https://www.reddit.com/r/dataisbeautiful/comments/3vnrjx/pokemon_plotted_by_pokedex_colour_entry_oc/)

# The Viz

Below is a plot of all current 720 Pokemon plotted in a bar chart based on Pokedex colour entry value.
Click the picture to see the full 5MB version in a new tab, or click here for a Google Drive link to the 100MB PDF if you require a massive resolution to work with (such as for desktop wallpapers or posters).

![](https://github.com/Brian-Yee/brian-yee.github.io/blob/master/assets/img/pics/pokemon-scrape/00-pokemon.png?raw=true)

# Pokemania
<img align="right" src="https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/pokemon-scrape/01-pokemon-battle.jpg" alt="Pokemon battle" style="width: 256px;"/>
Nothing, absolutely nothing, brings about more waves of childhood nostalgia to me than Pokemon.
With trading card games, TV shows/movies, video games and soon augmented reality mobile apps, it comes as no surprise that the Pokemon franchise is the second most successful video game franchise after Mario.
Certainly pokemon was an immersive culture back when I was a child, with journalists coining Pokemania as a way to describe a part of the young millennial zeitgeist.
Perhaps then, it is not too surprising that after so many hours of conditioning as a child, my first web scraping project turned out to be related to Pokemon.
Seventeen years later and still brainwashed to catch ’em all, I guess.



# The Code
The code for this program was written a year ago as a project for teaching myself web scraping.
It is written in  Python and uses the BeautifulSoup module to obtain data from Bulbapedia and visualize it with some simple LaTeX code.
As the code snippet for this post, I will provide an example of collecting data from the table of all pokemon on Wikipedia, as a demonstration of what one can do.

```python
from bs4 import BeautifulSoup
import requests, urllib
import pandas as pd

# Identify the url leading to your data
url = 'https://en.wikipedia.org/wiki/List_of_Pok%C3%A9mon'

# Request the information and use BeautifulSoup to organize the content
r = requests.get(url)
soup = BeautifulSoup(r.content)

# There is only one table on the page so it is very easy to find
table = soup.find('table')

# for each table entry we now need to record the data given to us
# we define an empty list of pokemon to hold all that data
listOfPokemon = []
for tr in enumerate(table.findAll('tr')):
    # skip the first row which is just none types due to formatting
    if (tr[0] > 0):
        # we use a list comprehension to store every column in a row
        # further more we store it as the text instead of HTML code
        rowEntries = [td.text for td in tr[1].findAll('td')]
        # add the entries to the list
        listOfPokemon.append(rowEntries)

# to make sure we got them all lets look at four of the entries
# say the eevee evolutions 
df_pokemon = pd.DataFrame(listOfPokemon)
df_pokemon.columns = ['num', 'name', 'japan', 'Johto', 'Hoenn', 'Sinnoh', 'Unova', 'Kalos1', 'Kalos2', 'Kalos3', 'prior']
print df_pokemon[132:136]

# OUTPUT
#      num      name     japan Johto Hoenn Sinnoh Unova Kalos1 Kalos2 Kalos3      prior  
# 132  133     Eevee    Eievui   184     –    163   091      –    077      –    133aEgg  
# 133  134  Vaporeon   Showers   185     –    164   092      –    078      –  133bEevee  
# 134  135   Jolteon  Thunders   186     –    165   093      –    079      –  133cEevee  
# 135  136   Flareon   Booster   187     –    166   094      –    080      –  133dEevee 
# OUTPUT
```

In the last column, we can see undesirable traits bleeding through the scraping process.
These can be readily fixed by applying a function to that column which strips everything before the beginning of a proper noun.

```python
def cleanEvolutionColumn(x):
    for i in enumerate(x):
        if i[1].isupper():
            return x[i[0]:]

# and then apply said function to the column we want to clean up
df_pokemon['prior'] = df_pokemon['prior'].apply(cleanEvolutionColumn)

print df_pokemon[132:136]

# OUTPUT
#      num      name     japan Johto Hoenn Sinnoh Unova Kalos1 Kalos2 Kalos3  prior  
# 132  133     Eevee    Eievui   184     –    163   091      –    077      –    Egg  
# 133  134  Vaporeon   Showers   185     –    164   092      –    078      –  Eevee  
# 134  135   Jolteon  Thunders   186     –    165   093      –    079      –  Eevee  
# 135  136   Flareon   Booster   187     –    166   094      –    080      –  Eevee 
# OUTPUT
```

Awesome!
We now have a whole table of pokemon to work with in the future!
A word of caution should be added here.
Web scraping can be an intensive process and possibly shut down smaller sites if done improperly, whenever you collect data from the internet please do it kindly so as not to disrupt the page owners.
Some pages also express wishes that you not scrape their data so if you would like to gather some info be sure to read and reflect on the terms of use.

# Bonus!

Still have some latent, nostalgic, Pokemania?
Check out these other awesome visualizations/organizations of Pokemon.
The website [pokepalettes.com](www.pokepalettes.com) break down the colour palettes of each Pokemon and pokedex.org is a comprehensive Pokedex!
I should also mention the free and largely fan based [Pokemon Zeta/Omicron](https://zo.p-insurgence.com).
Much of the community exists in the subreddit dedicated to it and for a long while fan feedback went into making an adult Pokemon game.
It kept much of the original Pokemon charm while incorporating grown up themes such as: harder mini-puzzles, elaborate trading, swearing, and even characters dying via human sacrifice to power up other Pokemon.

# What Did You Think?

Love the viz?
Think it’s sacrilege that throughout the article I wrote Pokemon instead of Pokémon?
Questions about how you could go about doing this for Digimon or Yu-Gi-Oh?
Have a burning desire to share your Pokemon dream team of only one colour (I’m thinking: Venomoth, Cloyster, Nidoking, Mewtwo, Arbok, Gengar)?
I’d love if you would add a comment below or tweet at microbrewdata.
As always please subscribe to the Twitter or Facebook pages.
Seeing others enjoy/critique my work is one of my favourite parts of this website and subscribing is a good way to never miss a post!
