---
layout: post
title: "Does the Dog Die? Pet Safety by Movie Genre."
featured-img: does-the-dog-die
---

# The Viz

[www.doesthedogdie.com](www.doesthedogdie.com) is a website which hosts the outcome of pets in movies.
The site was created as a way for sensitive viewers to learn the outcome of pets before choosing to watch a movie.
I was curious to see how the genre of the movie affects the outcome of a dog die.

### <image>

# Dogs & Movies

This is my cat, his name is Chai, as in chai tea.
I love him – even though he gets in the way while I cook (you won’t even eat veggies silly cat!), or when he’s outside my door meowing at 5 am (because, fun fact: cats are crepuscular).
He will always be my adorable stinky cat that knows if I’ve had a bad day.
I think many people can relate to these feelings I have with Chai with their own special friend, which is why I thought www.doesthedogdie.com was such a nice site.
It’s quite nice to see over 3000 people contribute to a somewhat whimsy but helpful service directed towards audiences who might’ve lost a dear friend recently.
I thought I’d contribute to the site by taking a look at all of results on the site.

### <image of beanie>

# The Data
All data was obtained from www.doesthedogdie.com as well as the little dog icons on the visualization.
The data was collected a few months ago before I started microbrewdata and it has been updated a little since.

# The Code
For our code snippet we  start at a point in the analysis where  have the number of crying (c), happy (h) and sad (s) dog icons sorted into how many instances were found across the genres.

```python
print (df)

# status       c    h    s
# Crime      103  174   29
# Romance     95  205   32
# Animation   70  119   55
# Comedy     245  471  125
# Horror     327  205   41
# Adventure  177  209   96
# Thriller   351  302   63
# Mystery    110  139   24
# Drama      461  488   99
# Action     180  214   57
# Family      93  199  105
# Fantasy    103  174   59
# Biography   41   52    6
```

We want to perform an elementary (after you learn it!) operation, to find the average contribution of each status to a genre.
To do this we need to break it down into two efficient steps

1.
find how many instances are in each genre
2.
divide each genre by the value found in step 1

we can put these steps into a small function that does this with some simple Pandas commands.

```python
def averageAllRows(df):
    # create a separate row comprised solely of sums
    df['sums'] = df.sum(axis=1)
    # for each column in our dataframe divided it by the sum entry
    for col in df.columns:
        df[col] = df[col].divide(df['sums'])
    # return the dataframe and drop the temporary column we used to
    # hold all of our sum values
    return df.drop('sums', axis=1)

# call said function
df = averageAllRows(df)
```

Could it really of been that easy?!
Let’s take a look (and if you’re going to execute this code you can plot it as well).

```python
print (df)

# status            c         h         s
# Crime      0.336601  0.568627  0.094771
# Romance    0.286145  0.617470  0.096386
# Animation  0.286885  0.487705  0.225410
# Comedy     0.291320  0.560048  0.148633
# Horror     0.570681  0.357766  0.071553
# Adventure  0.367220  0.433610  0.199170
# Thriller   0.490223  0.421788  0.087989
# Mystery    0.402930  0.509158  0.087912
# Drama      0.439885  0.465649  0.094466
# Action     0.399113  0.474501  0.126386
# Family     0.234257  0.501259  0.264484
# Fantasy    0.306548  0.517857  0.175595
# Biography  0.414141  0.525253  0.060606

# or alternatively plot this if you're going to try executing this example
df.plot(kind='bar', stacked=True)
```

# Bonus!
Ever wondered how dog (or more generally pet) deaths affect movie ratings over the years?
I did, unfortunately I didn’t really see any patterns so I stop wondering shortly after!
Below is one of the outputs I looked at plotting IMDB Rating versus year for movies on www.doesthedogdie.com.
This was one of my intermediate graphs for this project to see if the rating/year/outcome topic might be interesting to pursue.
 The y-axis is the IMDB rating, x-axis is the year the movie was released, blue means the dog lives (‘h’ for ‘happy icon’), green means the dog gets injured (‘s’ for ‘sad icon’), and red means the dog dies (‘c’ for ‘crying icon’).

# What Did You Think?
Think less dogs should die in movies?
Want to tell me you’re favourite sad pet movie?
I’d love if you would add a comment below or tweet at microbrewdata.
As always please subscribe to the Twitter or Facebook pages.
Seeing others enjoy/critique my work is one of my favourite parts of this website and subscribing is a good way to never miss a post – or perhaps see a picture of Chai!
