---
layout: post
title: "Show Recommendation by Television Tropes"
featured-img: television-tropes
---

What You’re Getting Into
This article demonstrates an implementation/example of the Jaccard index to calculate shows.
In this post you can expect:

To have a 15 minute read
An explanation of the Jaccard index
An implementation/example in SciPy, with a more intutive example output.
# The Golden age of Television
### post the golden age of television post


# There are two things I love in this world — everybody and television
### <bingeimage >
A fitting quote from Kenneth Parcel of 30 Rock, to succinctly describe the youtube compilation video “The Golden Age of Video”‘, as he sing talks it at the halfway point.
Like Mr.
Parcel, the creator must have a great love for videos to make a song out of a string of infamous and semi-obscure quotes.
As did one of his viewers who was inspired to remaster the whole video in HD.
However, the general public’s love of television doesn’t have to be nearly as extensive as creating youtube compilations or analysing every frame of a show.
The title of the video alludes to the real age we are currently living in “The Golden Age of Television”.

The Golden Age of Television is marked by a great improvement in aesthetics and story telling.
Shows like The Sopranos, The Wire, Breaking Bad and Mad Men elevated the quality of television spurring writers/directors to aspire to new heights.
Often, attributes of these shows have been mimicked to try to recreate their quality.
It has been noted that television shows are visually getting darker and dimmer.
We have recently seen a rise in longer television shows, leading us now to so called eight hour movies on Netflix or the practice of binge watching.
Today in the television world, supply and demand are both rapidly increasing and viewers are now left possibly more than ever, with the question: what am I going to watch next?’.

# Tropes: Tricks of the Trade
A trope is a writing device used to convey a message or idea to an audience in a manner which they can relate.
Consider the “Enhance Button” whereby any image can be magically resolved to an unparalleled resolution simply by pressing a magical button.

### <tv enhace button>

While super resolution imaging is an active field of computer science, I highly doubt we will be able to do “corneal imaging” anytime soon with available methods.
What we can do though, is create groupings of shows which use a similar set of tropes to let the viewer pick a new show to watch.
If you’re willing to forgive (or God forbid enjoy) a show which uses the Enhance Button — perhaps you’d be interested in other shows which also use enhancing buttons.

# Measuring Similarity

We now seek a way to relate a show by their tropes.
One method is to simply check how many tropes 2 shows share.
Consider the overly simplified example

| CSI: Miami                      | Archer                          |
| ---                             | ---                             |
| The Enhace Button               | Hunting the Most Dangerous Game |
| Fanservice                      | Fanservice                      |
| Hunting the Most Dangerous Game | One Bullet Left                 |
| Real Song Opening Theme         |                                 |
| Hot Scientist                   |                                 |

In total we see Archer shares 2 out of its 3 tropes with CSI.
Therefore, one might expect that fans of CSI may also enjoy Archer.
However, this comparison fails to encompass the size of each set (the amount of tropes each show has).
In this example, CSI has more tropes than Archer and we were bound to get lucky and find some overlap between them.
 To remedy this, we penalise shows with larger sets of tropes, by dividing the amount of tropes shared between both shows by the number of unique tropes between each.
This score is called the Jaccard Index and it is a measure of similarity between two shows or more generally sets.
Formally, we refer to tropes shared as the intersection of sets and the amount of unique tropes between each as the union of sets.
Mathematically, we write this similarity measure as

    \[ J = \frac{A \cap B}{A \cup B} = \frac{A \cap B}{A + B - A \cap B} \]

For this example, there are `6` unique tropes shared between each show, only `2` of which are shared, giving us a score of `2/6`.
Sometimes to reduce the number of computations the unions of the set can be expressed as the sum of the size of each set minus the intersection, in this case `(5 + 3 – 2)`.

# Applying the Jaccard Index
While implementing your own Jaccard Index isn’t particularly complex, the use of the SciPy’s module makes calculating a full distance matrix even easier.
We begin by collecting all relevant tropes for American TV shows from TVtropes and storing them in a giant matrix.
For illustrative purposes here, we will consider only four shows with 5 attributes each.
```python
df = pd.DataFrame([
    ['s1', 0,0,1,0,1],
    ['s2', 0,1,1,0,0],
    ['s3', 0,0,1,1,1],
    ['s4', 1,0,0,0,0]
])
df.columns = ['show', 't1', 't2', 't3', 't4', 't5']
df = df.set_index('show')
```

which creates the following dataframe

```python
      t1  t2  t3  t4  t5
show                    
s1     0   0   1   0   1
s2     0   1   1   0   0
s3     0   0   1   1   1
s4     1   0   0   0   0
```

where `1` denotes the presence of a trope and `0` indicates a lack thereof.
The rows are referred to as *hot vectors* and aid in SciPy calculating their sizes of unique entries.
If some other constants were used (e.g.
`2` and `3`) the result would change as SciPy would believe each non-zero element was a unique entry creating a length of `5` (go ahead try it!).
Applying the Jaccard index to form a distance matrix we obtain

```python
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, jaccard

dist = pdist(df, 'jaccard')
dist = 1 - pd.DataFrame(squareform(dist))
dist.index, dist.columns = df.index, df.index
```

resulting in the following symmetric matrix

```python
show        s1        s2        s3         s4
show                                  
s1    1.000000  0.333333  0.666667   0.000000
s2    0.333333  1.000000  0.250000   0.000000
s3    0.666667  0.250000  1.000000   0.000000
s4    0.000000  0.000000  0.000000   1.000000
```

we now arrange the shows in descending order of similarity for a given show such as “s3”

```python
list(dist['s3'].sort_values(ascending=0).index.values)[1:]
```

and obtain

```python
['s1', 's2', 's4']
```

# Results

That was fairly painless!
However simply showing numbers in minimal context makes it hard to appreciate the skills we’ve learned.
To aid in better understanding these groupings we show what happens when you apply the Jaccard Index to American TV tropes.
Hopefully, we’ve guessed which shows are most similar and cross our fingers that the viewer will enjoy a new show with the same flavours as the old one.

Since this was a rather fun/loose analysis, we don’t really have any firm training/testing data to compare with, so I’ll let you decide how it worked out.
Below are the results of the Jaccard Index suggester.

The large poster is the queried show, and the rest are chosen by sorting the highest Jaccard Index values in descending order.
Shows on the top row, read left to right, are the top 6 and beneath them are the remaining positions up to `12`.

### <insert onslaugh of images>'''

It doesn’t look half bad!
