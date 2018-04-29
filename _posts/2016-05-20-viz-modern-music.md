---
layout: post
title: "Visualizing a Modern History of Music"
featured-img: visual-history-music-matplotlib
---

# Music & Dance

### <Pulp Fiction Chubby Checkers Dance Scene>
"I do believe Marsellus Wallace, my husband – your boss –  told you to take me out and do whatever I want.
Now I wanna dance, I wanna win, I want that trophy, so dance good."

Pulp Fiction is my favourite movie of all time.
Amazing story, cast, performance and director makes it my go to movie for the "have you seen?" ice breaker.
Possibly the most iconic scene from the movie is that with Uma Thurman and John Travolta twisting it up on the Jack Rabbit Slim’s floor to Chubby Checker.
While this may be Uma Thurman’s most iconic dance scene John Travolta seems to be an unstoppable, slicked back, jet black hair,  force of nature when it comes to dancing through the musical ages: Michael, Hairspray, Pulp Fiction, Grease, Saturday Night Fever, and Urban Cowboy all feature Travolta lighting up (sometimes literally) the dance floor.
While a lot of people may get a chuckle out of some of his dance floor disco music scenes, we can ask ourselves if all “cool” dance movies are destined for the same thing.
Will we be cringing at Bring it On and Stomp the Yard as time goes on?
Probably not as quickly as with disco, if we take a look at the music these scenes are dancing too.

# The Viz
### <insert the viz>

By analysing a list of Billboard’s top year end 100 songs we can quantify modern movements of music of  by the share of music genres represented for each years most popular songs according to Billboard.
We can see Disco’s short dozen year glory and the more sustained emergence of rap/hip-hop since the 90’s.
Interestingly the movement of disco corresponds to some of the lowest amount of genre shares from pop music, with history repeating again later in the mid 2000’s, when Rap/HipHop was at it’s peak.
Soft and Hard Rock faired even worse than disco with them completely dying out by 2005 as opposed to disco’s somewhat small influence that continues to limp on with Madonna’s revival in 2005 and Bruno Mars and Maroon 5 getting songs stuck in our heads post-2010.

### <bruno mars gif>

# How was this Plot Made?

Wikipedia is a gold mine of lists, lists of lists and even lists of lists of lists.
One of these lists of lists happens to be Billboard’s Hot 100 songs which allows us to browse Wikipedia’s data pretty easily.
Even easier after a quick look at URL’s we can simply generate each page we want to scrape data from.
We begin by loading the necessary modules and parameters for our program.

```python
%matplotlib inline
# magic inline comment above and import necessary modules
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pylab
import matplotlib.pyplot as plt
import requests, cPickle, sys, re, os
from bs4 import BeautifulSoup

# set notebook visualization preferences
pylab.rcParams['figure.figsize'] = 32, 16
```

Followed by a script to exploit the patterns we find in the URL’s and try to extract all possible links from the list pages

```python
# define a function which tries to scrape all relevant info from the table, if not  
# present return a NaN         

def tryInstance(td, choice):
    try:
        # songs only have one wikipedia link but artists
        # may have more than one, we create a choice flag
        # to allow us to say whether to read the element
        # in as an individual element or a list
        if (choice == 0):
            return td.text 
        elif (choice == 1):
            links = [x['href'] for x in td.findAll('a')]
            if (len(links) != 0):
                return links
            else:
                return float('NaN')
    except:
        return float('NaN')
    
# find the first table on the page and for all table rows with nonzero entries      
# (caused by the header formatting) try to scrape as much info as possible          

pandaTableHeaders = ['year', 'pos', 'song', 'artists', 'song_links', 'artist_links']
def scrapeTable(year):
    # create a url and soup out of a year and create a list for holding table data
    url = 'https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_'+str(year)
    soup = BeautifulSoup(requests.get(url).content)
    table = []
    
    # create a soup table variable, due to different formatting which can occur
    # from "this article needs more citations" etc.
we hard code in 4 special 
    # exceptions which were found after the first scrape.
    #
    # THIS PART HAS TO BE CHANGED WITH AS WIKIPEDIA UPDATES ITSELF
    # IT ___WILL___ RETURN A LIST INDEX OUT OF RANGE ERROR IF NOT
    # ACCOUNTED FOR 
    #
    # the best way to account for this is to go through the full analysis,
    # wait to reach the end at which point if a page was falsely scrape
    # the year will not appear, then manually fix the problem here
    souptable = soup.find('table')
    if (year in [2006, 2012, 2013]):
        souptable = soup.findAll('table')[1]
    elif (year in [2011]):
        souptable = soup.findAll('table')[4]
        
    # for the given chosen table from above iterate through all rows and try
    # collecting elements from each
    for pos, tr in enumerate(souptable.findAll('tr')):
        tds = tr.findAll('td')
        if (len(tds) > 0):
            # we request access backwards (e.g.
-2) to avoid differences between
            # the earlier years which dedicate the first column to the position
            # they were in the year while the first columns change the last 
            # columns never do 
            toAppend = [
                year, pos, 
                tryInstance(tds[-2], 0), tryInstance(tds[-1], 0),
                tryInstance(tds[-2], 1), tryInstance(tds[-1], 1)
            ]
            table.append(toAppend)
            
    # create and return a dataframe out of the table data
    df = pd.DataFrame(table)
    df.columns = pandaTableHeaders
    return df

# iterate over all available years and dump the data to a pickle for later use      
dfs = pd.DataFrame(pandaTableHeaders).set_index(0).T
for year in xrange(1956, 2016):
    print year,
    dfs = dfs.append(scrapeTable(year))
    cPickle.dump(dfs.reset_index().drop('index', axis=1), open('wikipediaScrape.p', 'wb'))
```

With all the links stored in a dataframe we can load each of the wikipedia pages to extract information from the info table at the top right corner of each page.
Unfortunately this can get especially brutal when all these info tables are variable lengths, have different HTML nesting and incomplete data (how have people not categorized The Gorrilaz’s music?!).
To deal with this we find the table object in our code and save it as a string to be loaded later for analysis.
The advantage of this is two fold – it allows us to gather all necessary information from one run and also gives us the power of simply looking for key-words in music genres to help sort through the zoo of user-defined definitions.

```python
To deal with this we find the table object in our code and save it as a string to be loaded later for analysis.
The advantage of this is two fold – it allows us to gather all necessary information from one run and also gives us the power of simply looking for key-words in music genres to help sort through the zoo of user-defined definitions.


# load up the data frame from above and create empty columns to populate while      
# scraping                                                                          

dfs = cPickle.load(open('wikipediaScrape.p', 'rb'))
subjects = ['Genre', 'Length', 'Producer', 'Label', 'Format', 'Released', 'B-side']
for subject in subjects:
    dfs[subject] = float('NaN')

# similar to the tryInstance function above try to extract as much info as possible 
# replacing missed instances caught by exceptions as NaNs.
Further more the problem 
# is aggrevated by the fact that tables are poorly managed, things may exist or not 
# things may have typos, different names, different children nesting it's a         
# nightmare.
Further issues can occur when trying to save the soup format because   
# one can exceed the maximum number of permitted iterations                         
#                                                                                   
# http://stackoverflow.com/questions/32926299/how-to-fix-statsmodel-warning-maximum-
# no-of-iterations-has-exceeded                                                     
#                                                                                   
# so we save the string of HTML to process later                                    
def extractInfoTable(url):
    infoTable = []
    # try exceptions for headers, table rows and pages
    try:
        soup = BeautifulSoup(requests.get(url).content)
        for tr in soup.find('table').findAll('tr'):
            try:
                header = tr.find('th').text
                if (header == 'Music sample'):
                    # music sample indicates the end of info table if 
                    # found simply break out of the loop to save time
                    break
                try:
                    # else collect all info possible in the form of a
                    # soup 'tr' object for analysis later
                    trs = tr.findAll('td')
                    infoTable.append([header, trs])
                except:
                    noTrsFound = True
            except:
                noHeaderFound = True
    except:
        noPageFound = True
        
    # if an entry exists for a given subject:
    # SAVE THE STRING OF HTML
    # for processing later, not the object
    infoColumns = []
    for subject in subjects:
        instanceBool = False
        # try to find a related header of a subject, if found
        # break, else append a NaN
        for header, info in infoTable:
            if (subject in header):
                infoColumns.append([subject, str(info)])
                instanceBool = True
                break
        if (not instanceBool):
            infoColumns.append([subject, float('NaN')])

    # return all scraped information
    return infoColumns

# for all songs in the dataframe apply the above scraping function
for songIndex in xrange(0,dfs.shape[0]):
    # print status update
    print songIndex, dfs.ix[songIndex].year, dfs.ix[songIndex].song
    try:
        # try accessing a link 
        song_links = ['https://en.wikipedia.org' + x for x in dfs.ix[songIndex].song_links]
        # extract info
        infoTable = extractInfoTable(song_links[0])
        # for index and subject store infromation 
        for idx, subject in enumerate(subjects):
            dfs.loc[:,(subject)].ix[songIndex] = str(infoTable[idx][1])
        # if 100 songs are processed dump the data as a back up, restart later by manually
        # changing the xrange() parameters 
        if (songIndex % 100 == 0):
            cPickle.dump(dfs.reset_index().drop('index', axis=1), open('full_df.p', 'wb'))
    except(TypeError):
        print 'NaN link found'

# dump the final data frame
cPickle.dump(dfs.reset_index().drop('index', axis=1), open('full_df.p', 'wb'))
```

```python
# load up the data frame from above and create empty columns to populate while      
# scraping                                                                          

dfs = cPickle.load(open('wikipediaScrape.p', 'rb'))
subjects = ['Genre', 'Length', 'Producer', 'Label', 'Format', 'Released', 'B-side']
for subject in subjects:
    dfs[subject] = float('NaN')

# similar to the tryInstance function above try to extract as much info as possible 
# replacing missed instances caught by exceptions as NaNs.
Further more the problem 
# is aggrevated by the fact that tables are poorly managed, things may exist or not 
# things may have typos, different names, different children nesting it's a         
# nightmare.
Further issues can occur when trying to save the soup format because   
# one can exceed the maximum number of permitted iterations                         
#                                                                                   
# http://stackoverflow.com/questions/32926299/how-to-fix-statsmodel-warning-maximum-
# no-of-iterations-has-exceeded                                                     
#                                                                                   
# so we save the string of HTML to process later                                    
def extractInfoTable(url):
    infoTable = []
    # try exceptions for headers, table rows and pages
    try:
        soup = BeautifulSoup(requests.get(url).content)
        for tr in soup.find('table').findAll('tr'):
            try:
                header = tr.find('th').text
                if (header == 'Music sample'):
                    # music sample indicates the end of info table if 
                    # found simply break out of the loop to save time
                    break
                try:
                    # else collect all info possible in the form of a
                    # soup 'tr' object for analysis later
                    trs = tr.findAll('td')
                    infoTable.append([header, trs])
                except:
                    noTrsFound = True
            except:
                noHeaderFound = True
    except:
        noPageFound = True
        
    # if an entry exists for a given subject:
    # SAVE THE STRING OF HTML
    # for processing later, not the object
    infoColumns = []
    for subject in subjects:
        instanceBool = False
        # try to find a related header of a subject, if found
        # break, else append a NaN
        for header, info in infoTable:
            if (subject in header):
                infoColumns.append([subject, str(info)])
                instanceBool = True
                break
        if (not instanceBool):
            infoColumns.append([subject, float('NaN')])

    # return all scraped information
    return infoColumns

# for all songs in the dataframe apply the above scraping function
for songIndex in xrange(0,dfs.shape[0]):
    # print status update
    print songIndex, dfs.ix[songIndex].year, dfs.ix[songIndex].song
    try:
        # try accessing a link 
        song_links = ['https://en.wikipedia.org' + x for x in dfs.ix[songIndex].song_links]
        # extract info
        infoTable = extractInfoTable(song_links[0])
        # for index and subject store infromation 
        for idx, subject in enumerate(subjects):
            dfs.loc[:,(subject)].ix[songIndex] = str(infoTable[idx][1])
        # if 100 songs are processed dump the data as a back up, restart later by manually
        # changing the xrange() parameters 
        if (songIndex % 100 == 0):
            cPickle.dump(dfs.reset_index().drop('index', axis=1), open('full_df.p', 'wb'))
    except(TypeError):
        print 'NaN link found'

# dump the final data frame
cPickle.dump(dfs.reset_index().drop('index', axis=1), open('full_df.p', 'wb'))
```

Now with all the HTML strings we can begin analysing it.
I extract a list of key words when a music genre can be found and put them into a column comprised of what I ended up calling ‘dirty lists’ – lists filled with:  typos, non-uniform proper nouns, references, etc etc.

```python
# create a dictionary of genres with lists to group together
# e.g.
for the scope of this analysis 'folk' and 'country' are 
# consider the same genre of music
genreList = {
    'electronic': ['electronic'],
    'latin'     : ['latin'],
    'reggae'    : ['reggae'],
    'pop'       : ['pop'], 
    'dance'     : ['dance'],
    'disco'     : ['disco', 'funk'], 
    'folk'      : ['folk', 'country'],
    'r&b'       : ['r&b'],
    'blues'     : ['blues'], 
    'jazz'      : ['jazz'],
    'soul'      : ['soul'],
    'rap'       : ['rap', 'hip hop'],
    'metal'     : ['metal'], 
    'grunge'    : ['grunge'], 
    'punk'      : ['punk'],
    'alt'       : ['alternative rock'],
    'soft rock' : ['soft rock'],
    'hard rock' : ['hard rock'],
}

# load the dataframe and extract relevant genres

# we create a column of 'dirty' lists which is comprised of the elements of the HTML 
# we find and DOES NOT account for all the possible exceptions which could occur for 
# example the dirty lists contain entries with typos, references, and un-unifrom    
# capitilizatiom like:                                                              
# 5977                                    [Indie folk]                              
# 5978                               [Gangsta raptrap]                              
# 5979    [Pop[1][2][3]pop soul[2][3]R&B[3]hip hop[3]]                              
# 5980                                       [SoulR&B]                              
# 5981                                [PBR&Bcloud rap]                              
# but what we are interested is extracting relevant keywords from the messy string  
# to the final "pop" genre count by simply matching all lowercase instances         

df = cPickle.load(open('full_df.p', 'rb'))
def extractGenre(x):
    sx = str(x)
    try:
        dirtyList = [td.text.replace('\n', '') for td in BeautifulSoup(sx).findAll('td')]
        return dirtyList
    except:
        return float('NaN')
    
df['Genre'] = df['Genre'].apply(extractGenre)
# print df['Genre']
```

Finally we create flag columns of each music genre a song contains to make plotting songs easier.

```python
# create columns with genre boolenas 0 for not prsent 1 for present for each 
# key in the genre dictionary, create a copy and use the .loc[(tuple)] method
# to avoid slicing chain caveat

for key in genreList.keys():
    df[key] = 0
dfs = df.copy()

# for each genre in the genreList see if you can match a string, if possible
# flag it's specified genre column so we can plot with these boolean results 
# later, we also keep a running tally 
for genre in genreList:
    ans=0
    for idx in xrange(0, df.shape[0]):
        if (len(df.loc[(idx,'Genre')]) > 0):
            if (any([x in df.loc[(idx,'Genre')][0].lower() for x in genreList[genre]])):
                dfs.loc[(idx, genre)] = 1
                ans+=1
    print genre, ans
    sys.stdout.flush()
    
cPickle.dump(dfs, open('genre_df.p', 'wb'))
```
and finally printing them out after trying to tweak some parameters

```python
def averageAllRows(gdf):
    # create a separate row comprised solely of sums
    gdf['sums'] = gdf.sum(axis=1)
    # for each column in our dataframe divided it by the sum entry
    # prevent a divsion by zero
    for col in gdf.columns:
        gdf[col] = gdf[col].divide(gdf['sums']+1e-12)
    # return the dataframe and drop the temporary column we used to
    # hold all of our sum values
    return gdf.drop('sums', axis=1)


pylab.rcParams['figure.figsize'] = 32, 16
gdf = pd.DataFrame()
for g in genreList.keys():
     gdf[g] = df.groupby('year')[g].sum()
# custom arrangement of printing order
gl2 = [ 
    'jazz', 'blues', 'folk', 'soul', 'pop', 'disco', 'rap', 'soft rock',
    'hard rock', 'dance', 'r&b', 'alt', 'latin', 'reggae', 'electronic', 'punk', 
    'grunge', 'metal',
]
# reorder the data frame and average all rows 
gdf = gdf[gl2]
gdf = averageAllRows(gdf)

# create percentage bar plot
ax = gdf.plot(kind='bar', width=1,stacked=True, legend=False, cmap='Paired', linewidth=0.1)
ax.set_ylim(0,1)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

locs, labels = plt.xticks()
plt.setp(labels, rotation=90)

plt.show()
```

With a final output of

### <matplotlib plot>

The raw data is exported to LaTeX to create a slightly more polished look.

# What Did You Think?

Enjoy the plot?
Inspired to do your own analysis?
Have a question about the code?
Sad I didn’t use your favourite song as the musical box info comparison picture?
Please tweet at me on my twitter like the Facebook page or leave a comment below.
Receiving Feedback is one of the most rewarding part of create visualizations and I would love if you’d open a dialogue or share!
