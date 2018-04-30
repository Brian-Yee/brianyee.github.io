---
layout: post
title: "Preferential Differences in Aspect Ratios of Subreddits"
featured-img: gb-lb-subreddits
---

# Abstract
By the end of this post, it is the hope that the enthusiastic reader will be able to implement a knn-algorithim using Python to classify images from different subreddits of their choice.

What your getting into

- ~10 minute read for the short version, 1hr for the long (nitty gritty) detailed version
- An example of a K nearest neighbours (knn) classifier
- How to download thousands of images from subreddits
- Learning about image processing and data manipulation in Python.

# Sweet Intentions
I wasn’t really born with a sweet tooth.
I just don’t like sweet things with few somewhat almost universal exceptions:

- Cheesecake of any kind
- Black rice pudding
- Sweet Caroline (bop, bop, bahhh)
- Eye-candy

That last point (arguably shoe horned in) is probably the most universal of them all.
Regardless of ear-drums and taste-buds varying from person to person, visually some people are so handsome or pretty we feel compelled to simply admire their beauty.
It is perhaps unsurprising then that the #107 and #265 top subreddits are PG-13 eye-candy based.
More surprising are their unique names /r/gentlemanboners and /r/ladyboners, pushing the limit of oxymoronic compound words.

Perhaps it is their names that has gathered so much intrigue to their little corner of the internet; over 360k and 150k men and women have subscribed to add sprinklings of eye-candy to their Reddit feeds.
Clearly these men and women (or at least the majority) are not subscribing simply for the name but the content provided by the subreddit.
It was with this intention that I set out to classify the images of each subreddit and see if I could learn something about the viewing preferences of men and women in relation to viewing the opposite sex along the way.

# Data Collection
First we need to collect images from each subreddit.
For that we use the absolutely fantastic program RedditImageGrab.
 If we want to download the top 1000 pictures from /r/ladyboners we would run the following command in the terminal.

```bash
python redditdl.py ladyboners ~/Pictures/ladyboners --sort-type topall --num 1000 --update
```

# Data Preparation
To prepare the data for sampling all images were reduced using PIL while maintaining aspect ratios to a minimum height or width of 256 pixels, and the remaining picture had the top 63 colours selected as shown below with Alison Brie and Hugh Jackman.
Before resizing the images the dimensions of the images were retained, this will turn out to be vastly more important, than the colours.
However for pedagogical purposes a complete analysis including colours is used for two reasons

kNN extends easily in higher dimensional spaces, while the addition of colours may not vastly improve the result there is a small amount gained by the additional features for little cost.
For other subreddits colour can be used with a relatively high success rate.
For example one can run this program on the /r/earthporn and /r/urbanhell subreddits and achieve a ~78% success rate using colour alone

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/00-alison-brie-hugh-jackman.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/00-alison-brie-hugh-jackman.png)

# Data Analysis
A thousand images were prepared from each subreddit and put into a Pandas dataframe for various manipulations.
A kNN model was trained using either the RGB or HSV and the original height and width of each image as a feature.
The images’ corresponding subreddit as a target where /r/gentlemanboners and /r/ladyboners were set to `0` and `1` respectively as their target value.
More specifically we train SciPy’s implementation of kNN

```python
knn = KNeighborsClassifier(
    algorithm='auto', 
    leaf_size=16, 
    metric='minkowski',
    metric_params=None,
    n_jobs=1,
    n_neighbors=15,
    p=2,
    weights='uniform'
)
```

using 2/3rds of our data set and test the remaining third.
 A 63×5  array is assessed where we have 63 different RGB or HSV values and two constant columns of height or width in our parameter space.
We then take a majority vote from the predicted results of the array to predict the final classification of the picture.
It is important that the amount of RGB and HSV present is an odd number to prevent ties in voting (e.g.
if `23` say it’s from /r/gentlemanboners and 40 say /r/ladyboners we predict the latter).

We are able to attain a `84%` success rate in classifying images.
Indicating that there is at least some clustering going on, encouraging us to search for distinguishing features between the two subreddits using the parameters of our model.
For readers who are unsure exactly what the knn algorithm is here is a great youtube video on how knn works.

# Data Inference

The RGB and HSV were plotted to see if there were any clear colour preferences.
As mentioned in the preparation section, little correlation between the sexes was found for colour.

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/01-rgb-hsv.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/01-rgb.hsv.png)

Clearly no clustering of significance is readily apparent here, implying that the majority of the classification is a consequence of the images’ dimensions.
To see this we plot the image dimensions in a 3D plot, setting the z direction as the mean value of targets For example, if the resolution `(1000×1100)` has `8` women and `4` men it would be plotted at `(1000,1100,0.66)` in space.
A colour map was used to help visualize the differences between the clusters.

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/02-scatter-plot-viz.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/02-scatter-plot-viz.png)

Wow!
Okay, now we’re getting somewhere there clearly to be some trends in the size of images submitted to each subreddit.
Lets quantify these differences a bit better by plotting the respective KDEs for resolutions predominantly held by men or women where predominantly” is defined as a mean less than `0.2` for men and greater than `0.8` for women.

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/03-kde.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/03-kde.png)

# Mining the Subreddit Rules and Viewer Preferences

<img align="right" src="https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/04-discovered-rule.png" alt="Color Reduction via k-means" style="width: 150px;"/>
And the differences beome readily clear.
/r/ladyboners clusters at a much lower resolution than /r/gentlemanboners.
In fact it seems peculiar and suspicious just how large all entries for /r/gentlemanboners are, a good scientist here takes a step back and asks if they have made a mistake.
An investigation of the subreddit shows we are – in fact – correct, and have actually found an underlying rule seperating the two subreddits.
Looking at the sidebar of /r/gentlemanboners we see a set of rules which govern the subreddit, one of which is a minimum picture size.
No such set of rules exist for /r/ladyboners!
Three other direct features jump out:
- A large portion of men submit very high resolution images
- People seem to like capping pictures at a 3000 pixel height resolution
- While many pictures scale for women vertically the same way as for men a stronger trend seems to exist for pictures submitted to/r/ladyboners to also scale horizontally 

The first point may be a by product of Rule IV of /r/gentlemanboners as shown in the pictures.
The second point is a bit more mysterious, I am not sure why 3000 is such a popular vertical resolution but perhaps if you know why this could be, leave a comment below this post.
The third point is the most interesting one in my opinion.
There seems to be a weak trend for women to submit/upvote  pictures with a wider aspect ratio on /r/ladyboners than their male counterparts on /r/gentlemanboners, however pinning down the reason (or proving the difference is even statistically relevant) is beyond the scope of this post.

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/05-aspect-ratio.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/05-aspect-ratio.png)

# The Nitty Gritties (knn and PIL in python)

Thus far, a general path to quantify the differences between the two subreddits: /r/gentlemanboners and /r/ladyboners has been laid out.
This section tackles the impementation of the code allowing the reader to reproduce/understand the code for their own purposes.
We begin assuming all images have been downloaded as laid out in the “Data Collection” section above.

Were going to need a few packages so lets load those in

```python
%matplotlib notebook
# what we will use to train the KNN classifier
from sklearn.neighbors import KNeighborsClassifier 
# visualization purposes
from mpl_toolkits.mplot3d import axes3d, Axes3D 
# to work with image data
from PIL import Image 

# used for visualization
import matplotlib.pyplot as plt, seaborn as sns 
# used for data manipulation
import numpy as np, pandas as pd
# convenient modules
import collections, os, pickle, random, colorsys, sys 
```

# Defining Functions
Okay great, now that we have the libraries lets write some functions in advance to use later.
 The first function is a short code snippet, which returns a dataframe with it’s indexes mixed up, this will be useful later when we wish to randomly split all the data into a training and testing set.
The next function relies on the Python PIL library to collect the image dimensions of a photo and than return a down-sampled version of the image.

```python
# return an assorted mix of rows from a dataframe
# taken from stack overflow:
# http://stackoverflow.com/questions/15923826/random-row-selection-in-pandas-dataframe
def some(x, n):
    return x.ix[random.sample(list(x.index), n)]

# prepare Image for processing
def prepareImage(im):
    # store the resolution of each image
    dx, dy = im.size[0], im.size[1]
    
    # resize image to mantain it's aspect ratio while reducing it to a 
    # smaller size more appropriate for processing
    if (im.size[1] < im.size[0]):
        im = im.resize((256,256*im.size[1]//im.size[0]), Image.ANTIALIAS)
    else:
        im = im.resize((256*im.size[0]//im.size[1],256), Image.ANTIALIAS)

    # return the image using the top 127 colours present and 
    # the original resolution of each image 
    return [dx, dy, im.convert("P", palette = Image.ADAPTIVE, colors = 127)]
```

Lets take it out for a spin

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/06-output-1.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/06-output-1.png)

and Hugh Jackman as another example

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/07-output-2.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/07-output-2.png)

# Splitting the Data

Okay great, so now we can load down-sampled images so our laptops don’t explode like hot potatoes during analysis.
The next thing to do is to gather a list of all image paths and split them into training and testing lists.

```python
# intialize an empty list which will eventually hold multiple dataframes
dfs = []

# for each case read in the list of images from the folder
for en, appearance in enumerate(['gentlemanboners', 'ladyboners']):
    dfs.append(pd.DataFrame([''.join([appearance, '/', x]) for x in os.listdir(appearance)], columns=['image']))
    dfs[-1]['target'] = en

# scope the dataframe variable 
df = None

# if one subreddit has more pics than the other take 
# an equal amount from each 
if (dfs[0].shape[0] < dfs[1].shape[0]):
    df = dfs[0].append(some(dfs[1], dfs[0].shape[0]))
else: 
    df = dfs[1].append(some(dfs[0], dfs[1].shape[0]))

# reset the index for bookkeeping
df = df.reset_index().drop('index', axis=1)

# take a random without repetition arrangement of the
# data frames present in the rows
df = some(df, df.shape[0])

# again reset the index for bookkeeping
df = df.reset_index().drop('index', axis=1)

# we now define a training dataframe and a testing dataframe
# the training dataframe contains two thirds of the images
# while the testing dataframe contains one third of the images
# reset the index for bookkeeping
df_train = df[:-df.shape[0]//3].reset_index().drop('index', axis=1)
df_test  = df[-df.shape[0]//3:].reset_index().drop('index', axis=1)
```

# Gathering Attributes

With these lists ready we can now extract the colours (and original dimensions) present in the down-sampled images and dump them into some pickles to make our lives easier for reading in later.

```python
# extract the coordinates in RGB and HSV and pickle them for later use
# in theory we only need RGB to attain HSV but we will just make our
# live easier and define two seperate dataframes
for purpose in ['training', 'testing']:
    # define dataframe of interest
    df = df_train if (purpose == 'training') else df_test 

    # iterate over every row of the dataframe taking
    # the colours present from our preparation
    # and storing them in an empty list 
    features = [[],[]]
    for en in range(df.shape[0]):
        # as a bench mark print out our progress
        if ((en+1)%10==0):
            print (purpose, en+1, '/', df['image'].shape[0])
            sys.stdout.flush()
            
        # prepare image
        row = df.ix[en]
        
        # avoid any moving pictures 
        if (all([x not in row.image for x in ['.mp4', '.webm', '.gif']])):
            im = Image.open(row.image)
            dx, dy, im = prepareImage(im)
            
            # extract RGB and HSV, conv is for convention
            for en, conv in enumerate(['RGB', 'HSV']):
                features[en].append([row.target, dx, dy, [x[1] for x in im.convert(conv).getcolors()]])

    # dump the lists into a folder of pickles
    for en, conv in enumerate(['RGB', 'HSV']):
        pickle.dump(features[en], open('pickles/'+purpose+conv+'.p', 'wb'))
```

# Cleaning the Structure

Currently everything is stored in a pickled list of lists with lists, really not the cleanest way to present the data so we should load them into dataframes.
We write a function to read in a pickle and return two data frames one with RGB and HSV values.

```python
# we define a function now to read in the dataframes for RGB and HSV
# dataframes.
we use a function so we can easily read in the testing
# data later
def readinRGBHSVpickles(purpose):
    # define an empty list of dataframes called cols, short for colours
    # this will become our training datasets for RGB and HSV
    dfs = [None, None]

    # again, as explained above, we deal with both RGB and HSV
    for en, conv in enumerate(['RGB', 'HSV']):
        # load in the pickled list and store it as a seperate dataframe
        dfs[en] = pickle.load(open('pickles/'+purpose+conv+'.p', 'rb'))
        df = pd.DataFrame(dfs[en], columns=['target','dx', 'dy', 'cols'])
        # reset the space in the list to be an empty dataframe
        dfs[en] = pd.DataFrame()

        # assign all colours taken from the list a gentlemanboner
        # or ladyboner quality, then unzip the list of lists 
        # and create a list of corresponding targets.
More 
        # advanced useres of python may prefer to use the 
        # zip package here instead
        for i in range(df.shape[0]):
            r  = [x[0] for x in df.iloc[i].cols]
            g  = [x[1] for x in df.iloc[i].cols]
            b  = [x[2] for x in df.iloc[i].cols]
            t  = [df.iloc[i].target for x in df.iloc[i].cols]
            dx = [df.iloc[i].dx for x in df.iloc[i].cols]
            dy = [df.iloc[i].dy for x in df.iloc[i].cols]
            p  = [i for x in df.iloc[i].cols]
            
            # append those to the empty dataframe we scoped earlier outside this loop 
            dfs[en] = dfs[en].append(pd.DataFrame([r,g,b,dx,dy,t,p]).T)

        # append the RGB or HSV dataframes into a list
        dfs[en].columns = list(conv) + ['dx', 'dy', 'target', 'picture']
    return dfs

# read in training data
dfs_train = readinRGBHSVpickles('training')
```

# Exploratory Data Analysis

Now we can plot the colours to see if there are anything of value from the RGB and HSV info we extracted

```python
# now we do some 3D plots, for ease we turn to our good friend
# stackoverflow to help us with 3D matplotlib plots
# https://stackoverflow.com/questions/3810865/matplotlib-unknown-projection-3d-error

for en, conv in enumerate(['RGB', 'HSV']):
    # create a figure and set a 3D axis
    fig = plt.figure()
    ax = Axes3D(fig)
    
    # convert our RGB and HSV to xyz coordinates and set
    # them as numpy arrays to be plotted
    xyz = list(conv)
    X = [x for x in dfs_train[en][xyz[0]]]
    Y = [x for x in dfs_train[en][xyz[1]]]
    Z = [x for x in dfs_train[en][xyz[2]]]
    T = ['red' if x==0 else 'blue' for x in dfs_train[en].target]

    # perform cylindrical coordinate transformations to
    # the HSV data  and than reswap them back in
    if (conv == 'HSV'):
        H, S = Y*np.cos(X), Y*np.sin(X)
        X, Y = H, S
    
    # create axis labels and shuffle the data so blue and red are
    # plotted in an interspersed fashion to prevent visual bias
    dfp = pd.DataFrame([X,Y,Z,T], index=['X','Y','Z','T']).T
    dfp = some(dfp, dfp.shape[0])
        
    # plot the distribution, using red for gentlemanboners and blue for ladyboners
    ax.scatter(dfp['X'], dfp['Y'], dfp['Z'], s=10, alpha=0.01, c=dfp['T'])
    plt.show()
```

The answer of which is no.
However for other subreddits this is not necessarily the case.
The use of colour alone can classify images between /r/earthporn and /r/urbanhell with a 78% success rate so if you are going to analyse two of your favourite subreddits don’t necessarily write this stage out.

# Training the KNN Model
Now we load in the testing dataframe…

```python
dfs_test = readinRGBHSVpickles('testing')
```

and than call upon SciPy to train the classifier...

```python
knns = [None, None]
for en, conv in enumerate(['RGB', 'HSV']):
    # explicitly highlight the X Y variables in our training set
    X_train = dfs_train[en][dfs_train[en].columns[:5]]
    Y_train = dfs_train[en]['target']

    # create and a KNN classifier
    knns[en] = KNeighborsClassifier(
        algorithm='auto', 
        leaf_size=10, 
        metric='minkowski',
        metric_params=None,
        n_jobs=1,
        n_neighbors=9,
        p=2,
        weights='uniform'
    )

    # train it using the training set of pictures
    knns[en].fit(X_train, Y_train)
```

# Run the KNN Model
and run it!
HOWEVER we have a choice here.
Either we take an individual vote from each colour and than classify the image based on a majority vote; OR we take all colours from the pictures and assess them directly in space averaging all votes with no rounding.
We have done the latter, one can remove a few lines of code and modify the below cell to try doing the prior.

```python
# creating holding containers
results, percents = [], []

for en, conv in enumerate(['RGB', 'HSV']):
    # perform a groupby to get all colours associated with each picture
    test_g = dfs_test[en].groupby('picture')
    
    # for each picture predict the target of each colour 
    correct = 0
    for group in test_g.groups:
        # get group
        g = test_g.get_group(group)

        # declare testing variables
        X_test = g[g.columns[:5]]
        Y_test = g['target']
        probabilities = knns[en].predict_proba(X_test)
        Y_pred_a = np.array([x[0] for x in probabilities]).mean()
        Y_pred_b = np.array([x[1] for x in probabilities]).mean()
        
        # compare results
        if (Y_pred_a > Y_pred_b):
            results.append(0==g.target.ix[0])
        else:
            results.append(1==g.target.ix[0])
        
        if (results[-1]):
            correct += 1
            
        # record the percent we got for each colour
        percent_correct = correct/len(Y_test)
        percents.extend(results) 

    # moment of truth: print out the success rate of each seperate point and 
    # each ensemble of points predicting the final image classification
    print (conv, 'point accuracy:', 100.0*np.array(percents).mean(), '%')
    print (conv, 'image accuracy:', 100.0*results.count(True)/len(results), '%')
```

and we get something like

```python
RGB point accuracy: 84.0338877289 %
RGB image accuracy: 85.27131782945736 %
HSV point accuracy: 83.840902131 %
HSV image accuracy: 83.72093023255815 %
```

# Inspecting Results
Cool so the knn algorithm seems to give adequate results, now it’s time to try and figure out why and reproduce the work at the very beginning of this post.
We know colour apparently didn’t give us much so lets plot the image dimensions.

```python
# create new 2D fig
fig = plt.figure()

# plot sizes
X = [x for x in dfs_train[en]['dx']]
Y = [x for x in dfs_train[en]['dy']]
T = [x for x in dfs_train[0].target]

# prep labels and shuffle data to prevent visual bias
dfp = pd.DataFrame([X,Y,T], index=['X','Y','T']).T
dfp = some(dfp, dfp.shape[0])

# plot the distribution, using red for LB and blue for GB
plt.scatter(dfp['X'], dfp['Y'], s=20, alpha=0.2, c=['r' if x==0 else 'b' for x in dfp['T']])
plt.show()
```

Mmmmm, not so clear.
Lets try plotting them by their mean value.
Where on the spectrum of [0,1] does a given image dimension lie?
Is it closer to men’s preferences or women’s preferences.

```python
# create new 3D fig
fig = plt.figure()
ax = Axes3D(fig)

# find dominant property for a given image dimension
dfp2 = dfp.groupby(['X', 'Y']).mean().reset_index()

# plot the distribution, using red for ladyboner and blue for gentlemanboner
ax.scatter(dfp2['X'], dfp2['Y'], dfp2['T'], s=100, alpha=0.5, c=dfp2['T'], cmap='coolwarm')
plt.xlim(0,6000)
plt.ylim(0,6000)
ax.set_zlim(0,1)
ax.set_xlabel(r'Image Width [Pixels]', size=12)
ax.set_ylabel(r'Image Height  [Pixels]', size=12)
ax.set_zlabel(r'Mean Target Value', size=12)

plt.show()
```

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/02-scatter-plot-viz.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/02-scatter-plot-viz.png)

# Summarizing our Findings
Okay so there are clearly preferences but still not clear why the knn works so well, let’s plot the KDEs to help clear it up a bit more.

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/03-kde.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/knn-image-class/03-kde.png)

Finally the trend shows itself.
It is clear that the knn works as well as it does because the large majority of women’s images have smaller resolutions then men’s.
There also seems to be a bit more of a trend for women to prefer or at least be indifferent to pictures with a wiser aspect ratio.
Pretty neat!
