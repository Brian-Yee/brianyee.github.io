---
layout: post
title: "Anime Hair Extraction via Clustering & Haar Features"
featured-img: anime-hair
---

# Waifu is Laifu
Krieger is by far my favourite character in Archer.
Undoubtedly a satire on “Q” (the gadget wizard from James bond) he builds all the fun toys for Archer, Lana and the gang to go about using on their missions.
However, unlike Q, Krieger has many quirks that make him special such as loving partner Mitsuko Miyazumi which happens to be a hologram waifu.


For those who don’t know, waifu is a term used to denote an imaginary anime partner — generally in the shape of a body pillow with which one forms romantic attachment too.
While not every anime watcher has a waifu, the anime community still remains extremely dedicated to assigning and cataloguing personas to anime characters.
In particular [MyAnimeList.net](https://myanimelist.net/) which has some extremely detailed profiles including but not limited to: height, weight, birthdays, occupations, school, classes, favourite foods, likes, dislikes, and a picture for each.
However, I didn’t see a lot of documentation on hair colour/style, most likely because it is so immediately apparent from the picture.

Thus, I set out to create a semi-supervised algorithm to extract the regions of hair in a given anime portrait using clustering techniques.

The clustering techniques used were DBSCAN and k-means (plus convex hulls if you count those I guess?) and pre-trained facial detection algorithms.
This article is meant predominantly as a Python tutorial to see what one can do when fooling around with scikit-learn, NumPy and OpenCV.

# Summary of Approach
## Outline
In plain English the logic of this program is the following

1. Reduce noise in image
2. Find a face
3. Look where there is probably hair
4. Look for other pixels with colours like your supposed hair region

### <insert image>

The first step reduces the original image of many colours, shades and hues to 12 base colours which approximate the original image.
This will be obtained by a clustering process known as k-means.
The next step than searches uses obtainable information on the general location of the face to create a region defined to be the face composed mainly of skin.
This region is then extended to look above some hypothetical scalp-line to guess the most likely region where one can find hair in the image.

## Limitations
Before beginning it is useful to think about what limitations this approach would cause.
A few come to mind:

- Determining the perfect k-value is expensive but extremely important too many colours and they will be hard to group, too few colours and everything will be over grouped.
Other objects in the picture may have similar colour to hair and may be incorrectly identified as hair.
- If no face is detected the approach will not work.
- Indeed we will find cases where all of these happen but using a naive algorithm as a first step towards solving this problem allows us to profile the difficulty of the project. A human can run this program on various portraits and select which ones work/fail gaining valuable insight before moving on to more complicated ML methods. If we can live with that, we can now get down to the nitty gritties of implementing the approach.

# Writing the Code
All of this is put up here on my GitHub in the form of a notebook should you wish to run it alongside the comments and discussion on the webpage,

This tutorial focuses only on the clustering/data manipulation aspect of this problem it does not include anything on the scraping or pre-processing facial extraction.
If that part interests you it can be found on the GitHub repo. For now it should suffice to simply note that there is a folder called faces  which contains two subfolders pngs and jsons where pngs contains anime face shots and jsons contains corresponding json files with facial region information.
With that noted, we get started by shotgunning all the modules we need at the start.

```python
# machine learing and image processing
import cv2
from sklearn.utils import shuffle
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import ConvexHull
from matplotlib.path import Path

# import data manipulation pacakges
import pandas as pd
import numpy as np

# viewing preferecnes
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams

# miscenallous packages
from collections import Counter
import itertools
import os
```

The most confusing of these packages to install is cv2 which for the longest time had eluded me, until I found out that when installing with Anaconda it is hosted under the name as opencv even though it is imported as cv2.
Hopefully this small addendum can save a mild amount of pain for others later.

With the packages imported we then read in all the files and for the purpose of this example we will only deal with one filepath `f` leading to an image location.

```python
# read in and sort all files
files = sorted([x[:-4] for x in os.listdir('faces/pngs/')])

# select an example 
f = [x for x in files if 'Aoi-Akira' in x][0]
```

# Colour Reduction via K-Means Clustering

As mentioned earlier image reduction is performed using k-means. 
This is a fairly standard procedure and even an example in the scikit docs. 
For a brief primer on k-means I refer the reader to this succinct youtube video which explains it with an example. 
The following block reads in an image and storing it in a dictionary as the original representation of the image. 
The image is then passed to a function which reshapes the data from the numpy array representation of `(y, x, rgb)`  of shape `(168, 100, 3)` into an array of RGB values.
These RGB values are shuffled and passed to the scikit k-means wrapper, which tries it’s best to reduce an image into the requested `k=12` colours. 
The output passes the reduced RGB data reshaped back into the original image dimensions. 
A side by side comparison is shown to the right to visualize the approximation.

### <wrap text around this image>

```python
def reduceImage(x, NUM_COL):
    # create a working copy of image
    X = np.float64(x.copy().reshape(-1, 3))

    # train model to extract NUM_COL amount of dominant colours
    km = KMeans(n_clusters=NUM_COL, random_state=0)
    kmeans = km.fit(shuffle(X, random_state=0))
    colours = kmeans.cluster_centers_
    
    # return a reduced image only use the NUM_COL amount of colours
    predicted = np.array([colours[x] for x in kmeans.predict(X)])
    return colours, np.uint8(predicted.reshape(x.shape))


# read in and reduce image
img = {'orig': cv2.imread('faces/pngs/'+f+'.png').astype(np.uint8)}
colours, img['reduced'] = reduceImage(img['orig'], 12)

# visualize output
rcParams['figure.figsize'] = 8, 4
stacked = np.uint8(np.hstack([img['orig'], img['reduced']]))
plt.imshow(cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB))
plt.axis('off'), plt.show()
```

The approximation looks surprisingly good — it’s hard to believe it is formed of only `k=12` colours!
As a visual aid to the reader, the following block decomposes the image into the multiple colours which now represent the image.

```python
def splitImage(x, colours):
    """Modified from pyimagesearch.com/2014/08/04/opencv-python-color-detection."""
    boundaries = ((x-1, x+1) for x in colours)
    
    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # find the colors within the specified boundaries and apply mask
        mask = cv2.inRange(x, lower, upper)
        output = cv2.bitwise_and(x, x, mask=mask)
        yield output

# show the images
rcParams['figure.figsize'] = 40, 4

#show segmentations
stacked = np.hstack(splitImage(img['reduced'], colours))
plt.imshow(cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB))
plt.axis('off'), plt.show()
```

Split apart the process seems pretty remarkable.
It is clear that some of these regions are distinctly part of the hair giving our approach hope.
The next step is to collect the regions using an educated guess of where the hair is most likely to be.

### <Face boundar extraction>
# Face Boundary Extraction via OpenCV & Convex Hull

### <Extraction of skin region from colours obtained from k-means clustering.>

The facial properties obtained in the preprocessing portion of this project (as mentioned before the code will be provided at the end) return 4 values defining a bounding rectangle on the face given by:

- `(x, y)` the location of the upper right corner of the rectangle.
- `(w, h)` the dimensions of the rectangle.

The goal than is to find the outline of the face within the box.
The first step of which is identifying the skin colour which we assume to be the dominant colour located within the box.

```python
def removeSkin(img, f):
    """It puts the lotion in the basket or it gets the NaNs again."""
    # read in preaquired info
    fileName = 'faces/jsons/' + f + '.json'
    h, w, x, y = pd.read_json(fileName).squeeze()

    # using info obtain facial region
    M = np.zeros_like(img['orig'], dtype=np.uint32)
    M[y+h//4:y+3*h//4, x+w//4:x+3*w//4, :] = 1
    f = M*img['reduced']
    f = f[f.nonzero()].reshape(-1, 3)

    # extract skin from the dominant colour segmentation
    skinColour = np.array(Counter(map(tuple, f)).most_common(1)[0][:1])
    skin = next(splitImage(img['reduced'], skinColour))

    return skin

rcParams['figure.figsize'] = 10, 10
img['skin'] = removeSkin(img, f)
plt.imshow(cv2.cvtColor(np.hstack([img['orig'], img['skin']]), cv2.COLOR_BGR2RGB))
plt.axis('off'), plt.show()
```

While we were able to successfully extract the skin — it is riddled with sparse flecks and outlines.
This noise results from the edges colours acting as dislocated shades/accents rather than a representation of the colour region itself.
To remove these we use a technique called DBSCAN:

- (D)ensity
- (B)ased
- (S)patial
- (C)lustering of
- (A)pplications with
- (N)oise.

The idea behind DBSCAN is remarkably intuitive. The algorithm is as follows

1. Assign each pixel in our image is assigned a positional value (their position in the NumPy array)
2. For each point look at a surrounding neighbourhood governed by a circle of size  eps
3. Check how many pixels exist in this neighbourhood and introduce a `min_samps` cutoff to obtain a desirable minimum density of the neighbourhood. The output of running DBSCAN on the image is shown to the left by setting `SKIN_DEMONSTRATION=True` for demonstration purposes. After which it is set to False.

```python
def scanForFace(F, X, eps=5, min_samps=1/3, SKIN_DEMONSTRATION=False):
    """DBSCANS skin and compares to OpenCV facial region to detect facial skin."""
    # performa  DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samps*(eps**3)).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    
    # package indices and labels together
    labels = db.labels_[db.labels_ != -1].reshape(-1, 1)
    indices = X[db.labels_ != -1]
    LI = np.hstack([indices, labels])

    # group positions by labels
    clusters = (LI[np.where(LI[:, 3] == x)][:, :3] for x in set(LI[:, 3]))

    # create a set of indices which exist in the CNN Facial region
    F_set = set([tuple(x) for x in F])

    # check if a DBSCAN cluster intersects with the facial region
    facial_regions = (x for x in clusters if any(set(tuple(y) for y in x) & F_set))

    # for understanding show skin detection if requested, otherwise stack and return
    if (SKIN_DEMONSTRATION):
        return X[np.where(db.labels_ != -1)]
    else:
        return np.vstack(facial_regions)
```

Any clusters which remain after DBSCAN are then checked for intersection with the position of the facial region.
In this way we pick up the face and neck or hands positioned over faces as part of the approximate region of face.

Next we seek the convex hull — the polygon with minimal perimeter which can contain every single point in the cluster.
The region of the polygon is then filled with ones so as to create a solid mask of the face in the image.

```python
def extractFace(img, f, hull):
    """Sets up variables to pass to scanForFace() which contains the gory details."""
    # read in preaquired info
    fileName = 'faces/jsons/' + f + '.json'
    h, w, x, y = pd.read_json(fileName).squeeze()

    # get facial regions
    M = np.zeros_like(img)
    M[y+h//4:y+3*h//4, x+w//4:x+3*w//4, :] = 1
    facial = np.dstack(M.nonzero()).squeeze()

    # create a new mask
    M = np.zeros_like(img)
    pos = np.dstack(img.nonzero()).squeeze()

    # in a neighbourhood of five pixels we filter anything
    # which is less than 80 percent
    clust = scanForFace(facial, pos)
    
    if (not hull):
        indices = [clust[:, x] for x in range(3)]
        M[indices] = 1
    else:
        # create a polygon representation out of the convex hull
        hull = ConvexHull(clust[:, :2])
        polygon = clust[:, :2][hull.vertices]
    
        # create an indexing grid to compare if within the polygon
        nx, ny = img.shape[:2][::-1]
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.dstack([y.flatten(), x.flatten()]).squeeze()

        # check to see which points lie within the convex hull
        G = Path(polygon).contains_points(points).reshape(img.shape[:2])
        M[G.nonzero()] = 1

    return 255*M

# calculate various masks
img['face'] = extractFace(img['skin'], f, False)
img['hull'] = extractFace(img['skin'], f, True)
A = cv2.bitwise_and(img['reduced'], img['reduced'], mask=img['face'][:, :, 0])
B = cv2.bitwise_and(img['orig'], img['orig'], mask=img['hull'][:, :, 0])

rcParams['figure.figsize'] = 20, 5
plt.imshow(cv2.cvtColor(np.hstack([img['orig'], img['skin'], A, img['hull'], B]), cv2.COLOR_BGR2RGB))
plt.axis('off'), plt.show()
```

With a defined region of hair we now need to find the scalp-line where the hair is most likely to be positioned.
The following script performs this task by creating images `1.1` and `1.3` larger than the original size.
Note the position of an image being scaled lies along a diagonal of the vector defining the original position, as shown right.
Using this we should be able to devise a method to center the new images with enlarged masks around the face of the old image given by its `COM` (center of mass).

The `COM` of the face mask for our case is trivially the average of all `(x, y)` values for each pixel in the mask.
Using this, we recenter our images while placing a bias on re-scaling the y term, so as to exclude the area beneath the original COM which would have included the neck.

```python
def hairPerimeter(img):
    # get center mass of filled convex hull
    resized = [cv2.resize(img['hull'], (0,0), fx=x, fy=x) for x in [1.1, 1.3]]
    COM = [np.mean(x.nonzero()[:2], axis=1) for x in [img['hull']] + resized]
    D_COM = np.array(COM[1:]) - COM[0]
    coords = [list(resized[x].nonzero()) for x in range(2)]
    
    # realign using the difference in the center of mass (D_COM) with
    # a bias for above the head
    for e in range(2):
        coords[e][0] = (coords[e][0] - 1.5*D_COM[e][0]).astype(np.uint32)
        coords[e][1] = (coords[e][1] - 1*D_COM[e][1]).astype(np.uint32)

    coords = [[x.astype(np.uint8) for x in coords[y]] for y in range(2)]
    H, L = img['orig'].shape[:2]

    # limit y range (avoid out of bound errors)
    for e in range(2):
        coords[e][0] = [min(x, H-1) for x in coords[e][0]]
        coords[e][1] = [max(x, 0) for x in coords[e][1]]

    # limit x range (avoid out of bound errors)
    for e in range(2):
        coords[e][1] = [min(x, L-1) for x in coords[e][1]]
        coords[e][1] = [max(x, 0) for x in coords[e][1]]

    # get outline
    M = np.zeros_like(img['hull'])
    
    # carve out region of interest from the two sets of scaled coordinates
    M[coords[1]] = 1
    M[coords[0]] = 0
    return M

img['hairPerimeter'] = hairPerimeter(img)
plt.imshow(cv2.cvtColor(
    np.hstack([img['orig'], img['hairPerimeter']*img['orig']]),
    cv2.COLOR_BGR2RGB
))
plt.axis('off'), plt.show()
```

Finally, with this sample of hair we search the rest of the image for other pixels containing colours from the scalp-line.
The process is demonstrated on the right. Leftmost is the scalp-line region as obtained above.
The next image shows the all clustered colours which contribute to more than 5%  of the total area in the scalp-line.
The penultimate image restores the clustered colours to their original RGB values.
The final image, serves as a reminder of what we were extracting from.

```python
hairLine = (img['hairPerimeter']*img['reduced']).reshape(-1, 3)

# create structure [RGB, count] values
cols = np.array(list(Counter(map(tuple, hairLine)).most_common()[1:]))

# take colours comprising more than 5% of portion
cols[:, 1] /= cols[:, 1].sum()
cols = cols[cols[:, 1] > 0.05][:, 0]

# cast back into list of numpy arrays
cols = list(map(np.array, cols))

# get the predicted reduced output versus original output
stacked = np.sum(splitImage(img['reduced'], cols))
mask = np.zeros_like(stacked)
mask[stacked.nonzero()] = 1

# store variables
img['hairRedu_1'] = stacked
img['hairOrig_1'] = mask*img['orig']

# display first prediction
rcParams['figure.figsize'] = 20, 4
stacked = np.hstack([img['skin'], img['hairRedu_1'], img['hairOrig_1']])
plt.imshow(cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB))
plt.axis('off'), plt.show()
```

One might be tempted to ask “Was that really so hard? Do we really expect all those problems at the beginging to occur?”.
An appreciation for the nuances can be gained from viewing the raw RGB data.
Below all RGB values are plotted in HSV space, with a slight opacity added to quantify density.
The leftmost image is the full 3D space plot while the remaining images are 2D projections along each of the HSV axes.
In row order, they correspond to

1. The original colour distribution of the original image
2. The final colour distribution of the image obtained from the predicted hair region
3. The original colour distribution after k-means was performed
4. The final colour distribution of the k-means cluster describing the predicted hair region

The points at the origin `(0, 0, 0)` in each are simply the non-existant values and can be ignored from the plots.

### <insert plot>

We see then that the k-means clusters help to create a more organic boundary per se to separate via a non-linear decision boundary.
This is extremely useful if not necessary, given that the shades often change as the hair falls downward.
In the next part we will see cases where this is still not robust enough to recognize glares/shininess of certain hair regions.

# Viewing the Results
As mentioned before this technique is not perfect — it is merely meant to provide a starting point to pick out well-behaved images to form an intuition with which to work with on more complicated models after.
Below I have provided some examples sorted into 3 categories loosely assigned while sifting through some of the results.
In order of appearance they represent the:

1. Good
2. Comme ci comme ça
3. Just plain ugly

### <insert plots>

# Conclusion
I hope this project can serve as an example of the potential of OpenCV, NumPy and scikit-learn to those interested.
While far from perfect, in certain well seperable cases it seems to work well, while more complex palettes cause failure.
The next post will look into ML techniques to deal with this breakdown while improving upon the model.

