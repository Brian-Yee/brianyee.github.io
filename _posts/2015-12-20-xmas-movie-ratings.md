---
layout: post
title: "Nostalgiac Baby Boomers, Kinder Females & Love Actually"
featured-img: xmas-ratings
---

# Christmas
It’s not quite Christmas time in my life till my girlfriend and I participate in our annual tradition of couch forts, Glühwein and Christmas movie marathons.
Second only to snow, Christmas TV specials and movies define a tradition of knowing when it’s the season by watching feel good movies, but what is it that makes Christmas movies so nice?
An interesting perspective on a related topic was given by Randall Munroe when he showed that the top 20 Christmas  songs were released around the age when Baby Boomers were children (1946 to 1964).
It seems like a reasonable assumption that if Christmas music was found to be impacted by baby boomers other media forms could be as well.

![https://imgs.xkcd.com/comics/tradition.png](https://imgs.xkcd.com/comics/tradition.png)

# The Baby Boomers
The first thing to do is gather a list of Christmas movies to analyze from.
A quick Google search of “Christmas movie MEGALISTS” yields a few helpful results.
An excellent source was found from Brisbane Kids Every Christmas Movie Ever... list with a little over 400 movies listed – most importantly with dates to help distinguish between different remakes between the years.
It’s also a nice bonus that the data can be easily scraped.
With this list we can now check the Internet Movie Database (IMDB) for a corresponding rating for the movie and than plot it based on it’s year of release.

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/xmas/00-baby-boom.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/xmas/00-baby-boom.png)

Wow that’s not bad at all for a first look!
There is certainly a downward trend in recent years, and it seems that movies of years past fair a bit better.
Lets take a look at including male/female viewership ratings.

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/xmas/01-male-female-diff.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/xmas/01-male-female-diff.png)

Immediately two trends become apparent.

- Women rate higher on average than men
- Men seem to appreciate very old Christmas movies more than women

The first point seems reasonable I am willing to believe that on average female Christmas viewership is kinder than it’s male counterpart.
The second part however seems a bit odd, why should it be that men suddenly appreciate movies pre-1920 more than women?
Perhaps there are some reasons for this trend – more research could be done – but it seems like a better assumption to believe that the 7 earliest movies we have are outliers.
There is a notable gap in the amount of Christmas movies in this dataset after WWI and it could be of benefit to exclude these very very old Christmas movies.
Plotting the new data set our assumption immediately gains some credibility.

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/xmas/02-clean-male-female-diff.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/xmas/02-clean-male-female-diff.png)

Excluding very very old Christmas movies results in a predicted trend of women consistently rating Christmas movies higher than men.
It seems much more likely that women rate kinder on average for all movies – instead of kinder for all movies except those pre-WWI.
However a look at the confidence intervals (lighter colours around trend line) show that men have a much better trend prediction than women during early years.
This suggests the ratings for very very old Christmas movies are outliers for women and not necessarily for men.
Men also usually submit more ratings than women on IMDB which suggests that we would actually be okay to keep the original trend line.
However I decided to be conservative and remove the movies regardless due to the twenty year gap in the dataset.

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/xmas/03-clean-baby-boom.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/xmas/03-clean-baby-boom.png)

While not as extreme as the first plot, ratings are predicted to be higher if they were released around the time of the Baby Boomers infancy.
It would seem that the American Christmas movie tradition is affected by the Baby Boomers.

# What Genre do we Love (Actually)?

It seems a naturally extension to investigate the trends of movies over the years by genre as well.
A quick plot of trendlines for the 7 most common genres show a similar trend as before with negative convexity (frowny face trend) except for one genre.

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/xmas/04-romance.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/xmas/04-romance.png)

This rebel genre is the Romance category.
The years after of the love it or hate it movie Love Actually with its arguably creepy love confessions shows a small ~7% (0.22 over a range of ~3 IMDB range in recent yeas) increase in movie quality with movies like Joyeux Noel or The Most Wondeful Time of Year coming up starring arguably famous actors like Diane Kruger of Inglorious Basterds and Henry Winkler of Happy Days and Arrested Development acting in major roles.
Could it be that the only non-tacky modern Christmas movie genre which can attract well-known strong actors is Romance?
Perhaps – perhaps not – it certainly would be a Christmas gift to me if someone else investigated this in more depth!!

# Bonus!

Ever wondered what the most popular genres for Christmas Movies are?
Turns out it’s Family but Comedy has seen a large increase in modern years.
Below is the cumulative sum plotted versus year – that is everytime a movie is released it adds onto an ongoing tally of the total of movies released since it came out and plotted versus time.
The top 7 genres are plotted below.

![https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/xmas/05-cumulatives.png](https://raw.githubusercontent.com/Brian-Yee/brian-yee.github.io/master/assets/img/pics/xmas/05-cumulatives.png)
### <insert picture>

# What Did You Think?

Think Andrew Lincoln was sweet in Love Actually instead of creepy?
Did you just release by clicking the Andrew Lincoln link that Andrew Lincoln is Rick from The Walking Dead?
Do you think The Walking Dead should have a Christmas special with zombies wearing Christmas hats?
Do you think this analysis is as boring as one of The Walking Dead Zombies or actually interesting like The Walking Dead’s first few seasons.
Are you curious why I seem to be stuck in Walking Dead similes at the end of a Christmas Data Article?
Please tweet at me on my twitter like the Facebook page or leave a comment below.

# Merry Christmas & Happy Holidays Everyone!

And just remember when it comes to Christmas Movies, [nothing is too scary for kids.
It’s all about that culture](https://www.youtube.com/watch?v=ZlDi3umBzHY).

### <insert youtube video>
