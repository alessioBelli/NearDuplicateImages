# Near-duplicate images :mag_right:
Fetching similar images in (near) real time is an important use case of information
retrieval systems.
The reasons why this kind of system is needed are many:
- Copyright issues. After images are posted on the internet, someone might modify
them and repost them claiming that those images are his.
- Performance. The presence of near-duplicates affects the performance of the
search engines critically.
- Criminal investigation.
- Medical diagnosis.
- Storage optimization.
- Privacy.

The problem of finding near-duplicate images is very well known, therefore there are
already plenty of algorithms for this purpose.
All the algorithms we studied for the implementation of our system were somehow
incomplete: some of them required too much computational power, some others were
simply not accurate enough to be used alone, and others had some problems in particular
cases (see ORB with blurred images).
Our idea is to combine different methods in order to fill the weakness of one algorithm
with the accuracy of another. We choose to use three algorithms:
1. Histograms comparisons
2. Features comparisons
3. ORB

The proposed result will be the weighted sum of the three methods. The three weights are
part of the algorithm, but they could be changed if necessary.
Finally, whether two images are near-duplicates or not is decided using a threshold.
Therefore, the algorithm is regulated by 4 parameters, whose values are already proposed
by our implementation considering the best performance in our dataset.
Then we implemented two different search algorithms based on what situation we are in.
The first one consists in searching in a large dataset all the images that are
near-duplicates of another external image (useful for detection), the second one selects
all the groups of near-duplicate images within a folder (useful for storage optimization).

## How to run it :wrench:
In order to run the application, you need to install Python on your PC with several libraries, including:
- tensorflow
- flask
- openCV
- numpy
- pillow
- scikit_learn

After downloading the zip file from Github and unzipping it, to install the libraries you just need to execute on your terminal the 
`pip install -r requirements.txt` command in the path of the unzipped folder.

Finally, if you want to run the application, you have to execute the command `python3 theFinder.py`. Once you have run the application, to view the web-app you have to type in a search engine `localhost:4555`.

## Possible problems :name_badge:
If run on MacOS, the following error could be encountered `certificate verify failed: unable to get local issuer certificate`. In this case, just type on the terminal `open /Applications/Python\ <Python Version>/Install\ Certificates.command`