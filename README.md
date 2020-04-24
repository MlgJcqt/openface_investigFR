# openface_investigFR
Using Openface algorithm from https://cmusatyalab.github.io/openface to compare 2 folders of images and generate list of matching scores

Requires :
* Docker
* Python

## Build Docker Container

Go in your Dockerfile folder and build container

```
$ cd /mjacquet/Investigation/Script/      
$ docker build -t mjacquet/openface
```

Run container and link folders to use as script and data sources (to simplify further code)

```
$ docker run -v /home/mjacquet/Investigation/Data/:/data –v home/mjacquet/Investigation/Script/:/script -it mjacquet/openface bash
```

Run modified Openface face comparison script to compare images from 2 folders and generate scores list in result .csv file

```
python script/OpenFace_compare.py /data/POI/ /data/persons_database/ /data/results.csv
```
