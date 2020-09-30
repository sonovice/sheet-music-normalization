# Sheet Music Normalization
This script removes global rotation in (scanned) sheet music images and tries to scale them to a given "musical size" by looking at the space between staff lines.

## Quick Start
Clone this repository and make sure that all dependencies listed in `requirements.txt` are installed, e.g. using pip:
```bash
$ git clone https://github.com/sonovice/sheet-music-normalization.git
$ cd sheet-music-normalization
$ pip install -r requirements.txt
```
Perform normalization on a bunch of png files in a folder:
```bash
$ python normalize.py /folder/with/images --dst normalized --db stats.sqlite
```

Result stats (and errors) can be logged in an SQLite database for easier inspection.

## Options
Check `python normalize.py -h` for more parameters:
```
usage: normalize.py [-h] [--dst DST] [--db DB] [--staff-height STAFF_HEIGHT]
                    [--prefix PREFIX] [--num NUM] [--seed SEED] src

Normalize sheet music images.

positional arguments:
  src                          path to root directory with source images

optional arguments:
  -h, --help                   show this help message and exit
  --dst DST                    path to root directory for result images
  --db DB                      path to sqlite database for results
  --staff-height STAFF_HEIGHT  target pixel height of staves
  --prefix PREFIX              prefix to be used in result image filenames
  --num NUM                    number of images to sample
  --seed SEED                  seed value for random sampling
  --skip                       skip images with unrealistic scaling estimates
```
