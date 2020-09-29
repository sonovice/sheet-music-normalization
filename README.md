# sheet-music-normalization
Correction of rotation and scaling of sheet music images.

## Quick Start
Clone this repository and make sure that all dependencies listed in `requirements.txt` are installed, e.g. using pip:
```bash
$ git clone https://github.com/sonovice/sheet-music-normalization.git
$ cd sheet-music-normalization
$ pip install -r requirements.txt
```
Perform Perform normalization on a bunch of png files in a folder:
```bash
$ python normalize.py /folder/with/images --dst normalized --db stats.sqlite
```

Check `python normalize.py -h` for more parameters.