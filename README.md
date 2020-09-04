# AG News Clustering with Autoencoder

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
![ag-news-ae-clustering](https://github.com/AFAgarap/ag-news-ae-clustering/workflows/ag-news-ae-clustering/badge.svg?branch=master)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-377/)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-382/)

## Overview

An autoencoder is a neural network that aims to reconstruct a given input. It
learns to do so by learning the most salient features of a data. These salient
features are encoded in the latent space, i.e. feature representation with
lower dimensions than the original feature space. We can use the latent code of
an autoencoder for downstream tasks such as classification, regression, and
clustering. In this simple example, we use the latent code representation of a
text for clustering using the k-Means algorithm. This work was not intended to
reach state-of-the-art performance, but rather to show that the latent code
from an autoencoder can be used for downstream tasks, as how we use features
from principal components analysis, linear discriminant analysis, and locally
linear embedding.

## Usage

It is recommended to create a virtual environment for this repository to
isolate the dependencies by the modules.

```buildoutcfg
$ virtualenv venv --python=python3  # we use python 3
$ pip install -r requirements.txt
```

Then, download the AG News dataset, the benchmark dataset we will use for
clustering.

```buildoutcfg
$ bash setup/download_ag_news
```

Next, train the autoencoder model, and cluster the AG News
dataset by running the `train.py` module.

```buildoutcfg
$ python train.py
```

The `train.py` exports the trained autoencoder model, the fitted clustering
model, and the TF-IDF vectorizer used to vectorize the text data. These assets
can be used for an inference at a later time.

## Sample inference

We can use the exported assets for inference by running the `run.py` module. It
will accept a quoted string as the text to cluster. Note that the quotation
marks are needed since the module is using `sys.argv` for getting the command
arguments.

```buildoutcfg
$ python run.py "brown in line after livingston sack preston , livingston have sacked allan preston as manager . the former hearts and st johnstone defender and his assistant , alan kernaghan , were dismissed after a run of seven defeats left the club"
[INFO] Loading the trained autoencoder model...
[SUCCESS] Trained autoencoder ready for use.
Input text: brown in line after livingston sack preston , livingston have sacked allan preston as manager . the former hearts and st johnstone defender and his assistant , alan kernaghan , were dismissed after a run of seven defeats left the club
Predicted cluster: 0
$ python run.py "lockheed profit jumps on it , jet demand , &lt p&gt \&lt /p&gt &lt p&gt new york ( reuters ) - no . 1 u . s . defense contractor lockheed\martin corp . &lt lmt . n&gt reported a 41 percent rise in quarterly\profit on tuesday , beating wall street forecasts , as demand\soared for its combat aircraft and information technology\services . &lt /p&gt "
[INFO] Loading the trained autoencoder model...
[SUCCESS] Trained autoencoder ready for use.
Input text: lockheed profit jumps on it , jet demand , &lt p&gt \&lt /p&gt &lt p&gt new york ( reuters ) - no . 1 u . s . defense contractor lockheed\martin corp . &lt lmt . n&gt reported a 41 percent rise in quarterly\profit on tuesday , beating wall street forecasts , as demand\soared for its combat aircraft and information technology\services . &lt /p&gt
Predicted cluster: 3
$ python run.py "japan narrowly escapes recession , new figures show japan ' s economy is barely staying out of recession with annual growth of just 0 . 2 in the third quarter . "
[INFO] Loading the trained autoencoder model...
[SUCCESS] Trained autoencoder ready for use.
Input text: japan narrowly escapes recession , new figures show japan ' s economy is barely staying out of recession with annual growth of just 0 . 2 in the third quarter .
Predicted cluster: 3
$ python run.py "google lowers its ipo price range , san jose , calif . - in a sign that google inc . ' s initial public offering isn ' t as popular as expected , the company lowered its estimated price range to between \$85 and \$95 per share , down from the earlier prediction of \$108 and \$135 per share . . . "
[INFO] Loading the trained autoencoder model...
[SUCCESS] Trained autoencoder ready for use.
Input text: google lowers its ipo price range , san jose , calif . - in a sign that google inc . ' s initial public offering isn ' t as popular as expected , the company lowered its estimated price range to between $85 and $95 per share , down from the earlier prediction of $108 and $135 per share . . .
Predicted cluster: 1
```

## License

```
AG News Clustering with Autoencoder
Copyright (C) 2020  Abien Fred Agarap

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
