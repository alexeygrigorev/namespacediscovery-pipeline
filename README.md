# Pipeline for Mathematical namespace discovery

- https://github.com/alexeygrigorev/namespacediscovery is just a bunch of ipython notebooks - we need to make the code from there runnable

input:

- [Mathematical Language Processing](https://github.com/TU-Berlin/project-mlp) for extracting identifiers from wikipedia (and arXiv)
- [Java Language Processing](https://github.com/alexeygrigorev/JLP) for extracting variable declarations from Java

output:

- namespaces (like [here](http://0agr.ru/wiki/index.php/Discovered_namespaces))

## Running It

    git clone https://github.com/alexeygrigorev/namespacediscovery-pipeline.git
    cd namespacediscovery-pipeline/src
    python pipeline.py


Modify `luigi.cfg` to set different configuration parameters 

## Dependencies 

- python2
- numpy 
- scipy
- scikit-learn
- nltk
- python-Levenshtein
- fuzzywuzzy
- rdflib
- luigi 


for PyData stack libraries such as numpy, scipy, scikit-learn and nltk 
it's best to use [anaconda](http://docs.continuum.io/anaconda/install) 
installer 

Not all dependencies are available on anaconda, use `pip` to install them:

    pip install python-Levenshtein
    pip install fuzzywuzzy
    pip install luigi
    pip install rdflib


see SETUP.md for an example how to set up the environment


## Datasets 

We use the following datasets as input:

- mlp ... 


Classification schemes:

- MSC (downloaded from http://cran.r-project.org/web/classifications/MSC.html)
- PACS (downloaded from https://github.com/teefax/pacsparser)
- ACM (Can be downloaded from http://www.acm.org/about/class/class/2012 and has parsable skos format http://dl.acm.org/ft_gateway.cfm?id=2371137&ftid=1290922&dwn=1)


