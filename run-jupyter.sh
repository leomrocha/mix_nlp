
# list of images here https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
# exmamples of scripts here https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html


docker run --rm --name mix_nlp_jupyter -p 8888:8888 -v "$PWD":/home/jovyan jupyter/datascience-notebook:703d8b2dcb88
