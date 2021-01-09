
# list of images here https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
# exmamples of scripts here https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html

docker build -t mix_nlp_jupyter .
docker run --rm --name mix_nlp_jupyter -p 8888:8888 -v "$PWD":/home/jovyan mix_nlp_jupyter
