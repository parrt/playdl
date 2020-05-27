# Setup jupyter lab for use with AWS

Idea is to launch normally on AWS but use local browser to run the notebook interface. I tried opening port 8888 but no luck so I use port forwarding with ssh.

## Launch

Get the "*AWS deep learning (ubuntu 18.04) v29*" AMI for your instance.  Then a p2 or p3 instance.

## Connecting

Port forwarding to access jupyter on AWS from mac:

```
ssh -i ~/Dropbox/licenses/parrt.pem -L 8000:localhost:8888 ubuntu@54.151.101.201
```
## Upon entry

For pytorch:

```
source activate pytorch_latest_p36
```

For keras:

```
source activate tensorflow2_p36
```

## Needed installs / setup

**Git**

```
git config --global user.email parrt@antlr.org
git config --global user.name "Terence Parr"
```

Make sure to enable this to see progress bars for keras/tensorflow:

```
conda install nodejs
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

```
pip install -U tensorflow
pip install tqdm
pip install tensorflow_addons
```

For Bokeh:

```
conda install nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @bokeh/jupyter_bokeh
```

## Jupyter start up

Start on AWS:

```
jupyter lab --no-browser --port=8888
```

Then go to [http://localhost:8000?token=1d8158157...](http://localhost:8000) in mac browser
