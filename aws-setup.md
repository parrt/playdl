# Setup jupyter lab for use with AWS

Idea is to launch normally on AWS but use local browser to run the notebook interface. I tried opening port 8888 but no luck so I use port forwarding with ssh.

Make sure to enable this to see progress bars:

```
$ conda install nodejs
$ jupyter nbextension enable --py widgetsnbextension
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

```
pip install -U tensorflow
pip install tqdm
pip install tensorflow_addons
```

Start on AWS:

```
jupyter lab --no-browser --port=8888
```

Port forwarding to access jupyter on AWS from mac:

```
ssh -i ~/Dropbox/licenses/parrt.pem -L 8000:localhost:8888 ubuntu@54.151.101.201
```

Go to [http://localhost:8000?token=1d8158157...](http://localhost:8000) in mac browser
