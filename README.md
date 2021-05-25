# ACD_FrontEnds

# installs 



# for grakn 2.02

4. You can make an environment to isolate the required dependencies.

- make the environment

```
$conda create -n 'MyEnv' python=3.7
```

- activate the environment

```
$conda activate MyEnv
```
5. pip install the remaining requirements, the grakn-cleint version is important

```
pip3 install grakn-client==2.0.1
pip3 install wget pandas numpy
pip install untangle
```


# for holoviews/datashader

conda install -c pyviz holoviews bokeh
conda install dask
conda install -c conda-forge xarray dask netCDF4 bottleneck
conda install datashader
# you need to get around this error. ImportError: hammer_bundle operation requires scikit-image. Ensure you install the dependency before applying bundling.

conda install scikit-image=0.18.1 


# for plotly dash
conda install dash
pip install dash_bootstrap_components


# to discover latent processes and kill them
sudo lsof -i:8050
kill PID