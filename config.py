import logging, os

import matplotlib 

if os.path.exists("/home/stephanie/Dropbox/")==True:
    base_path = "/home/stephanie/code/python/k2phot/"
    logging.warning("Working on jaina")
else:
    base_path = "/vega/astro/users/sd2706/k2/c5/"
    logging.warning("Working on Yeti")
    matplotlib.use("agg")
