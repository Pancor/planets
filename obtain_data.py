import pandas as pd
import os
import stat
from datetime import datetime

_BASE_URL = "http://archive.stsci.edu/pub/kepler/lightcurves"
_WGET_CMD = ("wget -q -nH --cut-dirs=6 -r -l0 -c -N -np -erobots=off "
             "-R 'index*' -A _llc.fits")

koi = pd.read_csv("q1_q17_dr24_tce_2021.07.15_10.03.07.csv", comment="#")

kepids = set()
for index, row in koi.iterrows():
    kepids.add(row["kepid"])

kepids_length = len(kepids)

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
file_name = "download-light-curves-" + current_time + ".sh"
with open(file_name, "w") as f:
    f.write("#!/bin/sh\n")
    f.write("echo 'Starting to download {} light curves'\n".format(kepids_length))

    for i, kepid in enumerate(kepids):
        if i and not i % 10:
            f.write("echo 'Downloaded {}/{}'\n".format(i, kepids_length))
        kepid = "{0:09d}".format(int(kepid))
        subdir = "{}/{}".format(kepid[0:4], kepid)
        download_dir = os.path.join(os.getcwd(), "light-curves", subdir)
        url = "{}/{}/".format(_BASE_URL, subdir)
        f.write("{} -P {} {}\n".format(_WGET_CMD, download_dir, url))

    f.write("echo 'Finished downloading {} KOI'".format(kepids_length))

os.chmod(file_name, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)





