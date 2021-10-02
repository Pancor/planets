import matplotlib.pyplot as plt
import numpy as np
import kepler_io
import os.path
import tensorflow as tf

KEPLER_ID = 11442793
KEPLER_DATA_DIR = "./light_curves"
TFRECORD_DIR = "./tfrecord"

# file_names = kepler_io.kepler_filenames(KEPLER_DATA_DIR, KEPLER_ID)
# all_time, all_flux = kepler_io.read_kepler_light_curve(file_names)
#
# plt.plot(all_time[3], all_flux[3], ".")
# plt.show()
#
# for f in all_flux:
#     f /= np.median(f)
# plt.plot(np.concatenate(all_time), np.concatenate(all_flux), ".")
# plt.show()


def find_tce(kepid, tce_plnt_num, filenames):
    for filename in filenames:
        for record in tf.compat.v1.io.tf_record_iterator(filename):
            ex = tf.train.Example.FromString(record)
            if (ex.features.feature["kepid"].int64_list.value[0] == kepid and
                    ex.features.feature["tce_plnt_num"].int64_list.value[0] == tce_plnt_num):
                print("Found {}_{} in file {}".format(kepid, tce_plnt_num, filename))
                return ex
    raise ValueError("{}_{} not found in files: {}".format(kepid, tce_plnt_num, filenames))


filenames = tf.io.gfile.glob(os.path.join(TFRECORD_DIR, "*"))
ex = find_tce(KEPLER_ID, 1, filenames)
print(ex)
global_view = np.array(ex.features.feature["global_view"].float_list.value)
local_view = np.array(ex.features.feature["local_view"].float_list.value)
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
axes[0].plot(global_view, ".")
axes[1].plot(local_view, ".")
plt.show()
