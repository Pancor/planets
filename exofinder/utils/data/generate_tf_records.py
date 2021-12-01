import logging
import multiprocessing
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import preprocess

OUTPUT_DIR = "./tfrecord"
INPUT_FILE = "q1_q17_dr24_tce_2021.07.15_10.03.07.csv"
NUM_TRAINING_SHARDS = 8
WORKER_PROCESSES = 2

_LABEL_COLUMN = "av_training_set"
_ALLOWED_LABELS = {"PC", "AFP", "NTP"}


def _process_tce(tce):
    all_time, all_flux = preprocess.read_light_curve(tce.kepid, "./light_curves")

    time, flux = preprocess.process_light_curve(all_time, all_flux)
    return preprocess.generate_example_for_tce(time, flux, tce)


def _process_file_shard(tce_table, file_name):
    process_name = multiprocessing.current_process().name
    shard_name = os.path.basename(file_name)
    shard_size = len(tce_table)
    tf.get_logger().info("%s: Processing %d items in shard %s", process_name, shard_size, shard_name)

    with tf.io.TFRecordWriter(file_name) as writer:
        num_processed = 0
        for _, tce in tce_table.iterrows():
            example = _process_tce(tce)
            if example is not None:
                writer.write(example.SerializeToString())

            num_processed += 1
            if not num_processed % 10:
                tf.get_logger().info("%s: Processed %d/%d items in shard %s", process_name, num_processed, shard_size,
                                     shard_name)

    tf.get_logger().info("%s: Wrote %d items in shard %s", process_name, shard_size,
                         shard_name)


def main(argv):
    del argv

    tf.io.gfile.makedirs(OUTPUT_DIR)

    tce_table = pd.read_csv(INPUT_FILE, comment="#")
    tce_table["tce_duration"] /= 24
    tf.get_logger().info("Read TCE CSV file: %s with %d rows.", INPUT_FILE, len(tce_table))

    allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
    tce_table = tce_table[allowed_tces]
    num_tces = len(tce_table)
    tf.get_logger().info("Filtered to %d TCEs with labels in %s.", num_tces, list(_ALLOWED_LABELS))

    np.random.seed(420)
    tce_table = tce_table.iloc[np.random.permutation(num_tces)]
    tf.get_logger().info("Randomly shuffled TCEs.")

    train_cutoff = int(0.8 * num_tces)
    val_cutoff = int(0.9 * num_tces)
    train_tces = tce_table[0:train_cutoff]
    val_tces = tce_table[train_cutoff:val_cutoff]
    test_tces = tce_table[val_cutoff:]
    tf.get_logger().info("Partitioned %d TCEs into training (%d), validation (%d) and test (%d)",
                         num_tces, len(train_tces), len(val_tces), len(test_tces))

    file_shards = []
    boundaries = np.linspace(start=0, stop=len(train_tces), num=NUM_TRAINING_SHARDS + 1, dtype=int)

    for i in range(NUM_TRAINING_SHARDS):
        start = boundaries[i]
        end = boundaries[i + 1]
        filename = os.path.join(OUTPUT_DIR, "train-{:05d}-of{:05d}".format(i, NUM_TRAINING_SHARDS))
        file_shards.append((train_tces[start:end], filename))

    file_shards.append((val_tces, os.path.join(OUTPUT_DIR, "val-00000-of-00001")))
    file_shards.append((test_tces, os.path.join(OUTPUT_DIR, "test-00000-of-00001")))
    num_file_shards = len(file_shards)

    num_processes = min(num_file_shards, WORKER_PROCESSES)
    tf.get_logger().info("Launching %d subprocesses for %d total file shards.", num_processes, num_file_shards)

    pool = multiprocessing.Pool(processes=num_processes)
    async_results = [
        pool.apply_async(_process_file_shard, file_shard)
        for file_shard in file_shards
    ]
    pool.close()

    for async_result in async_results:
        async_result.get()

    tf.get_logger().info("Finished processing %d total file shards", num_file_shards)


if __name__ == "__main__":
    tf.get_logger().setLevel(logging.INFO)
    tf.compat.v1.app.run(main=main)
