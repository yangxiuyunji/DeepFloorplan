import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from utils.tf_record import read_record


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _create_dummy_tfrecord(path, size=2):
    """Create a minimal TFRecord file for testing."""
    image = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    wall = np.zeros((size, size, 1), dtype=np.uint8)
    close = np.zeros((size, size, 1), dtype=np.uint8)
    room = np.zeros((size, size), dtype=np.uint8)
    close_wall = np.zeros((size, size, 1), dtype=np.uint8)

    features = {
        "image": _bytes_feature(image.tobytes()),
        "wall": _bytes_feature(wall.tobytes()),
        "close": _bytes_feature(close.tobytes()),
        "room": _bytes_feature(room.tobytes()),
        "close_wall": _bytes_feature(close_wall.tobytes()),
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    with tf.io.TFRecordWriter(path) as writer:
        writer.write(example.SerializeToString())


def test_read_record(tmp_path):
    tfrecord_path = tmp_path / "dummy.tfrecords"
    _create_dummy_tfrecord(str(tfrecord_path), size=2)

    loader = read_record(str(tfrecord_path), batch_size=1, size=2)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        images, labels = sess.run([loader["images"], loader["labels"]])
        coord.request_stop()
        coord.join(threads)

    assert images.shape == (1, 2, 2, 3)
    # wall(1) + close(1) + room(9 one-hot) + close_wall(1) = 12
    assert labels.shape == (1, 2, 2, 12)

