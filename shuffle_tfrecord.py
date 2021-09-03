import tensorflow as tf

target_path = '/home/ai2020/ne6091069/P_learning/data/AWA2/tfrecord/none/train.tfrecords'
cardinality = 58176
target_dataset = tf.data.TFRecordDataset(target_path)
result_dataset = target_dataset.shuffle(cardinality, seed=42, reshuffle_each_iteration=False)
# Generate tfrecord writer
result_tf_file = '/home/ai2020/ne6091069/P_learning/data/AWA2/tfrecord/none/train_shuffle.tfrecords'
writer = tf.io.TFRecordWriter(result_tf_file)
count = 1
for serialized_example in result_dataset.take(-1):
    writer.write(serialized_example.numpy())
    # print(count)
    # count = count + 1