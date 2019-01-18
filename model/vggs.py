
import tensorflow as tf
import sys
sys.path.append('..')


from utilities.ops import relu, batch_norm
from utilities.ops import conv2d, fc


print_separater="#########################################################"
eps = 1e-9


### implementation for externet as feature extractors
def vgg_16_net(image,
               batch_size,
               device,
               keep_prob,
               initializer,
               reuse=False,
               network_usage='-1',
               output_high_level_features=[-1]):
    is_training = False
    weight_decay = False
    return_str="Vgg16Net"
    weight_decay_rate = eps

    usage_scope = network_usage + '/ext_vgg16net'

    with tf.variable_scope(usage_scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        ## block 1
        conv1_1 = relu(batch_norm(x=conv2d(x=image, output_filters=64,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv1_1',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn1_1',
                                  parameter_update_device=device))

        conv1_2 = relu(batch_norm(x=conv2d(x=conv1_1, output_filters=64,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv1_2',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn1_2',
                                  parameter_update_device=device))

        pool1 = tf.nn.max_pool(value=conv1_2,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')


        ## block 2
        conv2_1 = relu(batch_norm(x=conv2d(x=pool1, output_filters=128,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv2_1',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn2_1',
                                  parameter_update_device=device))

        conv2_2 = relu(batch_norm(x=conv2d(x=conv2_1, output_filters=128,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv2_2',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn2_2',
                                  parameter_update_device=device))

        pool2 = tf.nn.max_pool(value=conv2_2,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')


        ## block 3
        conv3_1 = relu(batch_norm(x=conv2d(x=pool2, output_filters=256,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv3_1',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn3_1',
                                  parameter_update_device=device))

        conv3_2 = relu(batch_norm(x=conv2d(x=conv3_1, output_filters=256,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv3_2',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn3_2',
                                  parameter_update_device=device))

        conv3_3 = relu(batch_norm(x=conv2d(x=conv3_2, output_filters=256,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv3_3',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn3_3',
                                  parameter_update_device=device))

        pool3 = tf.nn.max_pool(value=conv3_3,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')

        ## block 4
        conv4_1 = relu(batch_norm(x=conv2d(x=pool3, output_filters=512,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv4_1',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn4_1',
                                  parameter_update_device=device))

        conv4_2 = relu(batch_norm(x=conv2d(x=conv4_1, output_filters=512,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv4_2',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn4_2',
                                  parameter_update_device=device))

        conv4_3 = relu(batch_norm(x=conv2d(x=conv4_2, output_filters=512,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv4_3',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn4_3',
                                  parameter_update_device=device))

        pool4 = tf.nn.max_pool(value=conv4_3,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')


        ## block 5
        conv5_1 = relu(batch_norm(x=conv2d(x=pool4, output_filters=512,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv5_1',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn5_1',
                                  parameter_update_device=device))

        conv5_2 = relu(batch_norm(x=conv2d(x=conv5_1, output_filters=512,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv5_2',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn5_2',
                                  parameter_update_device=device))

        conv5_3 = relu(batch_norm(x=conv2d(x=conv5_2, output_filters=512,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv5_3',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn5_3',
                                  parameter_update_device=device))

        pool5 = tf.nn.max_pool(value=conv5_3,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool5')





        # block 6
        fc6 = tf.reshape(pool5, [batch_size, -1])
        fc6 = tf.nn.dropout(x=relu(fc(x=fc6,
                                      output_size=4096,
                                      scope="fc6",
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      parameter_update_device=device,
                                      weight_decay_rate=weight_decay_rate)),
                            keep_prob=keep_prob)

        # block 7
        fc7 = tf.nn.dropout(x=relu(fc(x=fc6,
                                      output_size=4096,
                                      scope="fc7",
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      parameter_update_device=device,
                                      weight_decay_rate=weight_decay_rate)),
                            keep_prob=keep_prob)



        features = list()
        if 1 in output_high_level_features:
            features.append(conv1_2)
        if 2 in output_high_level_features:
            features.append(conv2_2)
        if 3 in output_high_level_features:
            features.append(conv3_3)
        if 4 in output_high_level_features:
            features.append(conv4_3)
        if 5 in output_high_level_features:
            features.append(conv5_3)


        return  features, return_str