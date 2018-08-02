import tensorflow as tf
import sys
sys.path.append('..')


from utilities.ops import lrelu, relu,  batch_norm


from utilities.ops import conv2d
from utilities.ops import fc

print_separater="#########################################################"

FACE_AVG_IMG_CHANNEL_0 = 129.1862793
FACE_AVG_IMG_CHANNEL_1 = 104.76238251
FACE_AVG_IMG_CHANNEL_2 = 93.59396362

weight_decay_rate  = 0.004



def encoder_8_layers(image,
                     batch_size,
                     device,
                     logits_length_font,
                     logits_length_character,
                     is_training,
                     weight_decay=True,
                     name_prefix='None',
                     initializer='XavierInit',
                     reuse=False):


    def encode_layer(x, output_filters, layer, bn):
        act = lrelu(x)
        conv = conv2d(x=act, output_filters=output_filters,
                      scope="Enc8Layers_conv_%s" % layer,
                      parameter_update_device=device,
                      weight_decay=weight_decay,
                      initializer=initializer,
                      name_prefix=name_prefix,
                      weight_decay_rate=weight_decay_rate)
        if bn:
            enc = batch_norm(conv, is_training, scope="Enc8Layers_bn_%s" % layer,
                             parameter_update_device=device)
        else:
            enc = conv
        return enc

    if is_training:
        print(print_separater)
        print("Training on Encoder_8_Layers")
        print(print_separater)

    generator_dim=64

    with tf.variable_scope('ext_encoder8layers'):
        with tf.device(device):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            e1 = conv2d(x=image,
                        output_filters=generator_dim,
                        scope="Enc8Layers_conv_1",
                        parameter_update_device=device,
                        weight_decay=weight_decay,
                        initializer=initializer,
                        name_prefix=name_prefix,
                        weight_decay_rate=weight_decay_rate)
            e2 = encode_layer(e1, generator_dim * 2, '2', True)
            e3 = encode_layer(e2, generator_dim * 4, '3', True)
            e4 = encode_layer(e3, generator_dim * 8, '4', True)
            e5 = encode_layer(e4, generator_dim * 8, '5', True)
            e6 = encode_layer(e5, generator_dim * 8, '6', True)
            e7 = encode_layer(e6, generator_dim * 8, '7', True)

            if not logits_length_font == 1:
                e8_label1 = encode_layer(e7, logits_length_font, 'output_label1', False)
                e8_label1 = tf.squeeze(e8_label1)
            else:
                e8_label1 = tf.constant(value=-1,
                                        dtype=tf.float32,
                                        shape=[batch_size, 1])
            e8_label0 = encode_layer(e7, logits_length_character, 'output_label0', False)
            e8_label0 = tf.squeeze(e8_label0)

        return e8_label1,e8_label0

def encoder_6_layers(image,
                     batch_size,
                     device,
                     logits_length_font,
                     logits_length_character,
                     is_training,
                     weight_decay=True,
                     name_prefix='None',
                     initializer='XavierInit',
                     reuse=False):


    def encode_layer(x, output_filters, layer, bn):
        act = lrelu(x)
        conv = conv2d(x=act, output_filters=output_filters,
                      scope="Enc8Layers_conv_%s" % layer,
                      parameter_update_device=device,
                      weight_decay=weight_decay,
                      initializer=initializer,
                      name_prefix=name_prefix,
                      weight_decay_rate=weight_decay_rate)
        if bn:
            enc = batch_norm(conv, is_training, scope="Enc8Layers_bn_%s" % layer,
                             parameter_update_device=device)
        else:
            enc = conv
        return enc


    if is_training:
        print(print_separater)
        print("Training on Encoder_6_Layers")
        print(print_separater)

    generator_dim=64
    with tf.variable_scope('ext_encoder8layers'):
        with tf.device(device):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            e1 = conv2d(x=image,
                        output_filters=generator_dim,
                        scope="Enc8Layers_conv_1",
                        parameter_update_device=device,
                        weight_decay=weight_decay,
                        initializer=initializer,
                        name_prefix=name_prefix,
                        weight_decay_rate=weight_decay_rate)
            e2 = encode_layer(e1, generator_dim * 2, '2', True)
            e3 = encode_layer(e2, generator_dim * 4, '3', True)
            e4 = encode_layer(e3, generator_dim * 8, '4', True)
            e5 = encode_layer(e4, generator_dim * 8, '5', True)

            if not logits_length_font == 1:
                e6_label1 = encode_layer(e5, logits_length_font, 'output_label1', False)
                e6_label1 = tf.squeeze(e6_label1)
            else:
                e6_label1 = tf.constant(value=-1,
                                        dtype=tf.float32,
                                        shape=[batch_size, 1])
            e6_label0 = encode_layer(e5, logits_length_character, 'output_label0', False)
            e6_label0 = tf.squeeze(e6_label0)

        return e6_label1,e6_label0





def alexnet(image,
            batch_size,
            device,
            logits_length_font,
            logits_length_character,
            is_training,
            reuse=False,
            weight_decay=True,
            name_prefix='None',
            initializer='XavierInit'):

    if is_training:
        print(print_separater)
        print("Training on AlexNet")
        print(print_separater)

    if is_training:
        keep_prob = 0.5
    else:
        keep_prob = 1.0


    with tf.variable_scope('ext_alexnet'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        conv1 = relu(batch_norm(x=conv2d(x=image, output_filters=96,
                                         sh=1, sw=1,
                                         kh=11, kw=11,
                                         padding='VALID',
                                         parameter_update_device=device,
                                         weight_decay=weight_decay,
                                         initializer=initializer,
                                         name_prefix=name_prefix,
                                         scope='conv1',
                                         weight_decay_rate=weight_decay_rate),
                                is_training=is_training,
                                scope='bn1',
                                parameter_update_device=device))
        pool1 = tf.nn.max_pool(value=conv1,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool1')



        conv2 = relu(batch_norm(x=conv2d(x=pool1, output_filters=256,
                                         sh=1, sw=1,
                                         kh=5, kw=5,
                                         padding='SAME',
                                         parameter_update_device=device,
                                         weight_decay=weight_decay,
                                         initializer=initializer,
                                         name_prefix=name_prefix,
                                         scope='conv2',
                                         weight_decay_rate=weight_decay_rate),
                                is_training=is_training,
                                scope='bn2',
                                parameter_update_device=device))
        pool2 = tf.nn.max_pool(value=conv2,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool2')



        conv3 = relu(conv2d(x=pool2, output_filters=384,
                            sh=1, sw=1,
                            kh=3, kw=3,
                            padding='SAME',
                            parameter_update_device=device,
                            weight_decay=weight_decay,
                            initializer=initializer,
                            name_prefix=name_prefix,
                            scope='conv3',
                            weight_decay_rate=weight_decay_rate))

        conv4 = relu(conv2d(x=conv3, output_filters=384,
                            sh=1, sw=1,
                            kh=3, kw=3,
                            padding='SAME',
                            parameter_update_device=device,
                            weight_decay=weight_decay,
                            initializer=initializer,
                            name_prefix=name_prefix,
                            scope='conv4',
                            weight_decay_rate=weight_decay_rate))

        conv5 = relu(conv2d(x=conv4, output_filters=256,
                            sh=1, sw=1,
                            kh=3, kw=3,
                            padding='SAME',
                            parameter_update_device=device,
                            weight_decay=weight_decay,
                            initializer=initializer,
                            name_prefix=name_prefix,
                            scope='conv5',
                            weight_decay_rate=weight_decay_rate))
        pool5 = tf.nn.max_pool(value=conv5,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool5')





        fc6 = tf.nn.dropout(x=relu(fc(x=tf.reshape(pool5, [batch_size, -1]),
                                      output_size=4096,
                                      scope="fc6_",
                                      parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                      weight_decay_rate=weight_decay_rate)),
                            keep_prob=keep_prob)

        fc7 = tf.nn.dropout(x=relu(fc(x=fc6,
                                      output_size=4096,
                                      scope="fc7_",
                                      parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                      weight_decay_rate=weight_decay_rate)),
                            keep_prob=keep_prob)

        if not logits_length_font == 1:
            output_label1 = fc(x=fc7,
                               output_size=logits_length_font,
                               scope="output_label1",
                               parameter_update_device=device,
                               weight_decay=weight_decay,
                               initializer=initializer,
                               name_prefix=name_prefix,
                               weight_decay_rate=weight_decay_rate)
        else:
            output_label1 = tf.constant(value=-1,
                                        dtype=tf.float32,
                                        shape=[batch_size,1])

        output_label0 = fc(x=fc7,
                           output_size=logits_length_character,
                           scope="output_label0",
                           parameter_update_device=device,
                           weight_decay=weight_decay,
                           initializer=initializer,
                           name_prefix=name_prefix,
                           weight_decay_rate=weight_decay_rate)

        output_logit_list=list()
        output_logit_list.append(tf.nn.softmax(output_label1))
        output_logit_list.append(tf.nn.softmax(fc7))
        output_logit_list.append(tf.nn.softmax(fc6))
        output_logit_list.append(tf.nn.softmax(tf.reshape(pool5, [batch_size, -1])))

        return output_label1,output_label0




def vgg_11_net(image,
               batch_size,
               device,
               logits_length_font,
               logits_length_character,
               is_training,
               weight_decay=True,
               name_prefix='None',
               initializer='XavierInit',
               reuse=False):

    if is_training:
        print(print_separater)
        print("Training on Vgg-11")
        print(print_separater)

    if is_training:
        keep_prob=0.5
    else:
        keep_prob=1.0


    with tf.variable_scope('ext_vgg11net'):
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
                                           name_prefix=name_prefix,
                                           scope='conv1_1',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn1_1',
                                  parameter_update_device=device))
        pool1 = tf.nn.max_pool(value=conv1_1,
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
                                           name_prefix=name_prefix,
                                           scope='conv2_1',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn2_1',
                                  parameter_update_device=device))


        pool2 = tf.nn.max_pool(value=conv2_1,
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
                                           name_prefix=name_prefix,
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
                                           name_prefix=name_prefix,
                                           scope='conv3_2',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn3_2',
                                  parameter_update_device=device))

        pool3 = tf.nn.max_pool(value=conv3_2,
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
                                           name_prefix=name_prefix,
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
                                           name_prefix=name_prefix,
                                           scope='conv4_2',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn4_2',
                                  parameter_update_device=device))

        pool4 = tf.nn.max_pool(value=conv4_2,
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
                                           name_prefix=name_prefix,
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
                                           name_prefix=name_prefix,
                                           scope='conv5_2',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn5_2',
                                  parameter_update_device=device))

        pool5 = tf.nn.max_pool(value=conv5_2,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool5')




        # block 6
        fc6 = tf.reshape(pool5, [batch_size, -1])
        fc6 = tf.nn.dropout(x=relu(fc(x=fc6,
                                      output_size=4096,
                                      scope="fc6",
                                      parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                      weight_decay_rate=weight_decay_rate)),
                            keep_prob=keep_prob)

        # block 7
        fc7 = tf.nn.dropout(x=relu(fc(x=fc6,
                                      output_size=4096,
                                      scope="fc7",
                                      parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                      weight_decay_rate=weight_decay_rate)),
                            keep_prob=keep_prob)

        # block 8
        if not logits_length_font == 1:

            output_label1 = fc(x=fc7,
                               output_size=logits_length_font,
                               scope="output_label1",
                               parameter_update_device=device,
                               weight_decay=weight_decay,
                               initializer=initializer,
                               name_prefix=name_prefix,
                               weight_decay_rate=weight_decay_rate)
        else:
            output_label1 = tf.constant(value=-1,
                                        dtype=tf.float32,
                                        shape=[batch_size,1])

        output_label0 = fc(x=fc7,
                           output_size=logits_length_character,
                           scope="output_label0",
                           parameter_update_device=device,
                           weight_decay=weight_decay,
                           initializer=initializer,
                           name_prefix=name_prefix,
                           weight_decay_rate=weight_decay_rate)

        return output_label1, output_label0






def vgg_16_net(image,
               batch_size,
               device,
               logits_length_font,
               logits_length_character,
               is_training,
               weight_decay=True,
               name_prefix='None',
               initializer='XavierInit',
               reuse=False):

    if is_training:
        print(print_separater)
        print("Training on Vgg-16")
        print(print_separater)
    if is_training:
        keep_prob=0.5
    else:
        keep_prob=1.0


    with tf.variable_scope('ext_vgg16net'):
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
                                      name_prefix=name_prefix,
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
                                      name_prefix=name_prefix,
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
                                      name_prefix=name_prefix,
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
                                      name_prefix=name_prefix,
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
                                      name_prefix=name_prefix,
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
                                      name_prefix=name_prefix,
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
                                      name_prefix=name_prefix,
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
                                      name_prefix=name_prefix,
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
                                      name_prefix=name_prefix,
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
                                      name_prefix=name_prefix,
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
                                      name_prefix=name_prefix,
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
                                      name_prefix=name_prefix,
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
                                      name_prefix=name_prefix,
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
                                      parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                      weight_decay_rate=weight_decay_rate)),
                            keep_prob=keep_prob)

        # block 7
        fc7 = tf.nn.dropout(x=relu(fc(x=fc6,
                                      output_size=4096,
                                      scope="fc7",
                                      parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                      weight_decay_rate=weight_decay_rate)),
                            keep_prob=keep_prob)

        # block 8
        if not logits_length_font == 1:

            output_label1 = fc(x=fc7,
                               output_size=logits_length_font,
                               scope="output_label1",
                               parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                               weight_decay_rate=weight_decay_rate)
        else:
            output_label1 = tf.constant(value=-1,
                                        dtype=tf.float32,
                                        shape=[batch_size,1])

        output_label0 = fc(x=fc7,
                              output_size=logits_length_character,
                              scope="output_label0",
                              parameter_update_device=device,
                           weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                           weight_decay_rate=weight_decay_rate)
        return output_label1, output_label0



# for point04 only
def vgg_16_net_no_bn(image,
                     batch_size,
                     device,
                     logits_length_font,
                     logits_length_character,
                     is_training,
                     weight_decay=True,
                     name_prefix='None',
                     initializer='XavierInit',
                     reuse=False,
                     ):
    image_input = tf.image.resize_images(images=image, size=[224, 224])
    image_input = tf.multiply(tf.divide(tf.add(image_input,
                                               tf.constant(1, tf.float32)),
                                        tf.constant(2, tf.float32)),
                              tf.constant(255, tf.float32))

    image_input = tf.subtract(tf.cast(image_input, tf.float32),
                              tf.constant([FACE_AVG_IMG_CHANNEL_0,
                                           FACE_AVG_IMG_CHANNEL_1,
                                           FACE_AVG_IMG_CHANNEL_2],
                                          tf.float32))
    if is_training:
        print(print_separater)
        print("Training on Vgg-16 with no BatchNorm")
        print(print_separater)
    if is_training:
        keep_prob=0.5
    else:
        keep_prob=1.0


    with tf.variable_scope('ext_vgg16net_nobn'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        ## block 1
        conv1_1 = relu(x=conv2d(x=image_input, output_filters=64,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv1_1',
                                weight_decay_rate=weight_decay_rate))

        conv1_2 = relu(x=conv2d(x=conv1_1, output_filters=64,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv1_2',
                                weight_decay_rate=weight_decay_rate))

        pool1 = tf.nn.max_pool(value=conv1_2,
                                   ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool1')

        ## block 2
        conv2_1 = relu(x=conv2d(x=pool1, output_filters=128,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv2_1',
                                weight_decay_rate=weight_decay_rate))

        conv2_2 = relu(x=conv2d(x=conv2_1, output_filters=128,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv2_2',
                                weight_decay_rate=weight_decay_rate))

        pool2 = tf.nn.max_pool(value=conv2_2,
                                   ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool2')

        ## block 3
        conv3_1 = relu(x=conv2d(x=pool2, output_filters=256,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv3_1',
                                weight_decay_rate=weight_decay_rate))

        conv3_2 = relu(x=conv2d(x=conv3_1, output_filters=256,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv3_2',
                                weight_decay_rate=weight_decay_rate))

        conv3_3 = relu(x=conv2d(x=conv3_2, output_filters=256,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv3_3',
                                weight_decay_rate=weight_decay_rate))

        pool3 = tf.nn.max_pool(value=conv3_3,
                                   ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool3')

        ## block 4
        conv4_1 = relu(x=conv2d(x=pool3, output_filters=512,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv4_1',
                                weight_decay_rate=weight_decay_rate))

        conv4_2 = relu(x=conv2d(x=conv4_1, output_filters=512,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv4_2'))

        conv4_3 = relu(x=conv2d(x=conv4_2, output_filters=512,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv4_3',
                                weight_decay_rate=weight_decay_rate))

        pool4 = tf.nn.max_pool(value=conv4_3,
                                   ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool4')

        ## block 5
        conv5_1 = relu(x=conv2d(x=pool4, output_filters=512,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv5_1',
                                weight_decay_rate=weight_decay_rate))

        conv5_2 = relu(x=conv2d(x=conv5_1, output_filters=512,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv5_2',
                                weight_decay_rate=weight_decay_rate))

        conv5_3 = relu(x=conv2d(x=conv5_2, output_filters=512,
                                    kh=3, kw=3,
                                    sh=1, sw=1,
                                    padding='SAME',
                                    parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                    scope='conv5_3'))

        pool5 = tf.nn.max_pool(value=conv5_3,
                                   ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool5')

        # block 6
        fc6 = tf.reshape(pool5, [batch_size, -1])
        fc6 = tf.nn.dropout(x=relu(fc(x=fc6,
                                          output_size=4096,
                                          scope="fc6",
                                          parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                      weight_decay_rate=weight_decay_rate)),
                                keep_prob=keep_prob)

        # block 7
        fc7 = tf.nn.dropout(x=relu(fc(x=fc6,
                                          output_size=4096,
                                          scope="fc7",
                                          parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                                      weight_decay_rate=weight_decay_rate)),
                                keep_prob=keep_prob)

        # block 8
        output_label1 = fc(x=fc7,
                             output_size=logits_length_font,
                             scope="output_label1",
                             parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                           weight_decay_rate=weight_decay_rate)

        output_label0 = fc(x=fc7,
                                  output_size=logits_length_character,
                                  scope="output_label0",
                                  parameter_update_device=device,
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      name_prefix=name_prefix,
                           weight_decay_rate=weight_decay_rate)

        return output_label1, output_label0
