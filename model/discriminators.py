
import tensorflow as tf
import sys
sys.path.append('..')


from utilities.ops import lrelu,  layer_norm
from utilities.ops import conv2d, fc

print_separater="#########################################################"
eps = 1e-9
discriminator_dim = 32

def discriminator_mdy_6_convs(image,
                              parameter_update_device,
                              category_logit_num,
                              batch_size,
                              critic_length,
                              initializer,weight_decay,scope,weight_decay_rate,
                              reuse=False):
    return_str = ("Discriminator-6Convs")
    return_str = "WST-" + return_str + "-Crc:%d" % critic_length


    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h0 = lrelu(conv2d(x=image, output_filters=discriminator_dim,
                          scope="dis_h0_conv",
                          parameter_update_device=parameter_update_device,
                          initializer=initializer,
                          weight_decay=weight_decay,
                          name_prefix=scope,
                          weight_decay_rate=weight_decay_rate))
        h1 = lrelu(layer_norm(conv2d(x=h0,
                                     output_filters=discriminator_dim * 2,
                                     scope="dis_h1_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln1",
                              parameter_update_device=parameter_update_device))
        h2 = lrelu(layer_norm(conv2d(x=h1,
                                     output_filters=discriminator_dim * 4,
                                     scope="dis_h2_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln2",
                              parameter_update_device=parameter_update_device))
        h3 = lrelu(layer_norm(conv2d(x=h2,
                                     output_filters=discriminator_dim * 8,
                                     scope="dis_h3_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln3",
                              parameter_update_device=parameter_update_device))

        h4 = lrelu(layer_norm(conv2d(x=h3,
                                     output_filters=discriminator_dim * 16,
                                     scope="dis_h4_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln4",
                              parameter_update_device=parameter_update_device))

        h5 = lrelu(layer_norm(conv2d(x=h4,
                                     output_filters=discriminator_dim * 32,
                                     scope="dis_h5_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln5",
                              parameter_update_device=parameter_update_device))


        h5_reshaped = tf.reshape(h5, [batch_size, -1])
        fc_input = h5_reshaped



        # category loss
        fc2 = fc(x=fc_input,
                 output_size=category_logit_num,
                 scope="dis_final_fc_category",
                 parameter_update_device=parameter_update_device,
                 initializer=initializer,
                 weight_decay=weight_decay,
                 name_prefix=scope,weight_decay_rate=weight_decay_rate)

        fc1 = fc(x=fc_input,
                 output_size=critic_length,
                 scope="dis_final_fc_critic",
                 parameter_update_device=parameter_update_device,
                 initializer=initializer,
                 weight_decay=weight_decay,
                 name_prefix=scope, weight_decay_rate=weight_decay_rate)

        return fc2, fc1, return_str
