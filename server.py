import os
import numpy as np
import tensorflow as tf
from runway import RunwayModel
from module import encoder, decoder
from glob import glob

st = RunwayModel()


@st.setup(options={'styleCheckpoint': 'checkpoint'})
def setup(opts):
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    with tf.name_scope('placeholder'):
        input_photo = tf.placeholder(dtype=tf.float32,
                                     shape=[1, None, None, 3],
                                     name='photo')
    input_photo_features = encoder(image=input_photo,
                                   options={'gf_dim': 32},
                                   reuse=False)
    output_photo = decoder(features=input_photo_features,
                           options={'gf_dim': 32},
                           reuse=False)
    saver = tf.train.Saver()
    path = opts['styleCheckpoint']
    model_name = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))][0]
    checkpoint_dir = os.path.join(path, model_name, 'checkpoint_long')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    return dict(sess=sess, input_photo=input_photo, output_photo=output_photo)


@st.command('stylize', inputs={'contentImage': 'image'}, outputs={'stylizedImage': 'image'})
def stylize(model, inp):
    img = inp['contentImage']
    img = np.array(img)
    img = img / 127.5 - 1.
    img = np.expand_dims(img, axis=0)
    img = model['sess'].run(model['output_photo'], feed_dict={model['input_photo']: img})
    img = (img + 1.) * 127.5
    img = img.astype('uint8')
    img = img[0]
    return dict(stylizedImage=img)


if __name__ == '__main__':
    st.run()
