from __future__ import division
import tensorflow as tf
import numpy as np
import keras

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01

def resize_like(inputs, ref):
    """
    Resizes `inputs` to have the same spatial dimensions as `ref`.
    """
    iH, iW = inputs.shape[1], inputs.shape[2]
    rH, rW = ref.shape[1], ref.shape[2]
    
    # Use static shape comparison
    if iH == rH and iW == rW:
        return inputs
    
    # Use dynamic shape from ref for resizing
    return tf.image.resize(inputs, tf.compat.v1.shape(ref)[1:3], method='nearest')

def build_pose_exp_net(H, W, num_source, do_exp=True):
    """
    Builds the Pose and Explainability network as a keras.Model.
    """
    # --- Define shared layer arguments to replace slim.arg_scope ---
    conv_args = {
        'kernel_regularizer': keras.regularizers.l2(0.05),
        'activation': 'relu',
        'padding': 'same'  # slim.conv2d defaults to 'SAME' padding
    }
    deconv_args = {
        'kernel_regularizer': keras.regularizers.l2(0.05),
        'activation': 'relu',
        'padding': 'same'  # slim.conv2d_transpose defaults to 'SAME'
    }
    # Args for final prediction layers (no activation)
    pred_args = {
        'kernel_regularizer': keras.regularizers.l2(0.05),
        'activation': None,
        'padding': 'same'
    }

    # --- Define Inputs ---
    tgt_image = keras.layers.Input(shape=(H, W, 3), name='tgt_image')
    src_image_stack = keras.layers.Input(shape=(H, W, 3 * num_source), name='src_image_stack')
    
    inputs = keras.layers.Concatenate(axis=3)([tgt_image, src_image_stack])

    # --- Shared Encoder ---
    cnv1 = keras.layers.Conv2D(16, (7, 7), strides=2, name='cnv1', **conv_args)(inputs)
    cnv2 = keras.layers.Conv2D(32, (5, 5), strides=2, name='cnv2', **conv_args)(cnv1)
    cnv3 = keras.layers.Conv2D(64, (3, 3), strides=2, name='cnv3', **conv_args)(cnv2)
    cnv4 = keras.layers.Conv2D(128, (3, 3), strides=2, name='cnv4', **conv_args)(cnv3)
    cnv5 = keras.layers.Conv2D(256, (3, 3), strides=2, name='cnv5', **conv_args)(cnv4)

    # --- Pose Specific Layers ---
    with tf.name_scope('pose'):
        cnv6 = keras.layers.Conv2D(256, (3, 3), strides=2, name='cnv6', **conv_args)(cnv5)
        cnv7 = keras.layers.Conv2D(256, (3, 3), strides=2, name='cnv7', **conv_args)(cnv6)
        
        pose_pred = keras.layers.Conv2D(6 * num_source, (1, 1), strides=1, name='pose_pred', **pred_args)(cnv7)
        pose_avg = keras.layers.GlobalAveragePooling2D(name='pose_avg')(pose_pred)
        pose_final = keras.layers.Lambda(
            lambda x: 0.01 * x, name='pose_scale'
        )(pose_avg)
        pose_final = keras.layers.Reshape((num_source, 6), name='pose_final')(pose_final)

    # --- Dictionary for endpoints (replaces collections) ---
    endpoints = {
        'cnv1': cnv1, 'cnv2': cnv2, 'cnv3': cnv3, 'cnv4': cnv4, 'cnv5': cnv5,
        'cnv6': cnv6, 'cnv7': cnv7, 'pose_pred': pose_pred, 'pose_avg': pose_avg
    }
    
    masks_out = [None, None, None, None]

    # --- Explainability Mask Specific Layers ---
    if do_exp:
        with tf.name_scope('exp'):
            upcnv5 = keras.layers.Conv2DTranspose(256, (3, 3), strides=2, name='upcnv5', **deconv_args)(cnv5)

            upcnv4 = keras.layers.Conv2DTranspose(128, (3, 3), strides=2, name='upcnv4', **deconv_args)(upcnv5)
            mask4 = keras.layers.Conv2D(num_source * 2, (3, 3), strides=1, name='mask4', **pred_args)(upcnv4)

            upcnv3 = keras.layers.Conv2DTranspose(64, (3, 3), strides=2, name='upcnv3', **deconv_args)(upcnv4)
            mask3 = keras.layers.Conv2D(num_source * 2, (3, 3), strides=1, name='mask3', **pred_args)(upcnv3)
            
            upcnv2 = keras.layers.Conv2DTranspose(32, (5, 5), strides=2, name='upcnv2', **deconv_args)(upcnv3)
            mask2 = keras.layers.Conv2D(num_source * 2, (5, 5), strides=1, name='mask2', **pred_args)(upcnv2)

            upcnv1 = keras.layers.Conv2DTranspose(16, (7, 7), strides=2, name='upcnv1', **deconv_args)(upcnv2)
            mask1 = keras.layers.Conv2D(num_source * 2, (7, 7), strides=1, name='mask1', **pred_args)(upcnv1)
            
            masks_out = [mask1, mask2, mask3, mask4]
            endpoints.update({
                'upcnv5': upcnv5, 'upcnv4': upcnv4, 'mask4': mask4,
                'upcnv3': upcnv3, 'mask3': mask3, 'upcnv2': upcnv2, 'mask2': mask2,
                'upcnv1': upcnv1, 'mask1': mask1
            })

    # --- Build the Model ---
    model_outputs = {
        'pose': pose_final,
        'masks': masks_out,
        'endpoints': endpoints
    }
    model = keras.Model(
        inputs={'tgt_image': tgt_image, 'src_image_stack': src_image_stack},
        outputs=model_outputs,
        name='pose_exp_net'
    )
    
    return model

def build_disp_net(H, W):
    """
    Builds the Disparity (Depth) network as a keras.Model.
    """
    # --- Define shared layer arguments ---
    conv_args = {
        'kernel_regularizer': keras.regularizers.l2(0.05),
        'activation': 'relu',
        'padding': 'same'
    }
    deconv_args = {
        'kernel_regularizer': keras.regularizers.l2(0.05),
        'activation': 'relu',
        'padding': 'same'
    }
    # Args for disparity (disp) layers
    disp_args = {
        'kernel_regularizer': keras.regularizers.l2(0.05),
        'activation': 'sigmoid',  # Original used tf.sigmoid
        'padding': 'same'
    }
    
    # --- Define Input ---
    tgt_image = keras.layers.Input(shape=(H, W, 3), name='tgt_image')
    
    # --- Scaling helper function ---
    def scale_and_shift(x):
        return x * DISP_SCALING + MIN_DISP

    # --- Encoder ---
    cnv1 = keras.layers.Conv2D(32, (7, 7), strides=2, name='cnv1', **conv_args)(tgt_image)
    cnv1b = keras.layers.Conv2D(32, (7, 7), strides=1, name='cnv1b', **conv_args)(cnv1)
    cnv2 = keras.layers.Conv2D(64, (5, 5), strides=2, name='cnv2', **conv_args)(cnv1b)
    cnv2b = keras.layers.Conv2D(64, (5, 5), strides=1, name='cnv2b', **conv_args)(cnv2)
    cnv3 = keras.layers.Conv2D(128, (3, 3), strides=2, name='cnv3', **conv_args)(cnv2b)
    cnv3b = keras.layers.Conv2D(128, (3, 3), strides=1, name='cnv3b', **conv_args)(cnv3)
    cnv4 = keras.layers.Conv2D(256, (3, 3), strides=2, name='cnv4', **conv_args)(cnv3b)
    cnv4b = keras.layers.Conv2D(256, (3, 3), strides=1, name='cnv4b', **conv_args)(cnv4)
    cnv5 = keras.layers.Conv2D(512, (3, 3), strides=2, name='cnv5', **conv_args)(cnv4b)
    cnv5b = keras.layers.Conv2D(512, (3, 3), strides=1, name='cnv5b', **conv_args)(cnv5)
    cnv6 = keras.layers.Conv2D(512, (3, 3), strides=2, name='cnv6', **conv_args)(cnv5b)
    cnv6b = keras.layers.Conv2D(512, (3, 3), strides=1, name='cnv6b', **conv_args)(cnv6)
    cnv7 = keras.layers.Conv2D(512, (3, 3), strides=2, name='cnv7', **conv_args)(cnv6b)
    cnv7b = keras.layers.Conv2D(512, (3, 3), strides=1, name='cnv7b', **conv_args)(cnv7)

    # --- Decoder ---
    upcnv7 = keras.layers.Conv2DTranspose(512, (3, 3), strides=2, name='upcnv7', **deconv_args)(cnv7b)
    upcnv7 = resize_like(upcnv7, cnv6b)
    i7_in = keras.layers.Concatenate(axis=3)([upcnv7, cnv6b])
    icnv7 = keras.layers.Conv2D(512, (3, 3), strides=1, name='icnv7', **conv_args)(i7_in)

    upcnv6 = keras.layers.Conv2DTranspose(512, (3, 3), strides=2, name='upcnv6', **deconv_args)(icnv7)
    upcnv6 = resize_like(upcnv6, cnv5b)
    i6_in = keras.layers.Concatenate(axis=3)([upcnv6, cnv5b])
    icnv6 = keras.layers.Conv2D(512, (3, 3), strides=1, name='icnv6', **conv_args)(i6_in)

    upcnv5 = keras.layers.Conv2DTranspose(256, (3, 3), strides=2, name='upcnv5', **deconv_args)(icnv6)
    upcnv5 = resize_like(upcnv5, cnv4b)
    i5_in = keras.layers.Concatenate(axis=3)([upcnv5, cnv4b])
    icnv5 = keras.layers.Conv2D(256, (3, 3), strides=1, name='icnv5', **conv_args)(i5_in)

    upcnv4 = keras.layers.Conv2DTranspose(128, (3, 3), strides=2, name='upcnv4', **deconv_args)(icnv5)
    i4_in = keras.layers.Concatenate(axis=3)([upcnv4, cnv3b])
    icnv4 = keras.layers.Conv2D(128, (3, 3), strides=1, name='icnv4', **conv_args)(i4_in)
    disp4_raw = keras.layers.Conv2D(1, (3, 3), strides=1, name='disp4_raw', **disp_args)(icnv4)
    disp4 = keras.layers.Lambda(scale_and_shift, name='disp4')(disp4_raw)
    
    # Using Keras Resizing layer instead of tf.image.resize_bilinear
    disp4_up = keras.layers.Resizing(H // 4, W // 4, interpolation='bilinear', name='disp4_up')(disp4)

    upcnv3 = keras.layers.Conv2DTranspose(64, (3, 3), strides=2, name='upcnv3', **deconv_args)(icnv4)
    i3_in = keras.layers.Concatenate(axis=3)([upcnv3, cnv2b, disp4_up])
    icnv3 = keras.layers.Conv2D(64, (3, 3), strides=1, name='icnv3', **conv_args)(i3_in)
    disp3_raw = keras.layers.Conv2D(1, (3, 3), strides=1, name='disp3_raw', **disp_args)(icnv3)
    disp3 = keras.layers.Lambda(scale_and_shift, name='disp3')(disp3_raw)

    disp3_up = keras.layers.Resizing(H // 2, W // 2, interpolation='bilinear', name='disp3_up')(disp3)

    upcnv2 = keras.layers.Conv2DTranspose(32, (3, 3), strides=2, name='upcnv2', **deconv_args)(icnv3)
    i2_in = keras.layers.Concatenate(axis=3)([upcnv2, cnv1b, disp3_up])
    icnv2 = keras.layers.Conv2D(32, (3, 3), strides=1, name='icnv2', **conv_args)(i2_in)
    disp2_raw = keras.layers.Conv2D(1, (3, 3), strides=1, name='disp2_raw', **disp_args)(icnv2)
    disp2 = keras.layers.Lambda(scale_and_shift, name='disp2')(disp2_raw)

    disp2_up = keras.layers.Resizing(H, W, interpolation='bilinear', name='disp2_up')(disp2)

    upcnv1 = keras.layers.Conv2DTranspose(16, (3, 3), strides=2, name='upcnv1', **deconv_args)(icnv2)
    i1_in = keras.layers.Concatenate(axis=3)([upcnv1, disp2_up])
    icnv1 = keras.layers.Conv2D(16, (3, 3), strides=1, name='icnv1', **conv_args)(i1_in)
    disp1_raw = keras.layers.Conv2D(1, (3, 3), strides=1, name='disp1_raw', **disp_args)(icnv1)
    disp1 = keras.layers.Lambda(scale_and_shift, name='disp1')(disp1_raw)
    
    # --- Collect endpoints ---
    endpoints = {
        'cnv1': cnv1, 'cnv1b': cnv1b, 'cnv2': cnv2, 'cnv2b': cnv2b, 'cnv3': cnv3, 'cnv3b': cnv3b,
        'cnv4': cnv4, 'cnv4b': cnv4b, 'cnv5': cnv5, 'cnv5b': cnv5b, 'cnv6': cnv6, 'cnv6b': cnv6b,
        'cnv7': cnv7, 'cnv7b': cnv7b, 'upcnv7': upcnv7, 'icnv7': icnv7, 'upcnv6': upcnv6,
        'icnv6': icnv6, 'upcnv5': upcnv5, 'icnv5': icnv5, 'upcnv4': upcnv4, 'icnv4': icnv4,
        'disp4': disp4, 'upcnv3': upcnv3, 'icnv3': icnv3, 'disp3': disp3, 'upcnv2': upcnv2,
        'icnv2': icnv2, 'disp2': disp2, 'upcnv1': upcnv1, 'icnv1': icnv1, 'disp1': disp1
    }
    
    # --- Build the Model ---
    disparities = [disp1, disp2, disp3, disp4]
    model_outputs = {
        'disparities': disparities,
        'endpoints': endpoints
    }
    
    model = keras.Model(
        inputs={'tgt_image': tgt_image},
        outputs=model_outputs,
        name='disp_net'
    )
    
    return model

# --- Example Usage ---
if __name__ == '__main__':
    # Example dimensions
    H, W = 128, 416
    NUM_SOURCE = 2
    BATCH_SIZE = 4

    # --- Test Pose Exp Net ---
    print("--- Building Pose Exp Net ---")
    pose_net = build_pose_exp_net(H, W, NUM_SOURCE, do_exp=True)
    # pose_net.summary()

    # Create dummy inputs
    dummy_tgt = tf.random.normal([BATCH_SIZE, H, W, 3])
    dummy_src_stack = tf.random.normal([BATCH_SIZE, H, W, 3 * NUM_SOURCE])
    
    pose_inputs = {'tgt_image': dummy_tgt, 'src_image_stack': dummy_src_stack}
    pose_outputs = pose_net(pose_inputs)
    
    print(f"Pose output shape: {pose_outputs['pose'].shape}")
    print(f"Mask 1 output shape: {pose_outputs['masks'][0].shape}")

    # --- Test Disp Net ---
    print("\n--- Building Disp Net ---")
    disp_net = build_disp_net(H, W)
    # disp_net.summary()
    
    disp_inputs = {'tgt_image': dummy_tgt}
    disp_outputs = disp_net(disp_inputs)
    
    print(f"Disp 1 output shape: {disp_outputs['disparities'][0].shape}")
    print(f"Disp 2 output shape: {disp_outputs['disparities'][1].shape}")
    print(f"Disp 3 output shape: {disp_outputs['disparities'][2].shape}")
    print(f"Disp 4 output shape: {disp_outputs['disparities'][3].shape}")