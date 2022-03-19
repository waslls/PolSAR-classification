import tensorflow as tf

def focal_loss_softmax(labels, logits, class_num, alpha=None, gamma=2, size_average=True):
    """
    Computer focal loss for segmentation
    Args:
      labels: A int32 tensor of shape [batch_size, H, W].
      logits: A float32 tensor of shape [batch_size, H, W, num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    if alpha is None:
        alpha_ = tf.Variable(tf.ones(class_num, 1))
    else:
        alpha_ = tf.Variable(alpha)
    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, tf.int32)

    N = logits.shape[0]
    C = logits.shape[1]
    P = tf.nn.softmax(logits)
    P = tf.reshape(P, (-1, class_num))

    ids = tf.reshape(labels, [-1, 1])
    class_mask = tf.one_hot(labels, class_num)  # one_hot labels

    alpha_ = tf.gather(alpha_, tf.reshape(ids, [-1]))  # 取出每个类别对应的权重,shape同ids
    probs = tf.math.reduce_sum(tf.math.multiply(P, class_mask), 1)
    probs = tf.clip_by_value(probs, 1e-6, 1-1e-6)#clip以下11111111111111111111111111
    log_p = tf.math.log(probs)
#     tf.double
    batch_loss = -alpha_ * (tf.math.pow((1 - probs), gamma)) * log_p

    if size_average:
        loss = tf.math.reduce_mean(batch_loss)
    else:
        loss = tf.math.reduce_sum(batch_loss)

    return loss
