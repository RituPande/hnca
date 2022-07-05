import tensorflow as tf  

def load_image():  
  img =  tf.image.decode_jpeg(img, channels=3)
  img =  tf.image.convert_image_dtype(img, tf.float32)
  return img