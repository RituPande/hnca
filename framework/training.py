from tqdm import tqdm
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class Train:
  @staticmethod
  def loop( hca_model,func,lr=1e-3, num_epochs= 10000,\
                                   use_pool=True, batch_size=4,\
                                     es_patience_cfg=1000, lr_patience_cfg=750,\
                                      num_batches_per_epoch=8  ):
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-08)
    history = []
    es_patience = es_patience_cfg
    lr_patience = lr_patience_cfg
    min_loss = np.inf
  
    for e in tqdm(tf.range(num_epochs)):
      batch_loss = 0
      for _ in tf.range(num_batches_per_epoch):
        loss, tape, inner_model = func(e,optimizer,use_pool, batch_size)
        variables = inner_model.trainable_variables
        grads = tape.gradient(loss, variables)
        grads = [g/(tf.norm(g)+1e-8) for g in grads]
        optimizer.apply_gradients(zip(grads, variables))
        batch_loss += loss
        
      batch_loss /= num_batches_per_epoch 
      history.append(batch_loss.numpy())

      if batch_loss + 1e-6 < min_loss:
        min_loss = batch_loss
        print("min_loss:",min_loss )
        early_stopping_patience = es_patience_cfg
        lr_patience = lr_patience_cfg
        best_model_weights = hca_model.get_weights()
      else:
        early_stopping_patience -= 1
        lr_patience -= 1
        print("early_stopping_patience:",early_stopping_patience," lr_patience:",lr_patience )
        if early_stopping_patience == 0:
          hca_model.set_weights(best_model_weights)
          break
          
      if lr_patience == 0:
        K.set_value(optimizer.lr, optimizer.lr * 0.1)
        print("New learing rate:",optimizer.lr)
        lr_patience = lr_patience_cfg
        hca_model.set_weights(best_model_weights)
                      
    return history
