import numpy as np
from triplet_loss import triplet_loss_adapted_from_tf_multidimlabels
m = [[0., 1., 1., 0.7638653, 0.87041914],
     [1., 0., 0.2978808, 0.49209085, 0.96417487],
     [0., 1., 0.9862048, 0.23977673, 0.37910876]]
m = np.array(m).astype('float32')
import triplet_loss

triplet_loss.N_LABELS = 3

print(triplet_loss_adapted_from_tf_multidimlabels(None, m))
