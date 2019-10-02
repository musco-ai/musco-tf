import sys
import logging
import warnings
from musco.tf.compressor.compress import CompressorVBMF, compress_seq, compress_noseq
from musco.tf.optimizer.trt import Optimizer

logging.disable(logging.CRITICAL)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
