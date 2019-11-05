"""singlecellfusion - Integrates single cell transcriptomic and epigenomic data"""

__version__ = '0.1.0'
__author__ = 'Mukamel Lab <lab@brainome.ucsd.edu>'
__all__ = ['decomposition',
           'features',
           'imputation',
           'integration',
           'recipes',
           'wrappers',
           'utils'
           ]
# Set-up logger
import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO,
                    format=log_fmt,
                    datefmt='%b-%d-%y  %H%M:%S')
logger = logging.getLogger(__name__)

# Import packages
from .features import find_common_variable
from .imputation import perform_imputation
from .integration import integrate_data
from .wrappers import fuse_data
