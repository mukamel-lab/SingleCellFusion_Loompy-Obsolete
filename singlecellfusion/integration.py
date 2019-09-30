"""
Functions used to prepare for and impute data across sequencing modalities

The idea of using MNNs and a Gaussian kernel to impute across modalities is
based on ideas from the Marioni, Krishnaswamy, and Pe'er groups. The relevant
citations are:

'Batch effects in single-cell RNA sequencing data are corrected by matching
mutual nearest neighbors' by Laleh Haghverdi, Aaron TL Lun, Michael D Morgan,
and John C Marioni. Published in Nature Biotechnology. DOI:
https://doi.org/10.1038/nbt.4091.

'MAGIC: A diffusion-based imputation method reveals gene-gene interactions in
single-cell RNA-sequencing data.' The publication was authored by: David van
Dijk, Juozas Nainys, Roshan Sharma, Pooja Kathail, Ambrose J Carr, Kevin R Moon,
Linas Mazutis, Guy Wolf, Smita Krishnaswamy, Dana Pe'er. Published in Cell.
DOI: https://doi.org/10.1101/111591.

Below code was written/developed by Fangming Xie, Ethan Armand, and Wayne Doyle

(C) 2019 Mukamel Lab GPLv2
"""

import logging

# Start log
int_log = logging.getLogger(__name__)
