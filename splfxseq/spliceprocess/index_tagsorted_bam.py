import sys

import os
import os.path
import argparse
from collections import defaultdict, OrderedDict

import numpy as np

import pandas as pd

import pysam
import itertools 

def main():
    opts = argparse.ArgumentParser()

    opts.add_argument('--in_bam', dest='in_bam',
        help='path to bam file containing insert-matching reads, tagged with clone ID', required=True)

    opts.add_argument('--clone_id_tag', default='BC', dest='clone_id_tag' )

    opts.add_argument('--out_bcindex', dest='out_bcindex',
        help='path to output per-bam barcode index; should include placeholder {i} which will be replaced by chunk #',
        required=True )
    
    o = opts.parse_args()
    
    bam = pysam.AlignmentFile( o.in_bam, 'rb' )

    bcindex_cur = OrderedDict( [(k,[]) for k in 
        ['barcode','start_ofs'] ] )

    lastofs=bam.tell()
    lasttag=None

    for l in bam:
        curtag = dict(l.tags)[ o.clone_id_tag ]
        if curtag != lasttag:
            bcindex_cur['barcode'].append(curtag)
            bcindex_cur['start_ofs'].append(lastofs)
            lasttag = curtag

        lastofs=bam.tell()

    pd.DataFrame(bcindex_cur).to_csv( o.out_bcindex, sep='\t', index=False )


if __name__ == '__main__':                
    main()

