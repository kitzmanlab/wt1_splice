import pandas as pd
import numpy as np
import subprocess as subp
import splanl.gnomad as gn

def download_phylop( chrom,
                     start,
                     end,
                     outdir ):

    if is_int( chrom ):
        chrom = 'chr' + str( chrom )

    assert end > start, 'End coordinates must be larger than then start coordinates'

    out = subp.run( '/home/smithcat/bigWigToBedGraph  \
                                   -chrom=%s \
                                   -start=%i \
                                   -end=%i \
                                   http://hgdownload.soe.ucsc.edu/goldenPath/hg19/phyloP100way/hg19.100way.phyloP100way.bw \
                                  %s%s_%s_%s.phyloP100way.bed' % ( chrom, start, end, outdir, chrom, str( start ), str( end ) ),
                    stdout = subp.PIPE,
                    stderr = subp.PIPE,
                    shell = True,
                )

def is_int( str ):
  try:
    int( str )
    return True
  except ValueError:
    return False

def read_phylop( phylop_file ):

    outtbl = pd.read_table( phylop_file,
                        names = [ 'chrom', 'start', 'end', 'phylop' ] )

    outtbl_nomiss = enforce_adj_starts( outtbl )

    return( outtbl_nomiss )


def enforce_adj_starts( bedgraph_tbl ):

    #if you rewrite this function to take in a 'value' column name instead of phylop
    #it would convert any bedgraph into a bed

    bg = bedgraph_tbl.sort_values( by = 'start' ).copy()

    begin = bg.iloc[ 0 ].start
    fin = bg.iloc[ bg.shape[0] - 1 ].end

    outtbl = {
               'chrom': [],
               'gdna_pos': [],
               'phylop': []
             }


    for pos in range( begin, fin ):

        #add one to make 1 based
        outtbl[ 'gdna_pos' ].append( pos + 1 )

        if pos in bg.start.tolist():

            outtbl[ 'chrom' ].append( bg.loc[ bg.start == pos ].chrom.values[0] )
            outtbl[ 'phylop' ].append( bg.loc[ bg.start == pos ].phylop.values[0] )

        #position isn't in dataframe - lets use previous positions values instead
        else:

            #first make sure the position is still within the previous interval
            assert bg.loc[ bg.start < pos ].iloc[ -1 ].end > pos, \
            ' Start coordinate %i is not contained in any interval' % pos

            #since everything is position sorted, we're grabbing the lowest row of all rows that start before
            #this given position
            outtbl[ 'chrom' ].append( bg.loc[ bg.start < pos ].iloc[ -1 ].chrom )
            outtbl[ 'phylop' ].append( bg.loc[ bg.start < pos ].iloc[ -1 ].phylop )

    outtbl = pd.DataFrame( outtbl )

    return outtbl

def merge_data_phylop( byvartbl,
                       phylop_tbl,
                       index_cols = [ 'gdna_pos' ] ):

    tbv = byvartbl.copy()
    p = phylop_tbl.copy()

    merge_tbl = gn.merge_data_gnomad( tbv,
                                      p,
                                      indexcols = index_cols )

    return merge_tbl
