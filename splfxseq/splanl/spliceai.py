import pandas as pd
import numpy as np
import pysam
import pybedtools

def create_spliceai_df( spliceai_tbx,
                        chrom,
                        coords ):

    out_tbl = { 'chrom': [],
                'pos': [],
                'ref': [],
                'alt': [],
                'DS_accept_GAIN': [],
                'DS_accept_LOSS': [],
                'DS_donor_GAIN': [],
                'DS_donor_LOSS': [],
                'POS_accept_GAIN': [],
                'POS_accept_LOSS': [],
                'POS_donor_GAIN': [],
                'POS_donor_LOSS': [] }

    if chrom:
        chrom = chrom.replace('chr','')

        if is_int( chrom ):
            chrom = int( chrom )

    for row in spliceai_tbx.fetch(  chrom ,
                                    coords[0],
                                    coords[1],
                                    parser=pysam.asVCF() ):

        out_tbl[ 'chrom' ].append( chrom )
        #add one to account for stupid 0 and 1 based indexing issues
        out_tbl[ 'pos' ].append(row.pos+1)
        out_tbl[ 'ref' ].append(row.ref)
        out_tbl[ 'alt' ].append(row.alt)

        values=row.info.split('|')
        out_tbl['DS_accept_GAIN'].append( float( values[2] ) )
        out_tbl['DS_accept_LOSS'].append( float( values[3] ) )
        out_tbl['DS_donor_GAIN'].append( float( values[4] ) )
        out_tbl['DS_donor_LOSS'].append( float( values[5] ) )
        out_tbl['POS_accept_GAIN'].append( int( values[6] ) )
        out_tbl['POS_accept_LOSS'].append( int( values[7] ) )
        out_tbl['POS_donor_GAIN'].append( int( values[8] ) )
        out_tbl['POS_donor_LOSS'].append( int( values[9] ) )

    out_tbl=pd.DataFrame( out_tbl )

    out_tbl['DS_max'] = out_tbl[['DS_accept_GAIN','DS_accept_LOSS','DS_donor_GAIN','DS_donor_LOSS']].max(axis=1)
    out_tbl['DS_max_type'] = out_tbl[['DS_accept_GAIN','DS_accept_LOSS','DS_donor_GAIN','DS_donor_LOSS']].idxmax(axis=1)
    POS_max_type=pd.Series([col.replace('DS','POS') for col in out_tbl['DS_max_type']])
    out_tbl['POS_max'] = out_tbl.lookup(POS_max_type.index, POS_max_type.values)

    return out_tbl

def is_int( str ):
  try:
    int( str )
    return True
  except ValueError:
    return False

def get_exon_coords( ccds_exons_df,
                    gene_name,
                    exon_nums = None,
                    intron_flank = 200,
                    rev_trans = False ):

    exon_df = ccds_exons_df.copy().sort_values( by = [ 'chrom', 'start' ] )

    exon_df = exon_df.loc[ exon_df.gene == gene_name.upper() ]

    assert exon_df.shape[0] > 0, 'The gene name does not exist in the dataset - check for typos'

    assert len( set( exon_df.chrom ) ) == 1, 'This function is not intended to get coordinates across multiple chromosomes'

    if rev_trans:
        exon_df = exon_df.sort_values( by = [ 'chrom', 'start' ], ascending = False )

    if exon_nums:
        if isinstance( exon_nums, int ):
            exon_df = exon_df.iloc[ exon_nums - 1 ]
        elif isinstance( exon_nums, tuple ):
            exon_df = exon_df.iloc[ ( exon_nums[0] - 1 ): exon_nums[1] ]
        else:
            raise ValueError( 'Exon numbers but either be None, an integer, or a tuple of ranges' )

    #if the dataframe only has one row now
    if exon_df.shape == ( 4, ):
        chrom = exon_df.chrom
        max_chrom_pos = ccds_exons_df.loc[ ccds_exons_df.chrom == chrom ].end.max()
        start = exon_df.start - intron_flank if exon_df.start >= intron_flank else 0
        end = exon_df.end + intron_flank if ( exon_df.end + intron_flank ) <= max_chrom_pos else max_chrom_pos
        coords = ( start , end )
    else:
        chrom = exon_df.chrom.values[0]
        max_chrom_pos = ccds_exons_df.loc[ ccds_exons_df.chrom == chrom ].end.max()
        start = exon_df.start.min() - intron_flank if exon_df.start.min() >= intron_flank else 0
        end = exon_df.end.max() + intron_flank if ( exon_df.end.max() + intron_flank ) <= max_chrom_pos else max_chrom_pos
        coords = ( start , end )

    return chrom, coords

def create_exon_coord_df( exon_coord_file = '/nfs/kitzman2/smithcat/proj/spliceAI/ccdsGenes.exons.hg19.bed' ):

    exon_df = pd.read_table( exon_coord_file,
                            names = ['chrom', 'start', 'end', 'gene'])

    return exon_df

def rev_comp_seq( refseq ):

    comp_DNA = str.maketrans( 'ACGTacgt', 'TGCAtgca' )

    return refseq.translate( comp_DNA )

def merge_exper_spliceai( exper_df,
                          spliceai_df,
                          exper_idx_col = [ 'chrom', 'pos', 'ref', 'alt' ],
                          spliceai_idx_col = [ 'chrom', 'pos', 'ref', 'alt' ] ):

    edf = exper_df.set_index( exper_idx_col ).copy()
    sdf = spliceai_df.set_index( spliceai_idx_col ).copy()

    intersect_idx = edf.index.intersection( sdf.index )
    sdf_filt = sdf.loc[ intersect_idx ]

    out_tbl = pd.concat( [ sdf_filt, edf ], axis = 1 )
    out_tbl.index.names = spliceai_idx_col
    out_tbl = out_tbl.reset_index()

    return out_tbl

def count_SDVs( spliceai_df,
                exon_coords,
                intron_flank = 200,
                spliceai_thresh = [ .5 ] ):

    sa_df = spliceai_df.set_index( 'pos' ).copy()

    out_tbl = {
                'exon': [],
                'n_ex_bp': [],
                'n_us_int_bp': [],
                'n_ds_int_bp': [],
            }

    for t in spliceai_thresh:
        out_tbl[ 'n_ex_abv_'+str( t ) ] = []
        out_tbl[ 'per_ex_abv_'+str( t ) ] = []
        out_tbl[ 'n_us_int_abv_'+str( t ) ] = []
        out_tbl[ 'per_us_int_abv_'+str( t ) ] = []
        out_tbl[ 'n_ds_int_abv_'+str( t ) ] = []
        out_tbl[ 'per_ds_int_abv_'+str( t ) ] = []

    for ex, coords in exon_coords.items():

        ex_df = sa_df.loc[ coords[0]:coords[1] ]

        out_tbl[ 'exon' ].append( int( ex ) )
        out_tbl[ 'n_ex_bp' ].append( int( coords[1] - coords[0] ) )

        for t in spliceai_thresh:
            if ex_df.shape[ 0 ] > 0:
                out_tbl[ 'n_ex_abv_'+str( t ) ].append( ex_df.loc[ ex_df.DS_max >= t ].shape[0] )
                out_tbl[ 'per_ex_abv_'+str( t ) ].append( 100*( out_tbl[ 'n_ex_abv_'+str( t ) ][-1] / ( ex_df.shape[0] ) ) )
            else:
                out_tbl[ 'n_ex_abv_'+str( t ) ].append( np.nan )
                out_tbl[ 'per_ex_abv_'+str( t ) ].append( np.nan )

        us_int_df = sa_df.loc[ coords[0] - intron_flank :coords[0] - 1 ]

        out_tbl[ 'n_us_int_bp' ].append( int( us_int_df.shape[0] / 3 ) )

        for t in spliceai_thresh:
            if us_int_df.shape[0] > 0:
                out_tbl[ 'n_us_int_abv_'+str( t ) ].append( us_int_df.loc[ us_int_df.DS_max >= t ].shape[0] )
                out_tbl[ 'per_us_int_abv_'+str( t ) ].append( 100*( out_tbl[ 'n_us_int_abv_'+str( t ) ][-1] / ( us_int_df.shape[0] ) ) )
            else:
                out_tbl[ 'n_us_int_abv_'+str( t ) ].append( np.nan )
                out_tbl[ 'per_us_int_abv_'+str( t ) ].append( np.nan )

        ds_int_df = sa_df.loc[ coords[1] + 1 :coords[1] + intron_flank ]

        out_tbl[ 'n_ds_int_bp' ].append( int( ds_int_df.shape[0] / 3 ) )

        for t in spliceai_thresh:
            if ds_int_df.shape[0] > 0:
                out_tbl[ 'n_ds_int_abv_'+str( t ) ].append( ds_int_df.loc[ ds_int_df.DS_max >= t ].shape[0] )
                out_tbl[ 'per_ds_int_abv_'+str( t ) ].append( 100*( out_tbl[ 'n_ds_int_abv_'+str( t ) ][-1] / ( ds_int_df.shape[0] ) ) )
            else:
                out_tbl[ 'n_ds_int_abv_'+str( t ) ].append( np.nan )
                out_tbl[ 'per_ds_int_abv_'+str( t ) ].append( np.nan )

    out_tbl[ 'exon' ].append( None )
    out_tbl[ 'n_ex_bp' ].append( sum( out_tbl[ 'n_ex_bp' ] ) )
    out_tbl[ 'n_us_int_bp' ].append( sum( out_tbl[ 'n_us_int_bp' ] ) )
    out_tbl[ 'n_ds_int_bp' ].append( sum( out_tbl[ 'n_ds_int_bp' ] ) )

    for t in spliceai_thresh:
        out_tbl[ 'n_ex_abv_'+str( t ) ].append( np.nansum( out_tbl[ 'n_ex_abv_'+str( t ) ] ) )
        out_tbl[ 'per_ex_abv_'+str( t ) ].append( 100*( out_tbl[ 'n_ex_abv_'+str( t ) ][-1] / ( 3*out_tbl[ 'n_ex_bp' ][-1] ) ) )
        out_tbl[ 'n_us_int_abv_'+str( t ) ].append( np.nansum( out_tbl[ 'n_us_int_abv_'+str( t ) ] ) )
        out_tbl[ 'per_us_int_abv_'+str( t ) ].append( 100*( out_tbl[ 'n_ex_abv_'+str( t ) ][-1] / ( 3*out_tbl[ 'n_us_int_bp' ][-1] ) ) )
        out_tbl[ 'n_ds_int_abv_'+str( t ) ].append( np.nansum( out_tbl[ 'n_ds_int_abv_'+str( t ) ] ) )
        out_tbl[ 'per_ds_int_abv_'+str( t ) ].append( 100*( out_tbl[ 'n_ex_abv_'+str( t ) ][-1] / ( 3*out_tbl[ 'n_ds_int_bp' ][-1] ) ) )

    out_tbl = pd.DataFrame( out_tbl )

    return out_tbl
