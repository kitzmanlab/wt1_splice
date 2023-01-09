import pysam
import pandas as pd
import numpy as np

def create_gnomad_df( gnomad_tbx,
                        chrom,
                        coords, ):

    out_tbl = { 'chrom': [],
                'gdna_pos_hg38': [],
                'ref': [],
                'alt': [],
                'n_alt': [],
                'n_allele': [],
                'n_homo': [], }

    for row in gnomad_tbx.fetch( chrom ,
                                 coords[0],
                                 coords[1],
                                 parser=pysam.asVCF() ):

        out_tbl[ 'chrom' ].append( chrom )
        #add one to account for stupid 0 and 1 based indexing issues
        out_tbl[ 'gdna_pos_hg38' ].append( row.pos + 1 )

        out_tbl[ 'ref' ].append( row.ref )
        out_tbl[ 'alt' ].append( row.alt )

        linfo = row.info.split(';')
        #turn it into a dictionary so we can grab based on key names
        dinfo = { i.split( '=' )[ 0 ]: i.split( '=' )[1] for i in linfo if '=' in i }

        #keep int counts to avoid any loss of info for small allele freq
        out_tbl[ 'n_alt' ].append( int( dinfo['AC'] ) )
        out_tbl[ 'n_allele' ].append( int( dinfo['AN'] ) )
        out_tbl[ 'n_homo' ].append( int( dinfo['nhomalt'] ) )

    out_df = pd.DataFrame( out_tbl )

    out_df[ 'af' ] = out_df.n_alt / out_df.n_allele

    return out_df

def is_int( str ):
  try:
    int( str )
    return True
  except ValueError:
    return False

def merge_v2_v3( v2_df,
                 v3_df,
                 indexcols = [ 'chrom', 'gdna_pos_hg38', 'ref', 'alt' ]):

    v2 = v2_df.set_index( indexcols ).copy()
    v3 = v3_df.set_index( indexcols ).copy()

    out = v2.join( v3, how = 'outer', lsuffix = '_v2', rsuffix = '_v3' )

    #sum of all columns
    for col in out.columns:

        if col.endswith( '_v2' ):

            out[ col[ :-3 ] ] = out[ col ].fillna( 0 ) + out[ col[:-1] + '3' ].fillna( 0 )

    #drop all version 2, version 3 specific columns
    out = out.drop( columns = [ col for col in out.columns if col.endswith('_v2') or col.endswith('_v3') ] )

    out[ 'alt_AF' ] = out.n_alt / out.n_allele

    out = out.reset_index()

    return out

def merge_data_gnomad( psi_df,
                       gnomad_df,
                       indexcols = [ 'gdna_pos_hg38', 'ref', 'alt' ],
                       suffix = None ):

    psi = psi_df.set_index( indexcols ).copy()
    gnomad = gnomad_df.set_index( indexcols ).copy()

    #drop gnomad chromosome info
    gnomad = gnomad.drop( columns = [ 'chrom' ] )

    if suffix:
        renamed = { col: col + '_' + suffix for col in gnomad.columns }
        gnomad = gnomad.rename( columns = renamed )

    out = psi.join( gnomad, how = 'left' ).sort_values( by = 'pos' ).reset_index()

    return out

def gnomad_var( vartbl ):

    tbv = vartbl.copy()

    tbv[ 'gnomad_var' ] = False

    tbv.loc[ ~( tbv.n_allele_v2.isnull() ) | ~( tbv.n_allele_v3.isnull() ), 'gnomad_var' ] = True

    return( tbv )
