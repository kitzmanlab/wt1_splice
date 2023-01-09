import os
import pandas as pd
import numpy as np
import scipy.stats as ss

from collections import OrderedDict as odict  # default is for keys to come back out in order after I think python 3.7
from collections import Counter
import splanl.coords as cds


# make (ordered) dict of lists
def blanktbl(colnames):
    """Make a blank dictionary of column names --> lists
    Can append entries to these lists, then convert it to pandas DataFrame

    Args:
        colnames (list of str): column names, will become keys of resulting dict

    Returns:
        dict of column names, each associated w/ a blank list

    """
    return odict( [ (k,[]) for k in colnames] )


def merge_subasm_and_rna_tbls(
    subasm_tbl,
    rna_tbl,
  ):

    """
    Merge a subassembly and an RNA table; each should be indexed by barcode

    Args:
        subasm_tbl (pd.DataFrame): subassembly results, indexed by barcode seq
        rna_tbl (pd.DataFrame):  RNAseq results (e.g., psi values), indexed by barcode seq

    Returns:
        merged table, in DataFrame, containing the barcodes in both

    """

    # get rid of extra columns
    loc = subasm_tbl
    loc_remove = [ 'ref_target_length','minbq' ] + [c for c in loc if c.startswith('nbp_ge_') ]
    loc = [ c for c in loc if c not in loc_remove ]
    subasm_tbl2 = subasm_tbl[ loc ]

    join_tbl = pd.merge( subasm_tbl2,
                         rna_tbl,
                         how='inner',
                         left_index=True,
                         right_index=True,
                         suffixes=('_subasm','_rna') )

    return join_tbl

def compute_psi_values(
    in_df,
    iso_col = None,
    read_count_col = 'usable_reads'
):
    """

    Args:


    Returns:

    """

    out_df = in_df.copy()

    if not iso_col:
        iso_col = [ col for col in out_df.columns if col.startswith( 'iso' ) ]

    assert iso_col, 'Please specify columns for PSI'

    assert read_count_col in out_df.columns, '%s is not within %s' % ( read_count_col, str( in_df ) )

    for col in iso_col:
        out_df[ col + '_psi' ] = out_df[ col ] / out_df[ read_count_col ]

    return out_df

def summarize_byvar_singlevaronly(
    subasm_tbl,
    rna_tbl,
    min_usable_reads_per_bc,
    isonames=None ):
    """
    Summarize per-variant effects across associated barcodes.
    Considers only single-variant clones; barcodes w/ ≥1 variants are ignored.

    Args:
        subasm_tbl (pd.DataFrame): subassembly results, indexed by barcode seq
        rna_tbl (pd.DataFrame):  RNAseq results (e.g., psi values), indexed by barcode seq
        min_usable_reads_per_bc (int): min # reads associated with barcode to be considered
        isonames (list of str): names of isoforms; for each entry 'x', a column 'x_psi' should exist in rna_tbl

    Returns:
        pd.DataFrame with per-variant summary values;  mean_x, wmean_x, and median_x are the
        across barcodes mean, read-count-weighted mean, and median psi values for each
        isoform x
    """

    sa_filt = subasm_tbl.query( 'n_variants_passing==1' ).copy()

    li_rna = rna_tbl.index.intersection( sa_filt.index )

    rna_isect = rna_tbl.loc[ li_rna ].copy()

    rna_isect_psi = compute_psi_values( rna_isect, iso_col = isonames )

    if isonames is None:
        isonames = [ cn[ :cn.rindex('_') ] for cn in rna_isect_psi.columns if cn.endswith('psi') ]
        assert len(isonames)>0, 'cant infer the isoform name columns; please specify them in parameter isonames'

    rna_isect_psi['varlist'] = sa_filt.loc[ li_rna, 'variant_list' ]

    out_tbl = blanktbl(
        ['chrom','pos','ref','alt','varlist','n_bc','n_bc_passfilt',
         'sum_reads',
         'sum_reads_passfilt',
         'sum_usable_reads',
         'sum_unmapped_reads',
         'sum_badstart_reads',
         'sum_badend_reads',
         'sum_softclipped_reads',
         'sum_otheriso'] +
         [ 'mean_{}'.format(cn) for cn in isonames ] +
         [ 'wmean_{}'.format(cn) for cn in isonames ] +
         [ 'median_{}'.format(cn) for cn in isonames ]
     )

    for singlevar, subtbl in rna_isect_psi.groupby( 'varlist' ):

        subtbl_filt = subtbl.loc[ subtbl.usable_reads > min_usable_reads_per_bc ].copy()

        out_tbl['varlist'].append(singlevar)
        out_tbl['chrom'].append(singlevar.split(':')[0])
        out_tbl['pos'].append(int(singlevar.split(':')[1]))
        out_tbl['ref'].append(singlevar.split(':')[2])
        out_tbl['alt'].append(singlevar.split(':')[3])

        out_tbl['n_bc'].append( subtbl.shape[0] )
        out_tbl['n_bc_passfilt'].append( subtbl_filt.shape[0] )

        out_tbl['sum_reads'].append( subtbl['num_reads'].sum() )

        if subtbl_filt.shape[0]==0:
            out_tbl['sum_reads_passfilt'].append( 0 )
            out_tbl['sum_usable_reads'].append( 0 )
            out_tbl['sum_unmapped_reads'].append( 0 )
            out_tbl['sum_badstart_reads'].append( 0 )
            out_tbl['sum_badend_reads'].append( 0 )
            out_tbl['sum_softclipped_reads'].append( 0 )
            out_tbl['sum_otheriso'].append( 0 )

            for iso in isonames:
                out_tbl[ f'mean_{iso}' ].append( None )
                out_tbl[ f'wmean_{iso}' ].append( None )
                out_tbl[ f'median_{iso}' ].append( None )

            continue
        #this is tricky to think about
        #currently set so that its counting reads after removing the barcodes not passing the filter
        else:
            out_tbl['sum_reads_passfilt'].append( subtbl_filt['num_reads'].sum() )
            out_tbl['sum_usable_reads'].append( subtbl_filt['usable_reads'].sum()  )
            out_tbl['sum_unmapped_reads'].append( subtbl_filt['unmapped_reads'].sum()  )
            out_tbl['sum_badstart_reads'].append( subtbl_filt['bad_starts'].sum()  )
            out_tbl['sum_badend_reads'].append( subtbl_filt['bad_ends'].sum()  )
            out_tbl['sum_softclipped_reads'].append( subtbl_filt['soft_clipped'].sum()  )
            out_tbl['sum_otheriso'].append( subtbl_filt['other_isoform'].sum()  )

            for iso in isonames:
                # mean psi
                out_tbl[ f'mean_{iso}' ].append( subtbl_filt[ f'{iso}_psi' ].mean() )
                # mean psi, weighted by #usable reads
                if subtbl_filt['usable_reads'].sum() != 0:
                    out_tbl[ f'wmean_{iso}' ].append( ( subtbl_filt[ f'{iso}_psi' ] * subtbl_filt['usable_reads'] ).sum() / subtbl_filt['usable_reads'].sum() )
                else:
                    out_tbl[ f'wmean_{iso}' ].append( np.nan )
                # median psi
                out_tbl[ f'median_{iso}' ].append( subtbl_filt[ f'{iso}_psi' ].median() )

    out_tbl = pd.DataFrame( out_tbl )

    out_tbl = count_bcs_per_var_sa( out_tbl,
                                    sa_filt )

    #these two are have the total barcode/read count in the denominator
    out_tbl['per_bc_passfilt'] = 100*( out_tbl.n_bc_passfilt / out_tbl.n_bc )
    out_tbl['per_reads_passfilt'] = 100*( out_tbl.sum_reads_passfilt / out_tbl.sum_reads )

    #these columns are based of barcodes which are passing the filter
    #so only reads from barcodes passing the filter are used in the denominator
    out_tbl['per_reads_usable'] = 100*( out_tbl.sum_usable_reads / out_tbl.sum_reads_passfilt )
    out_tbl['per_unmapped'] = 100*( out_tbl.sum_unmapped_reads / out_tbl.sum_reads_passfilt )
    out_tbl['per_badend'] = 100*( out_tbl.sum_badend_reads / out_tbl.sum_reads_passfilt )
    out_tbl['per_badstart'] = 100*( out_tbl.sum_badstart_reads / out_tbl.sum_reads_passfilt )
    out_tbl['per_softclipped'] = 100*( out_tbl.sum_softclipped_reads / out_tbl.sum_reads_passfilt )
    out_tbl['per_otheriso'] = 100*( out_tbl.sum_otheriso / out_tbl.sum_reads_passfilt )


    return out_tbl

def summarize_byvar_singlevaronly_pe( subasm_tbl,
                                      rna_tbl,
                                      min_usable_reads_per_bc,
                                      summary_cols,
                                      isonames = None, ):
    """
    Summarize per-variant effects across associated barcodes.
    Considers only single-variant clones; barcodes w/ ≥1 variants are ignored.

    Args:
        subasm_tbl (pd.DataFrame): subassembly results, indexed by barcode seq
        rna_tbl (pd.DataFrame):  RNAseq results (e.g., psi values), indexed by barcode seq
        min_usable_reads_per_bc (int): min # reads associated with barcode to be considered
        isonames (list of str): names of isoforms; for each entry 'x', a column 'x_psi' should exist in rna_tbl

    Returns:
        pd.DataFrame with per-variant summary values;  mean_x, wmean_x, and median_x are the
        across barcodes mean, read-count-weighted mean, and median psi values for each
        isoform x
    """

    sa_filt = subasm_tbl.query( 'n_variants_passing==1' ).copy()

    li_rna = rna_tbl.index.intersection( sa_filt.index )

    rna_isect = rna_tbl.loc[ li_rna ].copy()

    rna_isect_psi = compute_psi_values( rna_isect, iso_col = isonames )

    if isonames is None:
        isonames = [ cn[ :cn.rindex( '_' ) ] for cn in rna_isect_psi.columns if cn.endswith( 'psi' ) ]
        assert len( isonames ) > 0, 'cant infer the isoform name columns; please specify them in parameter isonames'

    rna_isect_psi[ 'varlist' ] = sa_filt.loc[ li_rna, 'variant_list' ]

    out_tbl = blanktbl( ['chrom','pos','ref','alt','varlist','n_bc','n_bc_passfilt','sum_reads','sum_reads_passfilt', ] +
                        [ 'sum_{}'.format( cn ) for cn in summary_cols ] +
                        [ 'mean_{}'.format(cn) for cn in isonames ] +
                        [ 'wmean_{}'.format(cn) for cn in isonames ] +
                        [ 'median_{}'.format(cn) for cn in isonames ] )

    for singlevar, subtbl in rna_isect_psi.groupby( 'varlist' ):

        subtbl_filt = subtbl.loc[ subtbl.usable_reads > min_usable_reads_per_bc ].copy()

        out_tbl['varlist'].append(singlevar)
        out_tbl['chrom'].append(singlevar.split(':')[0])
        out_tbl['pos'].append(int(singlevar.split(':')[1]))
        out_tbl['ref'].append(singlevar.split(':')[2])
        out_tbl['alt'].append(singlevar.split(':')[3])

        out_tbl['n_bc'].append( subtbl.shape[0] )
        out_tbl['n_bc_passfilt'].append( subtbl_filt.shape[0] )

        out_tbl['sum_reads'].append( subtbl['num_reads'].sum() )

        if subtbl_filt.shape[0]==0:
            out_tbl['sum_reads_passfilt'].append( 0 )

            for col in summary_cols:
                out_tbl[ f'sum_{col}' ].append( 0 )

            for iso in isonames:
                out_tbl[ f'mean_{iso}' ].append( None )
                out_tbl[ f'wmean_{iso}' ].append( None )
                out_tbl[ f'median_{iso}' ].append( None )

            continue
        #this is tricky to think about
        #currently set so that its counting reads after removing the barcodes not passing the filter
        else:
            out_tbl['sum_reads_passfilt'].append( subtbl_filt['num_reads'].sum() )

            for col in summary_cols:
                out_tbl[ f'sum_{col}' ].append( subtbl_filt[ col ].sum()  )

            for iso in isonames:
                # mean psi
                out_tbl[ f'mean_{iso}' ].append( subtbl_filt[ f'{iso}_psi' ].mean() )
                # mean psi, weighted by #usable reads
                if subtbl_filt['usable_reads'].sum() != 0:
                    out_tbl[ f'wmean_{iso}' ].append( ( subtbl_filt[ f'{iso}_psi' ] * subtbl_filt['usable_reads'] ).sum() / subtbl_filt['usable_reads'].sum() )
                else:
                    out_tbl[ f'wmean_{iso}' ].append( np.nan )
                # median psi
                out_tbl[ f'median_{iso}' ].append( subtbl_filt[ f'{iso}_psi' ].median() )

    out_tbl = pd.DataFrame( out_tbl )

    out_tbl = count_bcs_per_var_sa( out_tbl,
                                    sa_filt )

    #these two are have the total barcode/read count in the denominator
    out_tbl['per_bc_passfilt'] = 100*( out_tbl.n_bc_passfilt / out_tbl.n_bc )
    out_tbl['per_reads_passfilt'] = 100*( out_tbl.sum_reads_passfilt / out_tbl.sum_reads )

    #these columns are based of barcodes which are passing the filter
    #so only reads from barcodes passing the filter are used in the denominator
    for col in summary_cols:
        out_tbl[ f'per_{col}' ] = 100*( out_tbl[ f'sum_{col}' ] / out_tbl.sum_reads_passfilt )

    return out_tbl

def count_bcs_per_var_sa( rna_tbl,
                          satbl ):

    #count the number of rows for each variant
    #this is the number of barcodes for that variant
    count_tbl = satbl.copy().groupby( [ 'variant_list' ] )[ 'variant_list' ].count().rename('n_bc_sa')

    rna_tbl = rna_tbl.copy().set_index( 'varlist' )

    out_tbl = pd.merge( rna_tbl, count_tbl, left_index = True, right_index = True )

    out_tbl.index.name = 'varlist'

    out_tbl = out_tbl.reset_index()

    assert rna_tbl.shape[0] == out_tbl.shape[0], 'RNA table rows were lost in the merge'

    return out_tbl

def summarize_byvar_WT( rna_tbl,
                        exon_coords,
                        min_usable_reads_per_bc,
                        chrom,
                        isonames=None ):
    """
    Summarize per-variant effects across associated barcodes.
    Considers only single-variant clones; barcodes w/ ≥1 variants are ignored.

    Args:
        subasm_tbl (pd.DataFrame): subassembly results, indexed by barcode seq
        rna_tbl (pd.DataFrame):  RNAseq results (e.g., psi values), indexed by barcode seq
        exon_coords (list of int tuples): coordinates of cloned exons
        min_usable_reads_per_bc (int): min # reads associated with barcode to be considered
        isonames (list of str): names of isoforms; for each entry 'x', a column 'x_psi' should exist in rna_tbl

    Returns:
        pd.DataFrame with per-variant summary values;  mean_x, wmean_x, and median_x are the
        across barcodes mean, read-count-weighted mean, and median psi values for each
        isoform x
    """

    rna_psi = compute_psi_values( rna_tbl, iso_col = isonames )

    if isonames is None:
        isonames = [ cn[ :cn.rindex('_') ] for cn in rna_psi.columns if cn.endswith('psi') ]
        assert len(isonames)>0, 'cant infer the isoform name columns; please specify them in parameter isonames'

    rna_psi_filt = rna_psi.loc[ rna_psi.usable_reads > min_usable_reads_per_bc ].copy()

    out_tbl = {}

    out_tbl[ 'chrom' ] = [ chrom ]
    out_tbl[ 'varlist' ] = [ 'WT' ]

    out_tbl[ 'n_bc' ] = [ rna_psi.shape[ 0 ] ]
    out_tbl[ 'n_bc_passfilt' ] = [ rna_psi_filt.shape[ 0 ] ]

    out_tbl[ 'sum_reads' ] = [ rna_psi.num_reads.sum() ]

    if rna_psi_filt.shape[ 0 ] == 0:

        out_tbl[ 'sum_reads_passfilt' ] = [ 0 ]
        out_tbl[ 'sum_usable_reads' ] = [ 0 ]
        out_tbl[ 'sum_unmapped_reads' ] = [ 0 ]
        out_tbl[ 'sum_badstart_reads' ] = [ 0 ]
        out_tbl[ 'sum_badend_reads' ] = [ 0 ]
        out_tbl[ 'sum_softclipped_reads' ] = [ 0 ]
        out_tbl[ 'sum_otheriso' ] = [ 0 ]

        for iso in isonames:
            out_tbl[ f'mean_{iso}' ] = [ None ]
            out_tbl[ f'wmean_{iso}' ] = [ None ]
            out_tbl[ f'median_{iso}' ] = [ None ]
            out_tbl[ f'stdev_{iso}' ] = [ None ]

    else:

        out_tbl[ 'sum_reads_passfilt' ] = [ rna_psi_filt.num_reads.sum() ]
        out_tbl[ 'sum_usable_reads' ] = [ rna_psi_filt.usable_reads.sum() ]
        out_tbl[ 'sum_unmapped_reads' ] = [ rna_psi_filt.unmapped_reads.sum() ]
        out_tbl[ 'sum_badstart_reads' ] = [ rna_psi_filt.bad_starts.sum() ]
        out_tbl[ 'sum_badend_reads' ] = [ rna_psi_filt.bad_ends.sum() ]
        out_tbl[ 'sum_softclipped_reads' ] = [ rna_psi_filt.soft_clipped.sum() ]
        out_tbl[ 'sum_otheriso' ] = [ rna_psi_filt.other_isoform.sum() ]

        for iso in isonames:
            # mean psi
            out_tbl[ f'mean_{iso}' ] = [ rna_psi_filt[ f'{iso}_psi' ].mean() ]
            # mean psi, weighted by #usable reads
            if rna_psi_filt.usable_reads.sum() != 0:
                out_tbl[ f'wmean_{iso}' ] = [ ( rna_psi_filt[ f'{iso}_psi' ] * rna_psi_filt.usable_reads ).sum() / rna_psi_filt.usable_reads.sum() ]
            else:
                out_tbl[ f'wmean_{iso}' ] = [ np.nan ]
                # median psi
            out_tbl[ f'median_{iso}' ] = [ rna_psi_filt[ f'{iso}_psi' ].median() ]
            out_tbl[ f'stdev_{iso}' ] = [ rna_psi_filt[ f'{iso}_psi' ].std() ]
            out_tbl[ f'wstdev_{iso}' ] = [ np.sqrt( ( rna_psi_filt.usable_reads  * ( rna_psi_filt[ f'{iso}_psi' ] - out_tbl[ f'wmean_{iso}' ][ 0 ] )**2 ).sum() \
                                                    / ( rna_psi_filt.usable_reads.sum() - 1 ) ) ]

    out_tbl = pd.DataFrame( out_tbl )

    #these two are have the total barcode/read count in the denominator
    out_tbl['per_bc_passfilt'] = 100*( out_tbl.n_bc_passfilt / out_tbl.n_bc )
    out_tbl['per_reads_passfilt'] = 100*( out_tbl.sum_reads_passfilt / out_tbl.sum_reads )

    #these columns are based of barcodes which are passing the filter
    #so only reads from barcodes passing the filter are used in the denominator
    out_tbl['per_reads_usable'] = 100*( out_tbl.sum_usable_reads / out_tbl.sum_reads_passfilt )
    out_tbl['per_unmapped'] = 100*( out_tbl.sum_unmapped_reads / out_tbl.sum_reads_passfilt )
    out_tbl['per_badend'] = 100*( out_tbl.sum_badend_reads / out_tbl.sum_reads_passfilt )
    out_tbl['per_badstart'] = 100*( out_tbl.sum_badstart_reads / out_tbl.sum_reads_passfilt )
    out_tbl['per_softclipped'] = 100*( out_tbl.sum_softclipped_reads / out_tbl.sum_reads_passfilt )
    out_tbl['per_otheriso'] = 100*( out_tbl.sum_otheriso / out_tbl.sum_reads_passfilt )

    return out_tbl


####################################
#
# routines to combine replicate per-variant tables
#

def combine_rep_pervartbls_wide(
    ltbls,
    lsampnames,
    indexcols=['chrom','pos','ref','alt','varlist'],
    group_cols_by_samp=False ):

    """
    Combine replicate variant effect tables in wide format

    Args:
        ltbls (list of pd.DataFrame): list of per-variant effect tables, one per replicate or condition
        lsampnames (list of str): list of respective names for those replciates or conditions
        indexcols (list of str): what columns to use to index each variant table
        group_cols_by_samp (bool): should columns from each sample by grouped together

    Returns:
        New pd.DataFrame with by variant effect tables merged together. There may be NAs for variants that are absent from some of the reps/conditions.
    """

    ltbls_ixd = [ tbl.set_index(indexcols).copy() for tbl in ltbls ]

    lcolnames = list(ltbls_ixd[0].columns)

    # all tables must have the same set of columns
    for t in ltbls_ixd:
        assert list(t.columns)==lcolnames

    for (tbl,sampname) in zip(ltbls_ixd,lsampnames):
        tbl.columns = [ '{}_{}'.format( sampname,col ) for col in tbl.columns ]

    tblout = pd.concat( ltbls_ixd, axis=1 )

    if group_cols_by_samp:
        loc = []
        for col in lcolnames:
            loc += [ '{}_{}'.format(sampname,col) for sampname in lsampnames  ]
        tblout = tblout[loc]


    tblout = tblout.reset_index()

    return tblout

def create_variables_across_samples( wide_tbl,
                                     lsampnames,
                                     median_cols = [],
                                     mean_cols = [],
                                     sum_cols = [],
                                     max_cols = [] ):

    wide = wide_tbl.copy()

    if median_cols:

        for col in median_cols:

            samp_cols = [ '_'.join( [ samp, col ] ) for samp in lsampnames ]

            wide[ col + '_med' ] = wide[ samp_cols ].median( axis = 1 )

    if mean_cols:

        for col in mean_cols:

            samp_cols = [ '_'.join( [ samp, col ] ) for samp in lsampnames ]

            wide[ col + '_mean' ] = wide[ samp_cols ].mean( axis = 1 )

    if sum_cols:

        for col in sum_cols:

            samp_cols = [ '_'.join( [ samp, col ] ) for samp in lsampnames ]

            wide[ col + '_sum' ] = wide[ samp_cols ].sum( axis = 1 )

    if max_cols:

        for col in max_cols:

            samp_cols = [ '_'.join( [ samp, col ] ) for samp in lsampnames ]

            wide[ col + '_max' ] = wide[ samp_cols ].max( axis = 1 )

    return wide

def compute_bc_weighted_psi( wide_tbl,
                             lsampnames,
                             isonames,
                             bccount,
                              ):

    wide = wide_tbl.copy()

    samp_bc_cols = [ '_'.join( [ samp, bccount ] ) for samp in lsampnames ]

    for col in isonames:

        #we tend to use the wmean more often so this is intentionally a mean of the wmeans
        wide[ 'mean_' + col ] = wide[ [ '_'.join( [ samp, 'wmean', col ] ) for samp in lsampnames ] ].mean( axis = 1 )

        #this probably would look better as a numpy array dot product but we survived
        wide[ 'wmean_' + col ] = pd.DataFrame( ( wide[ '_'.join( [ samp, 'wmean', col ] ) ] * wide[ '_'.join( [ samp, bccount ] ) ]
                                                   for samp in lsampnames ) ).T.sum( axis = 1) \
                                             / wide[ samp_bc_cols ].sum( axis = 1 )

    return wide

def combine_rep_pervartbls_long(
    ltbls,
    lsampnames,
    indexcols=['chrom','pos','ref','alt','varlist'],
):

    """
    Combine replicate variant effect tables in long format

    Args:
        ltbls (list of pd.DataFrame): list of per-variant effect tables, one per replicate or condition
        lsampnames (list of str): list of respective names for those replciates or conditions
        indexcols (list of str): what columns to use to index each variant table

    Returns:
        New pd.DataFrame with by variant effect tables merged together, with each replicate appearing as a separate row
    """

    ltbls_ixd = [ tbl.set_index(indexcols).copy() for tbl in ltbls ]

    lcolnames = list(ltbls_ixd[0].columns)

    # all tables must have the same set of columns
    for t in ltbls_ixd:
        assert list(t.columns)==lcolnames

    for (tbl,sampname) in zip(ltbls_ixd,lsampnames):

        #allows us to create long tables from long tables that already include a sample column
        if 'sample' in lcolnames:
            tbl['sample_grp']=sampname
        else:
            tbl['sample']=sampname

    tblout = pd.concat( ltbls_ixd, axis=0 )

    tblout = tblout[ ['sample_grp' if 'sample' in lcolnames else 'sample']+[cn for cn in lcolnames] ]

    tblout = tblout.reset_index()

    return tblout

def combine_rep_perbctbls_long(
    ltbls,
    lsampnames
):

    """
    Combine replicate variant effect tables in long format

    Args:
        ltbls (list of pd.DataFrame): list of per-variant effect tables, one per replicate or condition
        lsampnames (list of str): list of respective names for those replciates or conditions
        indexcols (list of str): what columns to use to index each variant table

    Returns:
        New pd.DataFrame with by variant effect tables merged together, with each replicate appearing as a separate row
    """

    ltbls_ixd = [ tbl.copy() for tbl in ltbls ]

    lcolnames = list(ltbls_ixd[0].columns)

    # all tables must have the same set of columns
    for t in ltbls_ixd:
        assert list(t.columns)==lcolnames

    for (tbl,sampname) in zip(ltbls_ixd,lsampnames):

        #allows us to create long tables from long tables that already include a sample column
        if 'sample' in lcolnames:
            tbl['sample_grp']=sampname
        else:
            tbl['sample']=sampname

    tblout = pd.concat( ltbls_ixd, axis=0 )

    tblout = tblout[ ['sample_grp' if 'sample' in lcolnames else 'sample']+[cn for cn in lcolnames] ]

    tblout.index.name = 'barcode'

    return tblout


def combine_allisos_pervartbls_long(
                                    ltbls,
                                    lsampnames,
                                    indexcols=['chrom','pos','ref','alt','varlist'],
                                    ):

    """
    Combine replicate variant effect tables with all isoforms (not necessarily matching column names) in long format

    Args:
        ltbls (list of pd.DataFrame): list of per-variant effect tables with all isoforms, one per replicate or condition
        lsampnames (list of str): list of respective names for those replciates or conditions
        indexcols (list of str): what columns to use to index each variant table

    Returns:
        New pd.DataFrame with by variant effect tables merged together, with each replicate appearing as a separate row
        Columns not represented in one input dataframe compared to other input dataframes will contain nan values
    """

    lcolnames = list( set( [ col for tbl in ltbls for col in tbl ] ) )

    ltbls_ixd = [ tbl.set_index( indexcols ).copy() for tbl in ltbls ]

    for (tbl,sampname) in zip(ltbls_ixd,lsampnames):

        #allows us to create long tables from long tables that already include a sample column
        if 'sample' in lcolnames:
            tbl['sample_grp']=sampname
        else:
            tbl['sample']=sampname

    tblout = ltbls_ixd[0].append( ltbls_ixd[1:], sort = True)

    tblout = tblout[ ['sample_grp' if 'sample' in lcolnames else 'sample']+[cn for cn in lcolnames if cn not in indexcols ] ].reset_index()

    return tblout

def combine_allisos_perbctbls_long(
    ltbls,
    lsampnames
):

    """
    Combine replicate barcode effect tables with all isoforms (not necessarily matching column names) in long format

    Args:
        ltbls (list of pd.DataFrame): list of per-variant effect tables with all isoforms, one per replicate or condition
        lsampnames (list of str): list of respective names for those replciates or conditions
        indexcols (list of str): what columns to use to index each variant table

    Returns:
        New pd.DataFrame with by variant effect tables merged together, with each replicate appearing as a separate row
        Columns not represented in one input dataframe compared to other input dataframes will contain nan values
    """

    lcolnames = list( set( [ col for tbl in ltbls for col in tbl ] ) )

    ltbls_ixd = [ tbl.copy() for tbl in ltbls ]

    for (tbl,sampname) in zip(ltbls_ixd,lsampnames):

        #allows us to create long tables from long tables that already include a sample column
        if 'sample' in lcolnames:
            tbl['sample_grp']=sampname
        else:
            tbl['sample']=sampname

    tblout = ltbls_ixd[0].append( ltbls_ixd[1:], sort = True)

    tblout = tblout[ ['sample_grp' if 'sample' in lcolnames else 'sample']+[cn for cn in lcolnames] ]

    #makes all missing columns 0's
    tblout = tblout.fillna(0)

    tblout.index.name = 'barcode'

    return tblout

def filter_byvartbl_snvonly(
    byvar_tbl
):
    """
    Filter by variant table to only SNVs

    Args:
        byvar_tbl (pd.DataFrame): per-variant effect table
    Returns:
        Copy of per-variant effect table with only SNV lines included
    """

    byvar_snvonly = byvar_tbl.loc[ (byvar_tbl.ref.str.len() == 1) & (byvar_tbl.alt.str.len() == 1) ].copy()
    return byvar_snvonly

def create_sparse_df( large_df,
                        sparse_val = 'nan' ):

    """Make sure the large_df has no meaningful index columns - ie enter as df.reset_index()"""

    in_df = large_df.copy()

    non_num_col = list( in_df.select_dtypes(exclude=np.number).columns )

    #the sparse dataframe can't have any non-numerical columns so lets set them all as indices
    out_df = in_df.set_index( non_num_col ).select_dtypes(include=np.number)

    if sparse_val == 'nan':
        out_df = out_df.astype( pd.SparseDtype( "float", np.nan ) )
    elif isinstance(sparse_val, int):
        out_df = out_df.astype( pd.SparseDtype( "float", sparse_val ) )

    return out_df.reset_index()
