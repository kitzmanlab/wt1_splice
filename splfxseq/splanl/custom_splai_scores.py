import pandas as pd
import numpy as np
from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import pysam
import splanl.post_processing as pp
import matplotlib.pyplot as plt

def rev_complement( seq ):
    """
    Creates reverse complement of DNA string (not case sensitive)

    Args: seq (str)

    Returns: comp (str) reverse complement of input string
    """

    trans_tbl = str.maketrans( 'ACGTNacgtn', 'TGCANtgcan' )

    rev = seq[::-1]

    comp = rev.translate( trans_tbl )

    return comp

def get_gene_bds( annots_df,
                  chrom,
                  position,
                  strand,
                  scored_context,
                  unscored_context = 5000,
                ):
    """
    Gets the number of bases which are within the sequence context but outside the gene boundaries on each side.

    Args: annots_df (pandas df) - columns: #NAME, CHROM, STRAND, TX_START, TX_END, EXON_START, EXON_END
                                    EXON_START, EXON_END are both comma separated strings of all exon bds
          chrom - (str) chromosome of center variant ( format should match your annots file (ie chr3 v 3) )
          position - (int) position (genomic coords) of center variant
          scored_context - (int) number of bases to score on each side of the variant
          unscored_context - (int) number of flanking unscored bases on each side of the variant

    Returns: gene_bds - (tuple of ints) ( upstream bases outside of gene bds, downstream bases outside of gene bds )
                        if sequence is contained entirely within the gene, returns ( 0, 0 )
                        These are the number of bases that will be replaced with N's on either side of the sequence
    """

    ann_df = annots_df.copy()

    idx = ann_df.index[ ( ann_df.CHROM == chrom )
                      & ( ann_df.TX_START <= position )
                      & ( ann_df.TX_END >= position )
                      & ( ann_df.STRAND == strand ) ]

    if len( idx ) != 1:
        print( 'The chromosome and position is not matching exactly one gene at %s:%i!' % ( chrom, position ) )

    tx_startsd = position - ( ann_df.at[ idx[ 0 ], 'TX_START' ] + 1 )

    tx_endsd = ann_df.at[ idx[ 0 ], 'TX_END' ] - position

    flanks = unscored_context + scored_context

    gene_bds = ( max( flanks - tx_startsd, 0 ), max( flanks - tx_endsd, 0 ) )

    return gene_bds

def get_2exon_bds( annots_df,
                  chrom,
                  position,
                  rev_strand = False,
                ):
    """
    Gets the distance from the center variant to the nearest acceptor and donor.

    Args: annots_df (pandas df) - columns: #NAME, CHROM, STRAND, TX_START, TX_END, EXON_START, EXON_END
                                    EXON_START, EXON_END are both comma separated strings of all exon bds
          chrom - (str) chromosome of center variant ( format should match your annots file (ie chr3 v 3) )
          position - (int) position (genomic coords) of center variant
          rev_strand - (bool) is the variant on the reverse strand?

    Returns: 2exon_bds - (tuple of ints) ( distance to nearest acceptor, distance to nearest donor )
                         SpliceAI default scoring would only return distance to nearest donor OR acceptor
                         These values are used for masking..
    """

    ann_df = annots_df.copy()

    idx = ann_df.index[ ( ann_df.CHROM == chrom )
                      & ( ann_df.TX_START <= position )
                      & ( ann_df.TX_END >= position ) ]

    assert len( idx ) == 1, \
    'The chromosome and position is not matching exactly one gene!'

    #add 1 to adjust to 0-based coords
    #compute distance to center variant
    exon_startd = [ ( int( start ) + 1 ) - position
                        for start in ann_df.at[ idx[ 0 ], 'EXON_START' ].split( ',' )
                        if start != '' ]

    start = exon_startd[ np.argmin( np.abs( exon_startd ) ) ]

    #compute distance to center variant
    exon_endd = [ int( end ) - position
                      for end in ann_df.at[ idx[ 0 ], 'EXON_END' ].split( ',' )
                      if end != '' ]

    end = exon_endd[ np.argmin( np.abs( exon_endd ) ) ]

    #if reverse strand, flip acceptors/donors
    exon_bds = ( start, end ) if not rev_strand else ( end, start )

    return exon_bds

def get_allexon_dist( annots_df,
                     chrom,
                     position,
                     scored_context,
                     rev_strand = False,
                ):
    """
    Gets the distance from the center variant to the nearest acceptor and donor.

    Args: annots_df (pandas df) - columns: #NAME, CHROM, STRAND, TX_START, TX_END, EXON_START, EXON_END
                                    EXON_START, EXON_END are both comma separated strings of all exon bds
          chrom - (str) chromosome of center variant ( format should match your annots file (ie chr3 v 3) )
          position - (int) position (genomic coords) of center variant
          rev_strand - (bool) is the variant on the reverse strand?

    Returns: 2exon_bds - (tuple of ints) ( distance to nearest acceptor, distance to nearest donor )
                         SpliceAI default scoring would only return distance to nearest donor OR acceptor
                         These values are used for masking..
    """

    ann_df = annots_df.copy()

    idx = ann_df.index[ ( ann_df.CHROM == chrom )
                      & ( ann_df.TX_START <= position )
                      & ( ann_df.TX_END >= position ) ]

    assert len( idx ) == 1, \
    'The chromosome and position is not matching exactly one gene!'

    #add 1 to adjust to 0-based coords
    #compute distance to center variant
    exon_startd = [ ( int( start ) + 1 ) - position
                        for start in ann_df.at[ idx[ 0 ], 'EXON_START' ].split( ',' )
                        if start != '' ]

    starts = [ start for start in exon_startd if np.abs( start ) <= scored_context ]

    #compute distance to center variant
    exon_endd = [ int( end ) - position
                      for end in ann_df.at[ idx[ 0 ], 'EXON_END' ].split( ',' )
                      if end != '' ]

    ends = [ end for end in exon_endd if np.abs( end ) <= scored_context ]

    #if reverse strand, flip acceptors/donors
    exon_bds = ( starts, ends ) if not rev_strand else ( ends, starts )

    return exon_bds

def create_input_seq( refseq,
                      center_var,
                      haplotype,
                      ref_var,
                      gene_bds,
                      scored_context,
                      rev_strand = False,
                      unscored_context = 5000,
                    ):
    """
    Creates the reference and variant sequences to input into SpliceAI models

    Args: refseq (str) - fasta file for an entire chromosome
          center_var (tuple) - center_variant to be scored:
                                ( position - (int) position (genomic coords) of center variant,
                                 reference base(s) - (str) reference base(s) relative to forward strand,
                                 alternate base(s) - (str) alternate base(s) relative to forward strand, )
          haplotype ( list of tuples ) - other variants to be added to the variant sequences
                                        ( position - (int) position (genomic coords) of variant,
                                        reference base(s) - (str) reference base(s) relative to forward strand,
                                        alternate base(s) - (str) alternate base(s) relative to forward strand, )
                                        Empty list adds no additional variants
          ref_var ( list of tuples ) - other variants to be added to the reference AND variant sequences
                                       ( position - (int) position (genomic coords) of variant,
                                       reference base(s) - (str) reference base(s) relative to forward strand,
                                       alternate base(s) - (str) alternate base(s) relative to forward strand, )
                                       Empty list adds no additional variants
          gene_bds ( tuple of ints ) - ( number of upstream bases outside of gene, number of downstream bases outside of gene)
                                       Adds N's to sequence outside of gene
                                       ( 0, 0 ) adds no N bases
          rev_strand - (bool) is the center variant on the reverse strand?
          scored_context - (int) number of bases to score on each side of the variant
          unscored_context - (int) number of flanking unscored bases on each side of the variant

    Returns: refvar_seq - (tuple of str) ( reference sequence, variant sequence )
                          Both sequences will be 2*( scored_context + unscored_context ) + 1 long if none of the variants are indels
    """

    #print( center_var )

    #print( haplotype )

    flanks = unscored_context + scored_context

    #subtracting one to adjust from 1 based to 0 based coords
    #gene bds is adding N's for locations outside the gene
    refseq = 'N'*gene_bds[ 0 ] \
             + refseq[ center_var[ 0 ] - flanks - 1 + gene_bds[ 0 ] : center_var[ 0 ] + flanks - gene_bds[ 1 ] ] \
             + 'N'*gene_bds[ 1 ]

    #print( len( refseq ) )
    assert len( refseq ) == flanks*2 + 1, 'Debug your math'

    refvar_seqpos = [ ( flanks + ( p - center_var[ 0 ] ), r, a ) for p,r,a in ref_var ]

    #start from the back so the coords stay the same despite indels
    #refvar_seqpos.sort( reverse = True )

    for pos,ref,alt in refvar_seqpos:

        assert len( ref ) == len( alt ), 'Code is not ready for indels within the reference variants yet'

        assert refseq[ pos: pos + len( ref ) ].upper() == ref, 'Reference base within reference variants does not match given position'

        refseq = refseq[ : pos ] + alt + refseq[ pos + len( ref ): ]

    hap_seqpos = [ ( flanks + ( p - center_var[ 0 ] ), r, a ) for p,r,a in haplotype + [ center_var ] ]

    #start from the back so the coords stay the same despite indels
    hap_seqpos.sort( reverse = True )

    varseq = refseq

    for pos,ref,alt in hap_seqpos:

        #print( pos, ref, alt )

        #print( refseq[ pos: pos + len( ref ) ].upper() )

        assert refseq[ pos: pos + len( ref ) ].upper() == ref, 'Reference base within haplotype does not match given position at %s:%s>%s' % ( center_var[ 0 ], ref, alt )

        varseq = varseq[ : pos ] + alt + varseq[ pos + len( ref ): ]

    if rev_strand:

        refseq = rev_complement( refseq )
        varseq = rev_complement( varseq )

    return ( refseq, varseq )

def splai_score_variants( annots_df,
                           models,
                           refvarseqs,
                           ref_name,
                           chrom,
                           center_var,
                           haplotypes,
                           scored_context = 50,
                           rev_strand = False,
                           mask_value = 0,
                         ):
    """
    Uses SpliceAI default scoring to compute SDV probabilities.

    Args: annots_df (pandas df) - columns: #NAME, CHROM, STRAND, TX_START, TX_END, EXON_START, EXON_END
                                    EXON_START, EXON_END are both comma separated strings of all exon bds
          models (list) - SpliceAI models
          refvarseqs ( list of tuples of strings ) - list of all of the reference and variant sequence pairs to score
          refname (str) - reference and/or annotation name to label the entry
          center_var (list of tuples) - center_variants to be scored:
                                ( position - (int) position (genomic coords) of center variant,
                                 reference base(s) - (str) reference base(s) relative to forward strand,
                                 alternate base(s) - (str) alternate base(s) relative to forward strand, )
          haplotypes ( list of list of tuples ) - other variants added to the variant sequences
                                        ( position - (int) position (genomic coords) of variant,
                                        reference base(s) - (str) reference base(s) relative to forward strand,
                                        alternate base(s) - (str) alternate base(s) relative to forward strand, )
                                        Empty list indicates no additional variants
          scored_context - (int) number of bases to score on each side of the variant
          rev_strand - (bool) is the center variant on the reverse strand?
          mask_value - (int) value to used to mask scores

    Returns: outdf - (pandas df) Pandas dataframe with probabilities for each type of splice site event
                                 and separate probabilities after masking
    """

    outtbl = { 'ref_name': [ ref_name ]*len( refvarseqs ),
               'chrom': [ chrom ]*len( refvarseqs ),
               'pos': [],
               'ref': [],
               'alt': [],
               'other_var': [],
               'acc_abs_chg': [],
               'don_abs_chg': [],
               'DS_AG': [],
               'DS_AL': [],
               'DS_DG': [],
               'DS_DL': [],
               'DP_AG': [],
               'DP_AL': [],
               'DP_DG': [],
               'DP_DL': [],
               'DS_max': [],
               'DS_max_type': [],
               'POS_max': [],
               'DS_AGm': [],
               'DS_ALm': [],
               'DS_DGm': [],
               'DS_DLm': [],
               'DP_AGm': [],
               'DP_ALm': [],
               'DP_DGm': [],
               'DP_DLm': [],
               'DS_maxm': [],
               'DS_maxm_type': [],
               'POS_maxm': [],
             }

    for idx, refvarseq in enumerate( refvarseqs ):

        refseq, varseq = refvarseq

        x_ref = one_hot_encode( refseq )[ None, : ]
        y_ref = np.mean( [ models[ m ].predict( x_ref ) for m in range( 5 ) ], axis=0 )

        x_var = one_hot_encode( varseq )[ None, : ]
        y_var = np.mean( [ models[ m ].predict( x_var ) for m in range( 5 ) ], axis=0 )

        #flips the results so the positions are relative to the forward strand
        if rev_strand:
                y_ref = y_ref[:, ::-1]
                y_var = y_var[:, ::-1]

        ref_acc = y_ref[0, :, 1]
        ref_don = y_ref[0, :, 2]
        var_acc = y_var[0, :, 1]
        var_don = y_var[0, :, 2]

        #transforms variants into sequence position coords
        allvars_seqpos = [ ( scored_context + ( p - center_var[ idx ][ 0 ] ), r, a )
                           for p,r,a in [ center_var[ idx ] ] + haplotypes[ idx ] ]

        var_acc, var_don = adjust_for_indels( var_acc,
                                              var_don,
                                              allvars_seqpos,
                                             )

        diff_acc = ref_acc - var_acc
        diff_don = ref_don - var_don

        outtbl[ 'pos' ].append( center_var[ idx ][ 0 ] )
        outtbl[ 'ref' ].append( center_var[ idx ][ 1 ] )
        outtbl[ 'alt' ].append( center_var[ idx ][ 2 ] )
        outtbl[ 'other_var' ].append( ';'.join( [ ':'.join( [ str( p ), '>'.join( [ r, a ] ) ] )
                                      for p,r,a in haplotypes[ idx ] ] ) )

        outtbl[ 'acc_abs_chg' ].append( sum( np.abs( diff_acc ) ) )
        outtbl[ 'don_abs_chg' ].append( sum( np.abs( diff_don ) ) )

        outtbl[ 'DS_AG' ].append( np.abs( np.min( [ 0, np.min( diff_acc ) ] ) ) )
        outtbl[ 'DS_AL' ].append( np.max( [ 0, np.max( diff_acc ) ] ) )
        outtbl[ 'DS_DG' ].append( np.abs( np.min( [ 0, np.min( diff_don ) ] ) ) )
        outtbl[ 'DS_DL' ].append( np.max( [ 0, np.max( diff_don ) ] ) )

        outtbl[ 'DP_AG' ].append( np.argmin( diff_acc ) - scored_context )
        outtbl[ 'DP_AL' ].append( np.argmax( diff_acc ) - scored_context )
        outtbl[ 'DP_DG' ].append( np.argmin( diff_don ) - scored_context )
        outtbl[ 'DP_DL' ].append( np.argmax( diff_don ) - scored_context )

        score_keys = [ 'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL' ]

        #first get the maximum probability across the difference scores
        outtbl[ 'DS_max' ].append( max( outtbl[ key ][ -1 ] for key in score_keys ) )
        #then get the type of event that represents the maximum probability
        outtbl[ 'DS_max_type' ].append( [ key for key in score_keys
                                          if outtbl[ key ][ -1 ] == outtbl[ 'DS_max' ][ -1 ] ][ 0 ] )
        #finally, get the location of the event associated with the highest difference score
        outtbl[ 'POS_max' ].append( outtbl[ outtbl[ 'DS_max_type' ][ -1 ].replace( 'DS', 'DP' ) ][ -1 ] )

        #this section changed 28 July to find max positions after masking
        exon_bds = get_allexon_dist( annots_df,
                                  chrom,
                                  center_var[ idx ][ 0 ],
                                  scored_context,
                                  rev_strand = rev_strand,
                                  )

        acc_wt = np.zeros( len( diff_acc ) )

        for acc in exon_bds[ 0 ]:
            acc_wt[ scored_context + acc ] = 1

        diff_acc_m = ( diff_acc > 0 ) * ( acc_wt ) * ( diff_acc ) \
                     + ( diff_acc < 0 ) * ( 1 - acc_wt ) * ( diff_acc )

        don_wt = np.zeros( len( diff_acc ) )

        for don in exon_bds[ 1 ]:
            don_wt[ scored_context + don ] = 1

        diff_don_m = ( diff_don > 0 ) * ( don_wt ) * ( diff_don ) \
                     + ( diff_don < 0 ) * ( 1 - don_wt ) * ( diff_don )

        outtbl[ 'DS_AGm' ].append( np.abs( np.min( [ 0, np.min( diff_acc_m ) ] ) ) )
        outtbl[ 'DS_ALm' ].append( np.max( [ 0, np.max( diff_acc_m ) ] ) )
        outtbl[ 'DS_DGm' ].append( np.abs( np.min( [ 0, np.min( diff_don_m ) ] ) ) )
        outtbl[ 'DS_DLm' ].append( np.max( [ 0, np.max( diff_don_m ) ] ) )

        outtbl[ 'DP_AGm' ].append( np.argmin( diff_acc_m ) - scored_context )
        outtbl[ 'DP_ALm' ].append( np.argmax( diff_acc_m ) - scored_context )
        outtbl[ 'DP_DGm' ].append( np.argmin( diff_don_m ) - scored_context )
        outtbl[ 'DP_DLm' ].append( np.argmax( diff_don_m ) - scored_context )

        score_keys = [ 'DS_AGm', 'DS_ALm', 'DS_DGm', 'DS_DLm' ]

        outtbl[ 'DS_maxm' ].append( max( outtbl[ key ][ -1 ] for key in score_keys ) )
        #then get the type of event that represents the maximum probability
        outtbl[ 'DS_maxm_type' ].append( [ key for key in score_keys
                                          if outtbl[ key ][ -1 ] == outtbl[ 'DS_maxm' ][ -1 ] ][ 0 ] )
        #finally, get the location of the event associated with the highest difference score
        outtbl[ 'POS_maxm' ].append( outtbl[ outtbl[ 'DS_maxm_type' ][ -1 ].replace( 'DS', 'DP' ) ][ -1 ] )

    outdf = pd.DataFrame( outtbl )

    return outdf

def adjust_for_indels( var_accp,
                       var_donp,
                       variants,
                     ):
    """
    Adjusts the length of the variant sequence probabilities when there are indels.
    Specifically, for deletions, fills deleted bases with a probability of 0 and for insertions,
    take the maximum probability across the inserted bases.

    Args:
          var_accp - (np array) acceptor probabilities for variant sequence
          var_donp - (np array) donor probabilities for variant sequence
          variants - ( list of tuples ) all center and haplotype variants on variant sequence
                     ( position - (int) distance from center variant,
                      reference base(s) - (str) reference base(s) relative to forward strand,
                      alternate base(s) - (str) alternate base(s) relative to forward strand, )

    Returns: variantp - (tuple of np arrays) ( acceptor probabilities, donor probabilities )
    """

    #make sure the variants are sorted
    variants.sort()

    for pos,ref,alt in variants:

        #if there's a deletion, fill the missing locations with 0's
        if len( ref ) > len( alt ):

            var_accp = np.concatenate( [ var_accp[ : pos + len( alt ) ],
                                       np.zeros( len( ref ) - len( alt ) ),
                                       var_accp[ pos + len( alt ): ] ] )

            var_donp = np.concatenate( [ var_donp[ : pos + len( alt ) ],
                                       np.zeros( len( ref ) - len( alt ) ),
                                       var_donp[ pos + len( alt ): ] ] )

        #if there's an insertion, fill use the maximum across the insertion as the variant prob
        #need to add 1 here since the final bd in python is exclusive..
        elif len( alt ) > len( ref ):

            var_accp = np.concatenate( [ var_accp[ : pos ],
                                       [ np.max( var_accp[ pos: pos + ( len( alt ) - len( ref ) ) + 1 ] ) ],
                                       var_accp[ pos + ( len( alt ) - len( ref ) ) + 1: ] ] )

            var_donp = np.concatenate( [ var_donp[ : pos ],
                                       [ np.max( var_donp[ pos: pos + ( len( alt ) - len( ref ) ) + 1 ] ) ],
                                       var_donp[ pos + ( len( alt ) - len( ref ) ) + 1: ] ] )

    return ( var_accp, var_donp )

def splai_score_mult_variants_onegene( annots_df,
                                      models,
                                      refseq,
                                      ref_name,
                                      chrom,
                                      center_var,
                                      haplotypes = None,
                                      ref_vars = None,
                                      mask_value = 0,
                                      scored_context = 50,
                                      unscored_context = 5000,
                                      rev_strand = False ):
    """
    Wrapper function to compute SpliceAI default probabilities across one gene.

    Args: annots_df (pandas df) - columns: #NAME, CHROM, STRAND, TX_START, TX_END, EXON_START, EXON_END
                                    EXON_START, EXON_END are both comma separated strings of all exon bds
          models (list) - SpliceAI models
          refseq (str) - fasta file for an entire chromosome
          ref_name (str) - reference and/or annotation name to label the entries
          chrom - (str) chromosome of center variant ( format should match your annots file (ie chr3 v 3) )
          center_var (list of tuples) - center_variants to be scored:
                                ( position - (int) position (genomic coords) of center variant,
                                 reference base(s) - (str) reference base(s) relative to forward strand,
                                 alternate base(s) - (str) alternate base(s) relative to forward strand, )
          haplotypes ( list of list of tuples ) - other variants added to the variant sequences
                                                  each ith list corresponds to the ith center_var
                                        ( position - (int) position (genomic coords) of variant,
                                        reference base(s) - (str) reference base(s) relative to forward strand,
                                        alternate base(s) - (str) alternate base(s) relative to forward strand, )
                                        None indicates no additional variants
          ref_vars ( list of list of tuples ) - other variants to be added to the reference AND variant sequences
                                                each ith list corresponds to the ith center_var
                                             ( position - (int) position (genomic coords) of variant,
                                             reference base(s) - (str) reference base(s) relative to forward strand,
                                             alternate base(s) - (str) alternate base(s) relative to forward strand, )
                                             None adds no additional variants
                                             NOTE: currently only substitutions can be handled - indels will raise error
          mask_value - (int) value to used to mask scores
          scored_context - (int) number of bases to score on each side of the center variant
          unscored_context - (int) number of flanking unscored bases on each side of the center variant
          rev_strand - (bool) is the gene on the reverse strand?

    Returns: outdf - (pandas df) Pandas dataframe with probabilities for each type of splice site event for each center_var
                                 and separate probabilities after masking
    """

    strand = '-' if rev_strand else '+'

    flanks = scored_context + unscored_context

    if not haplotypes:

        haplotypes = [ [] for var in center_var ]

    if not ref_vars:

        ref_vars = [ [] for var in center_var ]

    else:

        for p,r,a in ref_vars:

            ref_name += '+' + str( p ) + ':' + r + '>' + a

    assert all( len(i) == len( center_var ) for i in [ haplotypes, ref_vars ] ), \
    'Haplotypes and ref_vars input must be either missing or the same length as the center_var list'

    #creates a giant list of ( reference seq, variant seq ) tuples
    refvarseqs = [ create_input_seq( refseq,
                                     center,
                                     hapref[ 0 ],
                                     hapref[ 1 ],
                                     get_gene_bds( annots_df,
                                                   chrom,
                                                   center[ 0 ],
                                                   strand,
                                                   scored_context = scored_context,
                                                   unscored_context = unscored_context,
                                                    ),
                                     scored_context,
                                     rev_strand = rev_strand,
                                     unscored_context = unscored_context )
                  for center,hapref in zip( center_var, zip( haplotypes, ref_vars ) ) ]

    #this will fail if any of the other variants are indels...
    #maybe I should add some functionality to the splai_score_variant fn to handle this...
    outdf = splai_score_variants( annots_df,
                                  models,
                                  refvarseqs,
                                  ref_name,
                                  chrom,
                                  center_var,
                                  haplotypes,
                                  scored_context = scored_context,
                                  rev_strand = rev_strand,
                                  mask_value = mask_value
                          )

    return outdf

def score_mult_variants_multgene( annots_df,
                                  models,
                                  refseqs,
                                  ref_name,
                                  center_var,
                                  haplotypes = {},
                                  ref_vars = {},
                                  mask_value = 0,
                                  scored_context = 50,
                                  unscored_context = 5000,
                                ):
    """
    Wrapper function to compute SpliceAI default probabilities across multiple genes.

    Args: annots_df (pandas df) - columns: #NAME, CHROM, STRAND, TX_START, TX_END, EXON_START, EXON_END
                                    EXON_START, EXON_END are both comma separated strings of all exon bds
          models (list) - SpliceAI models
          refseqs (dict of str) - { chrom: fasta file for chromosome } ( format should match your annots file (ie chr3 v 3) )
          ref_name (str) - reference and/or annotation name to label the entries
          chrom - (str) chromosome of center variant ( format should match your annots file (ie chr3 v 3) )
          center_var (dict of list of tuples) - center_variants to be scored:
                            { chrom: [ ( position - (int) position (genomic coords) of center variant,
                                         reference base(s) - (str) reference base(s) relative to forward strand,
                                         alternate base(s) - (str) alternate base(s) relative to forward strand,
                                         rev_strand - (bool) is the variant on the reverse strand? ) ] }
          haplotypes ( dict of list of list of tuples ) - other variants added to the variant sequences
                                                          each ith list corresponds to the ith center_var
                                        { chrom: [ [ ( position - (int) position (genomic coords) of center variant,
                                                     reference base(s) - (str) reference base(s) relative to forward strand,
                                                     alternate base(s) - (str) alternate base(s) relative to forward strand,
                                                     rev_strand - (bool) is the variant on the reverse strand? ) ] ] }
                                        Empty dictionary indicates no additional variants
          ref_vars ( list of list of tuples ) - other variants to be added to the reference AND variant sequences
                                                each ith list corresponds to the ith center_var
                                             { chrom: [ [ ( position - (int) position (genomic coords) of center variant,
                                                          reference base(s) - (str) reference base(s) relative to forward strand,
                                                          alternate base(s) - (str) alternate base(s) relative to forward strand,
                                                          rev_strand - (bool) is the variant on the reverse strand? ) ] ] }
                                             Empty dictionary indicates no additional variants
                                             NOTE: currently only substitutions can be handled - indels will raise error
          mask_value - (int) value to used to mask scores
          scored_context - (int) number of bases to score on each side of the center variant
          unscored_context - (int) number of flanking unscored bases on each side of the center variant


    Returns: outdf - (pandas df) Pandas dataframe with probabilities for each type of splice site event for each center_var
                                 and separate probabilities after masking
    """

    outdfs = []

    for chrom in center_var.keys():

        chrseq = refseqs[ chrom ]

        assert len( chrseq ) > 0

        for_center = []
        for_indices = []
        rev_center = []
        rev_indices = []

        for i,var in enumerate( center_var[ chrom ] ):

            #have to remove the last entry in the tuple for it to fit in the other fn
            if var[ -1 ] == True:
                rev_center.append( ( var[ 0 ], var[ 1 ], var[ 2 ] ) )
                rev_indices.append( i )

            else:
                for_center.append( ( var[ 0 ], var[ 1 ], var[ 2 ] ) )
                for_indices.append( i )

        if chrom not in haplotypes:
            for_hap = None
            rev_hap = None
        else:
            for_hap = [ haplotypes[ chrom ][ i ] for i in for_indices ]
            rev_hap = [ haplotypes[ chrom ][ i ] for i in rev_indices ]

        if chrom not in ref_vars:
            for_rv = None
            rev_rv = None
        else:
            for_rv = [ ref_vars[ chrom ][ i ] for i in for_indices ]
            rev_rv = [ ref_vars[ chrom ][ i ] for i in rev_indices ]

        if len( for_center ) > 0:

            outdfs.append( score_mult_variants_onegene( annot,
                                                        models,
                                                        chrseq,
                                                        ref_name,
                                                        chrom,
                                                        for_center,
                                                        haplotypes = for_hap,
                                                        ref_vars = for_rv,
                                                        mask_value = mask_value,
                                                        scored_context = scored_context,
                                                        unscored_context = unscored_context,
                                                     )
                         )

        if len( rev_center ) > 0:

            outdfs.append( score_mult_variants_onegene( annot,
                                                        models,
                                                        chrseq,
                                                        refname,
                                                        chrom,
                                                        for_center,
                                                        haplotypes = for_hap,
                                                        ref_vars = for_rv,
                                                        mask_value = mask_value,
                                                        scored_context = scored_context,
                                                        unscored_context = unscored_context,
                                                        rev_strand = True
                                                     )
                         )

    outdf = pd.concat( outdfs, ignore_index = True )

    return outdf

def get_allexon_bds( annots_df,
                     chrom,
                     position,
                     scored_context,
                     rev_strand = False,
                    ):
    """
    Gets gdna position of all annotated acceptors and donors within range of scored_context

    Args: annots_df (pandas df) - columns: #NAME, CHROM, STRAND, TX_START, TX_END, EXON_START, EXON_END
                                    EXON_START, EXON_END are both comma separated strings of all exon bds
          chrom - (str) chromosome of center variant ( format should match your annots file (ie chr3 v 3) )
          position - (int) position (genomic coords) of center variant
          scored_context - (int) number of bases on each side to look for donors and acceptors
          rev_strand - (bool) is the variant on the reverse strand?

    Returns: allexon_bds - (list of tuples of ints)
                         [ ( acceptor1 gdna position,..., acceptorn gdna position ),
                           ( donor1 gdna position,..., donorn gdna position ) ]
                         SpliceAI default scoring would only return distance to nearest donor OR acceptor
                         get_2exon_bds returns only two values
                         This fn returns all values within the scored context range
                         The values returned are gdna coordinates and not distance to acceptor like other fns
                         These values are used for masking and getting relative jn use
    """

    ann_df = annots_df.copy()

    strand = '-' if rev_strand else '+'

    idx = ann_df.index[ ( ann_df.CHROM == chrom )
                      & ( ann_df.TX_START <= position )
                      & ( ann_df.TX_END >= position )
                      & ( ann_df.STRAND >= strand ) ]

    if len( idx ) != 1:
        print( 'The chromosome and position is not matching exactly one gene at %s:%i!' % ( chrom, position ) )

    exon_starts = ann_df.at[ idx[ 0 ], 'EXON_START' ].split( ',' )

    exon_startd = [ ( int( start ) + 1 ) - position
                            for start in exon_starts
                            if start != '' ]

    prox_starts = tuple( int( gpos ) + 1 for gpos, dist in zip( exon_starts, exon_startd )
                         if np.abs( dist ) <= scored_context )

    exon_ends = ann_df.at[ idx[ 0 ], 'EXON_END' ].split( ',' )

    exon_endd = [ int( end ) - position
                          for end in exon_ends
                          if end != '' ]

    prox_ends = tuple( int( gpos ) for gpos, dist in zip( exon_ends, exon_endd )
                         if np.abs( dist ) <= scored_context )

    exon_bds = [ prox_starts, prox_ends ] if not rev_strand else [ prox_ends, prox_starts ]

    return exon_bds

def create_rel_jn_use_tbl( pext_tbx,
                           pext_header,
                           chrom,
                           exon_bds,
                           col_name = 'mean_proportion',
                           extend_search = 100,
                           std_tol = 1e-10,
                        ):

    """
    Creates a pandas dataframe of pext base-level scores for each annotated acceptor/donor

    Args: pext_tbx (pysam tabix file) - tabixed file for pext scores
          pext_header (list of strings) - column names for each pext row
          chrom - (str) chromosome of center variant ( format should match your annots file (ie chr3 v 3) )
          exon_bds - (tuple of list of ints) position (genomic coords) of acceptors and donors within range
          col_name - (str) name of tissue column to extract as relative jn use score

    Returns: pext_df - (pandas df) columns: chrom, jn (0-based), pext
    """

    outtbl = { 'chrom': [],
               'jn': [],
               'pext': [] }

    acc, don = exon_bds.copy()

    sort_bds = sorted( acc + don )

    col_idx = pext_header.index( col_name )

    for pos in sort_bds:

        outtbl[ 'chrom' ].append( chrom )
        outtbl[ 'jn' ].append( pos )

        pext_scores = [ float( row.split( '\t' )[ col_idx ] ) for row in pext_tbx.fetch( chrom, pos - 1, pos ) ]

        #checks if the pext entry is empty
        if not pext_scores or np.isnan( pext_scores ).all():

            print( 'Pext table is empty at position %i' % pos )

            values = np.array( [ float( row.split( '\t' )[ col_idx ] )
                                 for row in pext_tbx.fetch( chrom, pos - 1 - extend_search, pos + extend_search ) ] )

            #there aren't any pext scores in the region - print a warning and use default masking values
            if np.isnan( values ).all():

                print( 'Pext table is empty at position %i after %i bp extension' % ( pos, extend_search ) )
                outtbl[ 'pext' ].append( 1 )
                #skip ahead - no need to check the standard deviation of missing values
                continue

            else:

                outtbl[ 'pext' ].append( np.nanmean( values ) )

            #checks if there is more than one mean value within range
            if np.std( values ) > std_tol:

                print( 'Standard deviation is non-zero for pext table extension at position %i' % pos )

        #checks if there are duplicate pext scores
        elif len( pext_scores ) > 1:

            print( 'Pext table has duplicate entries at position %i' % pos )

            outtbl[ 'pext' ].append( np.nanmean( pext_scores ) )

            #checks if there is more than one mean value across the duplicated rows
            if np.std( pext_scores ) > std_tol:

                print( 'Standard deviation is non-zero for duplicate pext scores at position %i' % pos )

        #there's only one pext score - just append it
        else:

            outtbl[ 'pext' ].append( pext_scores[ 0 ] )

    outdf = pd.DataFrame( outtbl )

    return outdf

def get_relative_jn_use( pext_df,
                         chrom,
                         position,
                         exon_bds, ):

    """
        Gets the relative use of each acceptor and donor within range

        Args: pext_df (pandas df) - columns: chrom, jn (0-based), pext
              chrom - (str) chromosome of center variant ( format should match your gtex file (ie chr3 v 3) )
              position - (int) position (genomic coords) of center variant
              exon_bds - (tuple of list of ints) position (genomic coords) of acceptors and donors within range

        Returns: acceptor and donor relative use for each acceptor and donor (list of dictionaries)
                 [ { acceptor1_pos: acceptor1_rel_use, ... acceptorn_pos: acceptorn_rel_use },
                   { donor1_pos: donor1_rel_use, ... donorn_pos: donorn_rel_use } ]
                 The positions are the relative distances to the center variant
    """

    pext = pext_df.copy()

    acc_gdna, don_gdna = exon_bds.copy()

    #gets jns into sequence coords
    acc_seq = [ acc_pos - position for acc_pos in acc_gdna ]
    don_seq = [ don_pos - position for don_pos in don_gdna ]

    acc_exp = { acc_spos: pext.loc[ ( pext.chrom == chrom ) & ( pext.jn == acc_gpos ) ].pext.values[ 0 ]
                    for acc_spos, acc_gpos in zip( acc_seq, acc_gdna ) }

    don_exp = { don_spos: pext.loc[ ( pext.chrom == chrom ) & ( pext.jn == don_gpos ) ].pext.values[ 0 ]
                    for don_spos, don_gpos in zip( don_seq, don_gdna ) }

    return [ acc_exp, don_exp ]

def jnuse_score_variants(  models,
                           refvarseqs,
                           ref_name,
                           ss_jn_use,
                           chrom,
                           center_var,
                           haplotypes,
                           scored_context,
                           rev_strand = False,
                 ):
    """
    Uses relative junction use to compute SDV probabilities.

    Args:
          models (list) - SpliceAI models
          refvarseqs ( list of tuples of strings ) - list of all of the reference and variant sequence pairs to score
          refname (str) - reference and/or annotation name to label the entry
          ss_jn_use (list of dicts) - acceptor and donor relative use for each acceptor and donor (list of dictionaries)
                   [ { acceptor1_pos: acceptor1_rel_use, ... acceptorn_pos: acceptorn_rel_use },
                     { donor1_pos: donor1_rel_use, ... donorn_pos: donorn_rel_use } ]
                   The values in both dictionaries should sum to 1
                   The positions are the relative distances to the center variant
          center_var (list of tuples) - center_variants to be scored:
                                ( position - (int) position (genomic coords) of center variant,
                                 reference base(s) - (str) reference base(s) relative to forward strand,
                                 alternate base(s) - (str) alternate base(s) relative to forward strand, )
          haplotypes ( list of list of tuples ) - other variants added to the variant sequences
                                        ( position - (int) position (genomic coords) of variant,
                                        reference base(s) - (str) reference base(s) relative to forward strand,
                                        alternate base(s) - (str) alternate base(s) relative to forward strand, )
                                        Empty list indicates no additional variants
          scored_context - (int) number of bases to score on each side of the variant
          rev_strand - (bool) is the center variant on the reverse strand?

    Returns: outdf - (pandas df) Pandas dataframe with probabilities for each type of splice site event
                                 and separate probabilities after masking
    """

    outtbl = { 'ref_name': [ ref_name ]*len( refvarseqs ),
               'chrom': [ chrom ]*len( refvarseqs ),
               'pos': [],
               'ref': [],
               'alt': [],
               'other_var': [],
               'AL_chg': [],
               'AG_chg': [],
               'DL_chg': [],
               'DG_chg': [],
               'DS_AGrw': [],
               'DS_ALrw': [],
               'DS_DGrw': [],
               'DS_DLrw': [],
               'DP_AGrw': [],
               'DP_ALrw': [],
               'DP_DGrw': [],
               'DP_DLrw': [],
               'DS_maxrw': [],
               'DS_maxrw_type': [],
               'POS_maxrw': [],
             }

    for idx, refvarseq in enumerate( refvarseqs ):

        #print( refvarseq )

        refseq, varseq = refvarseq

        #print( center_var[ idx ] )
        #print( len( refseq ) )
        #print( len( varseq ) )

        x_ref = one_hot_encode( refseq )[ None, : ]
        y_ref = np.mean( [ models[ m ].predict( x_ref ) for m in range( 5 ) ], axis=0 )

        x_var = one_hot_encode( varseq )[ None, : ]
        y_var = np.mean( [ models[ m ].predict( x_var ) for m in range( 5 ) ], axis=0 )

        #flips the results so the positions are relative to the forward strand
        if rev_strand:
                y_ref = y_ref[:, ::-1]
                y_var = y_var[:, ::-1]

        ref_acc = y_ref[0, :, 1]
        ref_don = y_ref[0, :, 2]
        var_acc = y_var[0, :, 1]
        var_don = y_var[0, :, 2]

        #transforms variants into sequence position coords
        allvars_seqpos = [ ( scored_context + ( p - center_var[ idx ][ 0 ] ), r, a )
                           for p,r,a in [ center_var[ idx ] ] + haplotypes[ idx ] ]

        var_acc, var_don = adjust_for_indels( var_acc,
                                              var_don,
                                              allvars_seqpos,
                                             )

        diff_acc = ref_acc - var_acc
        diff_don = ref_don - var_don

        outtbl[ 'AL_chg' ].append( sum( ( diff_acc > 0 ) * diff_acc ) )
        outtbl[ 'AG_chg' ].append( sum( ( diff_acc < 0 ) * diff_acc ) )
        outtbl[ 'DL_chg' ].append( sum( ( diff_don > 0 ) * diff_don ) )
        outtbl[ 'DG_chg' ].append( sum( ( diff_don < 0 ) * diff_don ) )

        acc_jn_use, don_jn_use = ss_jn_use[ idx ]

        acc_wt = np.zeros( len( diff_acc ) )

        for acc_jn in acc_jn_use.keys():

            acc_wt[ scored_context + acc_jn ] = acc_jn_use[ acc_jn ]

        diff_acc = ( diff_acc > 0 ) * ( acc_wt ) * ( diff_acc ) \
                     + ( diff_acc < 0 ) * ( 1 - acc_wt ) * ( diff_acc )

        don_wt = np.zeros( len( diff_acc ) )

        for don_jn in don_jn_use.keys():

            don_wt[ scored_context + don_jn ] = don_jn_use[ don_jn ]

        diff_don = ( diff_don > 0 ) * ( don_wt ) * ( diff_don ) \
                     + ( diff_don < 0 ) * ( 1 - don_wt ) * ( diff_don )

        outtbl[ 'pos' ].append( center_var[ idx ][ 0 ] )
        outtbl[ 'ref' ].append( center_var[ idx ][ 1 ] )
        outtbl[ 'alt' ].append( center_var[ idx ][ 2 ] )
        outtbl[ 'other_var' ].append( ';'.join( [ ':'.join( [ str( p ), '>'.join( [ r, a ] ) ] )
                                      for p,r,a in haplotypes[ idx ] ] ) )

        outtbl[ 'DS_AGrw' ].append( np.abs( np.min( [ 0, np.min( diff_acc ) ] ) ) )
        outtbl[ 'DS_ALrw' ].append( np.max( [ 0, np.max( diff_acc ) ] ) )
        outtbl[ 'DS_DGrw' ].append( np.abs( np.min( [ 0, np.min( diff_don ) ] ) ) )
        outtbl[ 'DS_DLrw' ].append( np.max( [ 0, np.max( diff_don ) ] ) )

        outtbl[ 'DP_AGrw' ].append( np.argmin( diff_acc ) - scored_context )
        outtbl[ 'DP_ALrw' ].append( np.argmax( diff_acc ) - scored_context )
        outtbl[ 'DP_DGrw' ].append( np.argmin( diff_don ) - scored_context )
        outtbl[ 'DP_DLrw' ].append( np.argmax( diff_don ) - scored_context )

        score_keys = [ 'DS_AGrw', 'DS_ALrw', 'DS_DGrw', 'DS_DLrw' ]

        #print( outtbl )
        #first get the maximum probability across the difference scores
        outtbl[ 'DS_maxrw' ].append( max( outtbl[ key ][ -1 ] for key in score_keys ) )
        #then get the type of event that represents the maximum probability
        outtbl[ 'DS_maxrw_type' ].append( [ key for key in score_keys
                                          if outtbl[ key ][ -1 ] == outtbl[ 'DS_maxrw' ][ -1 ] ][ 0 ] )
        #finally, get the location of the event associated with the highest difference score
        outtbl[ 'POS_maxrw' ].append( outtbl[ outtbl[ 'DS_maxrw_type' ][ -1 ].replace( 'DS', 'DP' ) ][ -1 ] )

    outdf = pd.DataFrame( outtbl )

    return outdf

def custom_score_mult_variants_oneexon( annots_df,
                                        pext_tbx,
                                        pext_header,
                                        models,
                                        refseq,
                                        ref_name,
                                        exon_cds,
                                        chrom,
                                        center_var,
                                        haplotypes = None,
                                        ref_vars = None,
                                        scored_context_pad = 10,
                                        unscored_context = 5000,
                                        rev_strand = False,
                                        pext_col = 'mean_proportion',
                                        extend_pext_search = 200,
                                        pext_std_tol = 1e-10, ):
    """
    Wrapper function to compute junction use weighted probabilities across one exon.

    Args: annots_df (pandas df) - columns: #NAME, CHROM, STRAND, TX_START, TX_END, EXON_START, EXON_END
                                    EXON_START, EXON_END are both comma separated strings of all exon bds
          pext_tbx (pysam tabix file) - tabixed file for pext scores
          pext_header (list of strings) - column names for each pext row
          models (list) - SpliceAI models
          refseq (str) - fasta file for an entire chromosome
          exon_cds (tuple of ints) - exon bds in genomic coords to determined scored context length
          ref_name (str) - reference and/or annotation name to label the entries
          chrom - (str) chromosome of center variant ( format should match your annots file (ie chr3 v 3) )
          center_var (list of tuples) - center_variants to be scored:
                                ( position - (int) position (genomic coords) of center variant,
                                 reference base(s) - (str) reference base(s) relative to forward strand,
                                 alternate base(s) - (str) alternate base(s) relative to forward strand, )
          haplotypes ( list of list of tuples ) - other variants added to the variant sequences
                                                  each ith list corresponds to the ith center_var
                                        ( position - (int) position (genomic coords) of variant,
                                        reference base(s) - (str) reference base(s) relative to forward strand,
                                        alternate base(s) - (str) alternate base(s) relative to forward strand, )
                                        None indicates no additional variants
          ref_vars ( list of list of tuples ) - other variants to be added to the reference AND variant sequences
                                                each ith list corresponds to the ith center_var
                                             ( position - (int) position (genomic coords) of variant,
                                             reference base(s) - (str) reference base(s) relative to forward strand,
                                             alternate base(s) - (str) alternate base(s) relative to forward strand, )
                                             None adds no additional variants
                                             NOTE: currently only substitutions can be handled - indels will raise error
          scored_context_pad - (int) number of additional bases above exon length to score on each side of the center variant
          unscored_context - (int) number of flanking unscored bases on each side of the center variant
          rev_strand - (bool) is the exon on the reverse strand?
          pext_col - (str) name of tissue column to use for pext scores

    Returns: outdf - (pandas df) Pandas dataframe with probabilities for each type of splice site event for each center_var
                                 and separate probabilities after masking
    """

    strand = '-' if rev_strand else '+'

    exon_len = exon_cds[ 1 ] - exon_cds[ 0 ]

    scored_context = exon_len + scored_context_pad

    flanks = scored_context + unscored_context

    if not haplotypes:

        haplotypes = [ [] for var in center_var ]

    if not ref_vars:

        ref_vars = [ [] for var in center_var ]

    else:

        for ref_var in ref_vars:

            for p,r,a in ref_var:

                ref_name += '+' + str( p ) + ':' + r + '>' + a

    assert all( len(i) == len( center_var ) for i in [ haplotypes, ref_vars ] ), \
    'Haplotypes and ref_vars input must be either missing or the same length as the center_var list'

    #creates a giant list of ( reference seq, variant seq ) tuples
    refvarseqs = [ create_input_seq( refseq,
                                     center,
                                     hapref[ 0 ],
                                     hapref[ 1 ],
                                     get_gene_bds( annots_df,
                                                   chrom,
                                                   center[ 0 ],
                                                   strand,
                                                   scored_context,
                                                   unscored_context = unscored_context ),
                                     scored_context,
                                     rev_strand = rev_strand,
                                     unscored_context = unscored_context )
                  for center,hapref in zip( center_var, zip( haplotypes, ref_vars ) ) ]

    #creates a giant list of lists of relative acceptor/donor use
    rel_jn_use = [ get_relative_jn_use( create_rel_jn_use_tbl( pext_tbx,
                                                               pext_header,
                                                               chrom,
                                                               get_allexon_bds( annots_df,
                                                                                chrom,
                                                                                center[ 0 ],
                                                                                scored_context,
                                                                                rev_strand = rev_strand
                                                                                ),
                                                               col_name = pext_col,
                                                               extend_search = extend_pext_search,
                                                               std_tol = pext_std_tol, ),
                                        chrom,
                                        center[ 0 ],
                                        get_allexon_bds( annots_df,
                                                         chrom,
                                                         center[ 0 ],
                                                         scored_context,
                                                         rev_strand = rev_strand
                                                        ), )
                  for center in center_var ]

    #this will fail if any of the other variants are indels...
    #maybe I should add some functionality to the score_variant fn to handle this...
    outdf = jnuse_score_variants(  models,
                                    refvarseqs,
                                    ref_name,
                                    rel_jn_use,
                                    chrom,
                                    center_var,
                                    haplotypes,
                                    scored_context,
                                    rev_strand = rev_strand,
                                  )

    return outdf

def custom_score_indiv_variants_multexon( tbl_byvar,
                                          annots_df,
                                          pext_tbx,
                                          pext_header,
                                          models,
                                          refseq_files,
                                          ref_name,
                                          scored_context_pad = 10,
                                          unscored_context = 5000,
                                          rev_strand = False,
                                          pext_col = 'mean_proportion',
                                          extend_pext_search = 100,
                                          pext_std_tol = 1e-10, ):

    tbv = tbl_byvar.sort_values( by = [ 'chrom', 'pos' ] ).copy()

    outtbls = []

    for chrom in tbv.chrom.unique().tolist():

        print( chrom )

        chr_seq = pp.get_refseq( refseq_files + chrom + '.fa' )[ 0 ]

        tbv_chrom = tbv.loc[ tbv.chrom == chrom ].copy()
        tbv_pos = tbv_chrom.loc[ tbv.strand == '+' ]
        tbv_neg = tbv_chrom.loc[ tbv.strand == '-' ]

        pos_snvs = [ ( pos, refalt[ 0 ], refalt[ 1 ] )
                     for pos, refalt in zip( tbv_pos.pos, zip( tbv_pos.ref, tbv_pos.alt ) ) ]

        if pos_snvs:

            pos_df = pd.concat( [ custom_score_mult_variants_oneexon( annots_df,
                                                                      pext_tbx,
                                                                      pext_header,
                                                                      models,
                                                                      chr_seq,
                                                                      'HG19',
                                                                      ex_cd,
                                                                      chrom,
                                                                      [ pos_snvs[ i ] ],
                                                                      scored_context_pad = scored_context_pad,
                                                                      unscored_context = unscored_context,
                                                                      pext_col = pext_col,
                                                                      extend_pext_search = extend_pext_search,
                                                                      pext_std_tol = pext_std_tol )
                                  for i, ex_cd in enumerate( tbv_pos.exon_cds.tolist() ) ],
                              ignore_index = True )

        neg_snvs = [ ( pos, refalt[ 0 ], refalt[ 1 ] )
                     for pos, refalt in zip( tbv_neg.pos, zip( tbv_neg.ref, tbv_neg.alt ) ) ]

        if neg_snvs:

            neg_df = pd.concat( [ custom_score_mult_variants_oneexon( annots_df,
                                                                      pext_tbx,
                                                                      pext_header,
                                                                      models,
                                                                      chr_seq,
                                                                      'HG19',
                                                                      ex_cd,
                                                                      chrom,
                                                                      [ neg_snvs[ i ] ],
                                                                      rev_strand = True,
                                                                      scored_context_pad = scored_context_pad,
                                                                      unscored_context = unscored_context,
                                                                      pext_col = pext_col,
                                                                      extend_pext_search = extend_pext_search,
                                                                      pext_std_tol = pext_std_tol )
                                    for i, ex_cd in enumerate( tbv_neg.exon_cds.tolist() ) ],
                                ignore_index = True )

        if pos_snvs and neg_snvs:

            outtbls.append( pd.concat( [ pos_df, neg_df ],
                                         ignore_index = True ) )

        elif pos_snvs:

            outtbls.append( pos_df )

        else:

            outtbls.append( neg_df )

    outdf = pd.concat( outtbls,
                       ignore_index = True )

    return outdf

def nearest_exon_cds_bed( tbl_byvar,
                          var_bed,
                          exon_bed ):

    tbv = tbl_byvar.copy()

    cl = var_bed.closest( exon_bed, s = True, t = 'first', d = True )

    closest = pd.read_table( cl.fn,
                             names = [ 'chrom', 'var_start', 'pos', 'var_name', 'var_score', 'strand',
                                       'ann_chrom', 'ann_start', 'ann_end', 'ann_name', 'ann_score', 'ann_strand',
                                       'dist' ] )

    closest = closest[ [ 'chrom', 'pos', 'strand', 'ann_chrom', 'ann_start', 'ann_end', 'ann_name', 'dist' ] ]

    tbv = tbv.set_index( [ 'chrom', 'pos', 'strand' ] ).merge( closest.set_index( [ 'chrom', 'pos', 'strand' ] ),
                                                               left_index = True,
                                                               right_index = True ).reset_index()

    return tbv

def create_exon_table( exon_dict,
                       annot,
                       gene_name,
                       UTR5_len = 0,
                       rev_strand = False ):

    outtbl = { 'num': [],
               'len': [],
               'gdna_start': [],
               'gdna_end': [],
               'seq': [] }

    for exon, seq in exon_dict.items():

        outtbl[ 'num' ].append( exon )
        outtbl[ 'len' ].append( len( seq ) )
        outtbl[ 'gdna_start' ].append( int( annot.loc[ annot[ '#NAME' ] == gene_name ].EXON_START.str.split( ',' ).tolist()[ 0 ][ exon - 1 ] )
                                       if not rev_strand else
                                       int( annot.loc[ annot[ '#NAME' ] == gene_name ].EXON_START.str.split( ',' ).tolist()[ 0 ][ -exon ] ) )
        outtbl[ 'gdna_end' ].append( int( annot.loc[ annot[ '#NAME' ] == gene_name ].EXON_END.str.split( ',' ).tolist()[ 0 ][ exon - 1 ] )
                                     if not rev_strand else
                                     int( annot.loc[ annot[ '#NAME' ] == gene_name ].EXON_END.str.split( ',' ).tolist()[ 0 ][ -exon ] ) )
        outtbl[ 'seq' ].append( seq if not rev_strand else rev_complement( seq ))

    if sum( outtbl[ 'len' ] ) % 3 != 0:
        print( 'Your exonic sequence is not divisible by 3 - did you make a mistake?' )

    outdf = pd.DataFrame( outtbl )

    outdf.sort_values( by = 'num' )

    outdf[ 'ccds_start' ] = outdf.len.cumsum() - outdf.len

    if not rev_strand:

        outdf.at[ 0, 'gdna_start' ] += UTR5_len

    else:

        outdf.at[ 0, 'gdna_end' ] -= UTR5_len

    return outdf

def amino_acid_subs( exon_tbl,
                     amino_seq,
                     dna_to_amino,
                     amino_to_dna,
                     scored_context,
                     unscored_context = 5000,
                     rev_strand = False ):

    exons = exon_tbl.copy()

    flanks = scored_context + unscored_context

    ccds_seq = ''.join( exons.seq.tolist() )

    #account for 0 based numbering
    gstart = exon_tbl.gdna_start.to_numpy() + 1 if not rev_strand else exon_tbl.gdna_end.to_numpy()
    cstart = exon_tbl.ccds_start.to_numpy()

    center_vars = []
    haplotypes = []
    ref_aa = []
    alt_aa = []
    aa_num = []

    for i in range( 0, len( ccds_seq ), 3):

        codon = ccds_seq[ i: i + 3 ].upper()

        assert amino_seq[ int( i / 3 ) ] == dna_to_amino[ codon ], 'Your amino acid sequence does not match the coding sequence'

        for res, sub_codons in amino_to_dna.items():

            for sub_cod in sub_codons:

                #if the current codon matches the substitution codon, move on
                if sub_cod == codon:
                    continue

                #first lets locate the first mismatch and make that the center var
                for j in range( 3 ):

                    if sub_cod[ j ] != codon[ j ]:

                        #gets current position starting from 0
                        cur_ccds = i + j

                        #figure out which exon you are at
                        #then take the gdna positon of the start
                        #and add however many bases you are past that
                        gpos = gstart[ cstart <= cur_ccds ].max() + ( cur_ccds - cstart[ cstart <= cur_ccds ].max() ) if not rev_strand \
                               else gstart[ cstart <= cur_ccds ].min() - ( cur_ccds - cstart[ cstart <= cur_ccds ].max() )

                        center_vars.append( ( gpos, codon[ j ], sub_cod[ j ] )
                                            if not rev_strand else
                                            ( gpos, rev_complement( codon[ j ] ), rev_complement( sub_cod[ j ] ) ) )

                        ref_aa.append( dna_to_amino[ codon ] )
                        alt_aa.append( dna_to_amino[ sub_cod ] )
                        aa_num.append( int( i / 3 ) + 1 )

                        #we only want one center variant so lets exit the loop
                        break

                haplotypes.append( [] )

                #if the first mismatch was at the final base, ext
                if j == 2:

                    continue

                for k in range( j + 1, 3 ):

                    if sub_cod[ k ] != codon[ k ]:

                        #gets current position starting from 0
                        cur_ccds = i + k

                        gpos = gstart[ cstart <= cur_ccds ].max() + ( cur_ccds - cstart[ cstart <= cur_ccds ].max() ) if not rev_strand \
                               else gstart[ cstart <= cur_ccds ].min() - ( cur_ccds - cstart[ cstart <= cur_ccds ].max() )

                        #print( gpos )

                        #we can't have variants outside the scored sequence
                        #this is for rare, out of frame aminos spanning two exons
                        if gpos < ( center_vars[ -1 ][ 0 ] + flanks ):

                            #for this one go all the way through k
                            haplotypes[ -1 ].append( ( gpos, codon[ k ], sub_cod[ k ] )
                                                     if not rev_strand else
                                                     ( gpos, rev_complement( codon[ k ] ), rev_complement( sub_cod[ k ] ) ) )

                        else:

                            haplotypes[ -1 ].append( 'skip' )

    return center_vars, haplotypes, ref_aa, alt_aa, aa_num


def compute_all_prob( models,
                      refvarseq,
                      ss_jn_use,
                      center_var,
                      haplotype,
                      scored_context,
                      rev_strand = False,
                 ):

    outd = {}

    refseq, varseq = refvarseq

    x_ref = one_hot_encode( refseq )[ None, : ]
    y_ref = np.mean( [ models[ m ].predict( x_ref ) for m in range( 5 ) ], axis=0 )

    x_var = one_hot_encode( varseq )[ None, : ]
    y_var = np.mean( [ models[ m ].predict( x_var ) for m in range( 5 ) ], axis=0 )

    #flips the results so the positions are relative to the forward strand
    if rev_strand:
            y_ref = y_ref[:, ::-1]
            y_var = y_var[:, ::-1]

    outd[ 'ref_acc' ] = y_ref[0, :, 1]
    outd[ 'ref_don' ] = y_ref[0, :, 2]
    outd[ 'var_acc' ] = y_var[0, :, 1]
    outd[ 'var_don' ] = y_var[0, :, 2]

    #transforms variants into sequence position coords
    allvars_seqpos = [ ( scored_context + ( p - center_var[ 0 ] ), r, a )
                           for p,r,a in [ center_var ] + haplotype ]

    outd[ 'var_acc' ], outd[ 'var_don' ] = adjust_for_indels( outd[ 'var_acc' ],
                                                                  outd[ 'var_don' ],
                                                                  allvars_seqpos,
                                                                )

    outd[ 'diff_acc' ] = outd[ 'ref_acc' ] - outd[ 'var_acc' ]
    outd[ 'diff_don' ] = outd[ 'ref_don' ] - outd[ 'var_don' ]

    acc_jn_use, don_jn_use = ss_jn_use

    acc_wtm = np.zeros( len( outd[ 'diff_acc' ] ) )
    acc_wtrw = np.zeros( len( outd[ 'diff_acc' ] ) )

    for acc_jn in acc_jn_use.keys():

        seq_jn = scored_context + acc_jn

        acc_wtm[ seq_jn ] = 1
        acc_wtrw[ seq_jn ] = acc_jn_use[ acc_jn ]

    outd[ 'diff_accm' ] = ( outd[ 'diff_acc' ] > 0 ) * ( acc_wtm ) * ( outd[ 'diff_acc' ] ) \
                          + ( outd[ 'diff_acc' ] < 0 ) * ( 1 - acc_wtm ) * ( outd[ 'diff_acc' ] )

    outd[ 'diff_accrw' ] = ( outd[ 'diff_acc' ] > 0 ) * ( acc_wtrw ) * ( outd[ 'diff_acc' ] ) \
                          + ( outd[ 'diff_acc' ] < 0 ) * ( 1 - acc_wtrw ) * ( outd[ 'diff_acc' ] )

    don_wtm = np.zeros( len( outd[ 'diff_acc' ] ) )
    don_wtrw = np.zeros( len( outd[ 'diff_acc' ] ) )

    for don_jn in don_jn_use.keys():

        seq_jn = scored_context + don_jn

        don_wtm[ seq_jn ] = 1
        don_wtrw[ seq_jn ] = don_jn_use[ don_jn ]

    outd[ 'diff_donm' ] = ( outd[ 'diff_don' ] > 0 ) * ( don_wtm ) * ( outd[ 'diff_don' ] ) \
                          + ( outd[ 'diff_don' ] < 0 ) * ( 1 - don_wtm ) * ( outd[ 'diff_don' ] )

    outd[ 'diff_donrw' ] = ( outd[ 'diff_don' ] > 0 ) * ( don_wtrw ) * ( outd[ 'diff_don' ] ) \
                          + ( outd[ 'diff_don' ] < 0 ) * ( 1 - don_wtrw ) * ( outd[ 'diff_don' ] )

    return outd

def plot_prob_by_pos( probs_in,
                      center_pos,
                      key_names,
                      colors,
                      rev_strand = False,
                      fig_size = ( 15, 4 ),
                      title = '',
                      x_label = 'Position',
                      y_label = None,
                      y_lim = None
                      ):

    probs = probs_in.copy()

    plt.figure( figsize = fig_size )

    scored_context = int( len( probs[ key_names[ 0 ] ] ) / 2 )

    pos_array = list( np.arange( center_pos - scored_context,
                           center_pos + scored_context + 1 ) )

    for line, col in zip( key_names, colors ):

        plt.scatter( pos_array,
                  probs[ line ],
                  color = col )

    if rev_strand:
        plt.gca().invert_xaxis()

    plt.xticks( fontsize = 18, rotation = 'vertical' )

    plt.xlabel( x_label, fontsize = 18 )

    plt.yticks( fontsize = 18 )

    if y_label:
        plt.ylabel( y_label, fontsize = 18 )

    if y_lim:
        plt.ylim( y_lim )

    plt.title( title, fontsize = 16 )

    plt.show()

def plot_allprobs_by_pos( annots_df,
                          pext_tbx,
                          pext_header,
                          models,
                          refseq,
                          exon_cds,
                          chrom,
                          center_var,
                          haplotype = [],
                          ref_var = [],
                          scored_context_pad = 10,
                          unscored_context = 5000,
                          rev_strand = False,
                          fig_size = ( 15, 4 ),
                          sharey = True,
                          pext_col = 'mean_proportion',
                          extend_pext_search = 200,
                          pext_std_tol = 1e-10, ):

    exon_len = exon_cds[ 1 ] - exon_cds[ 0 ]

    strand = '-' if rev_strand else '+'

    scored_context = exon_len + scored_context_pad

    flanks = scored_context + unscored_context

    #creates a giant list of ( reference seq, variant seq ) tuples
    refvarseq = create_input_seq( refseq,
                                  center_var,
                                  haplotype,
                                  ref_var,
                                  get_gene_bds( annots_df,
                                                chrom,
                                                center_var[ 0 ],
                                                strand,
                                                scored_context,
                                                unscored_context = unscored_context ),
                                  scored_context,
                                  rev_strand = rev_strand,
                                  unscored_context = unscored_context )

    #creates a giant list of lists of relative acceptor/donor use
    rel_jn_use = get_relative_jn_use( create_rel_jn_use_tbl( pext_tbx,
                                                                 pext_header,
                                                                 chrom,
                                                                 get_allexon_bds( annots_df,
                                                                                  chrom,
                                                                                  center_var[ 0 ],
                                                                                  scored_context,
                                                                                  rev_strand = rev_strand
                                                                                ),
                                                              col_name = pext_col,
                                                               extend_search = extend_pext_search,
                                                               std_tol = pext_std_tol ),
                                          chrom,
                                          center_var[ 0 ],
                                          get_allexon_bds( annots_df,
                                                               chrom,
                                                               center_var[ 0 ],
                                                               scored_context,
                                                               rev_strand = rev_strand
                                                         ) )

    plotd = compute_all_prob( models,
                              refvarseq,
                              rel_jn_use,
                              center_var,
                              haplotype,
                              scored_context,
                              rev_strand = rev_strand,
                            )

    plot_prob_by_pos( plotd,
                      center_var[ 0 ],
                      [ 'ref_acc', 'var_acc' ],
                      [ 'black', 'magenta' ],
                      rev_strand = rev_strand,
                      fig_size = fig_size,
                      title = 'Acceptor probabilities for WT (black) and MUT (pink)',
                      x_label = 'Position',
                      y_label = 'Acceptor probabilities',
                      y_lim = ( 0, 1 ) if sharey else None
                      )

    plot_prob_by_pos( plotd,
                      center_var[ 0 ],
                      [ 'ref_don', 'var_don' ],
                      [ 'black', 'magenta' ],
                      rev_strand = rev_strand,
                      fig_size = fig_size,
                      title = 'Donor probabilities for WT (black) and MUT (pink)',
                      x_label = 'Position',
                      y_label = 'Donor probabilities',
                      y_lim = ( 0, 1 ) if sharey else None
                      )

    plot_prob_by_pos( plotd,
                      center_var[ 0 ],
                      [ 'diff_acc', 'diff_accm', 'diff_accrw', ],
                      [ 'black', 'red', 'orange' ],
                      rev_strand = rev_strand,
                      fig_size = fig_size,
                      title = 'Acceptor difference scores:\nRAW (black), MASKED (red), and REWEIGHTED (orange)',
                      x_label = 'Position',
                      y_label = 'Acceptor difference scores\n( REF - MUT )',
                      y_lim = ( -1, 1 ) if sharey else None
                    )

    plot_prob_by_pos( plotd,
                      center_var[ 0 ],
                      [ 'diff_don', 'diff_donm', 'diff_donrw',  ],
                      [ 'black', 'red', 'orange' ],
                      rev_strand = rev_strand,
                      fig_size = fig_size,
                      title = 'Donor difference scores:\nRAW (black), MASKED (red), and REWEIGHTED (orange)',
                      x_label = 'Position',
                      y_label = 'Donor difference scores\n( REF - MUT )',
                      y_lim = ( -1, 1 ) if sharey else None
                    )

def compare_annots( ucsc_annot,
                    ucsc_gene,
                    gencode_annot,
                    gencode_gene,
                    gencode_transcript ):

    ucsc = ucsc_annot.loc[ ucsc_annot[ '#NAME' ] == ucsc_gene ].copy()

    gencode = gencode_annot.copy()

    gencode[ 'gene_id' ] = [ a.split( ' ' )[ 1 ].replace( '"', '' )
                             for att in gencode.attribute
                             for a in att.split( '; ' )
                             if a.split( ' ' )[ 0 ] == 'gene_id' ]

    gencode = gencode.loc[ gencode.gene_id == gencode_gene ].copy()

    gencode.loc[ gencode.feature != 'gene', 'transcript_id' ] = [ a.split( ' ' )[ 1 ].replace( '"', '' )
                                                                  for att in gencode.attribute
                                                                  for a in att.split( '; ' )
                                                                  if a.split( ' ' )[ 0 ] == 'transcript_id' ]

    gencode = gencode.loc[ gencode.transcript_id == gencode_transcript ].copy()

    gencode[ 'start' ] -= 1

    if int( gencode.loc[ gencode.feature == 'transcript' ].start ) == int( ucsc.TX_START ) and \
       int( gencode.loc[ gencode.feature == 'transcript' ].end ) == int( ucsc.TX_END ):

        print( 'Transcription sites match' )

    else:

        print( 'Mismatched transcription sites for gencode transcript %s' % gencode_transcript )
        print( 'UCSC: %i to %i' % ( int( ucsc.TX_START ), int( ucsc.TX_END ) ) )
        print( 'GENCODE: %i to %i' % ( int( gencode.loc[ gencode.feature == 'transcript' ].start ),
                                    int( gencode.loc[ gencode.feature == 'transcript' ].end ) ) )

    gen_ex_start = gencode.loc[ gencode.feature == 'exon' ].start.tolist()
    gen_ex_end = gencode.loc[ gencode.feature == 'exon' ].end.tolist()

    ucsc_ex_start = [ int( s ) for s in ucsc.EXON_START.tolist()[ 0 ].split( ',' ) ]
    ucsc_ex_end = [ int( s ) for s in ucsc.EXON_END.tolist()[ 0 ].split( ',' ) ]

    #print( set( gen_ex_start ).difference( set( ucsc_ex_start ) ) )

    if set( gen_ex_start ).difference( set( ucsc_ex_start ) ) or \
       set( ucsc_ex_start ).difference( set( gen_ex_start ) ):

        print( 'Mismatched exon start sites' )
        print( 'In UCSC but not GENCODE', set( ucsc_ex_start ).difference( set( gen_ex_start ) ) )
        print( 'In GENCODE but not UCSC', set( gen_ex_start ).difference( set( ucsc_ex_start ) ) )

    else:

        print( 'Exon starts match' )

    if set( gen_ex_end ).difference( set( ucsc_ex_end ) ) or \
       set( ucsc_ex_end ).difference( set( gen_ex_end ) ):

        print( 'Mismatched exon end sites' )
        print( 'In UCSC but not GENCODE', set( ucsc_ex_end ).difference( set( gen_ex_end ) ) )
        print( 'In GENCODE but not UCSC', set( gen_ex_end ).difference( set( ucsc_ex_end ) ) )

    else:

        print( 'Exon ends match' )

    return ucsc, gencode

def gtf_to_splai_annot( gtf_file ):

    gtf = gtf_file.copy()

    outtbl = { '#NAME': [],
               'CHROM': [],
               'STRAND': [],
               'TX_START': [],
               'TX_END': [],
               'EXON_START': [],
               'EXON_END': [],
               'TRANSCRIPT_ID': [] }

    gtf[ 'gene_id' ] = [ a.split( ' ' )[ 1 ].replace( '"', '' )
                         for att in gtf.attribute
                         for a in att.split( '; ' )
                         if a.split( ' ' )[ 0 ] == 'gene_id' ]

    gtf.loc[ gtf.feature != 'gene', 'transcript_id' ] = [ a.split( ' ' )[ 1 ].replace( '"', '' )
                                                              for att in gtf.attribute
                                                              for a in att.split( '; ' )
                                                              if a.split( ' ' )[ 0 ] == 'transcript_id' ]

    gtf[ 'transcript_id' ] = gtf[ 'transcript_id' ].fillna( '' )

    gtf[ 'start' ] -= 1

    for tran in list( set( gtf.transcript_id ) ):

        #gene rows are missing the transcript id - we want to skip these obviously
        if not tran:
            continue

        trans = gtf.loc[ gtf.transcript_id == tran ].copy()

        assert len( trans.gene_id.unique() ) == 1, 'Transcript %s is mapping to more than one gene' % tran

        outtbl[ '#NAME' ].append( trans.gene_id.unique()[ 0 ] )
        #this is also removing the 'chr' at the begginning of each chromosome
        outtbl[ 'CHROM' ].append( trans.chrom.unique()[ 0 ][ 3: ] )
        outtbl[ 'STRAND' ].append( trans.strand.unique()[ 0 ] )

        tx_tbl = trans.loc[ ( trans.feature == 'transcript' ) ].copy()
        outtbl[ 'TX_START' ].append( int( tx_tbl.start ) )
        outtbl[ 'TX_END' ].append( int( tx_tbl.end ) )

        ex_tbl = trans.loc[ trans.feature == 'exon' ].copy()
        outtbl[ 'EXON_START' ].append( ','.join( ex_tbl.start.apply( str ).tolist() ) )
        outtbl[ 'EXON_END' ].append( ','.join( ex_tbl.end.apply( str ).tolist() ) )

        outtbl[ 'TRANSCRIPT_ID' ].append( tran )

    outdf = pd.DataFrame( outtbl )

    return outdf

def gene_specific_gtf( gtf_file,
                       gene_name,
                       transcript_name ):

    gtf = gtf_file.copy()

    gtf[ 'gene_id' ] = [ a.split( ' ' )[ 1 ].replace( '"', '' )
                         for att in gtf.attribute
                         for a in att.split( '; ' )
                         if a.split( ' ' )[ 0 ] == 'gene_id' ]

    gene = gtf.loc[ gtf.gene_id.str.startswith( gene_name ) ].copy()

    gene.loc[ gtf.feature != 'gene', 'transcript_id' ] = [ a.split( ' ' )[ 1 ].replace( '"', '' )
                                                              for att in gene.attribute
                                                              for a in att.split( '; ' )
                                                              if a.split( ' ' )[ 0 ] == 'transcript_id' ]

    gene.transcript_id = gene.transcript_id.fillna( '' )

    assert transcript_name in gene.transcript_id.apply( lambda x: x.split( '.' )[ 0 ] ).unique().tolist(), \
    'Transcript %s not in table - only transcript %s' % ( transcript_name, gene.loc[ gene.transcript_id != '' ].transcript_id.unique().tolist() )

    #if there's more than one transcript for the gene just grab those rows
    if len( gene.loc[ gene.transcript_id != '' ].transcript_id.unique().tolist() ) > 1:
        trans = gene.loc[ ( gene.transcript_id == '' ) | ( gene.transcript_id.str.startswith( transcript_name ) ) ].copy()

    else:
        trans = gene.copy()

    trans = trans.drop( columns = [ 'gene_id', 'transcript_id' ] )

    return trans

def splai_score_wt_onegene( annots_df,
                            models,
                            refseq,
                            ref_name,
                            chrom,
                            pos_l,
                            scored_context = 50,
                            unscored_context = 5000,
                            rev_strand = False ):

    strand = '-' if rev_strand else '+'

    flanks = scored_context + unscored_context

    centers = [ ( p, refseq[ p - 1 ].upper(), 'X' ) for p in pos_l ]

    #creates a giant list of ( reference seq, variant seq ) tuples
    refvarseqs = [ create_input_seq( refseq,
                                         center,
                                         [],
                                         [],
                                         get_gene_bds( annots_df,
                                                           chrom,
                                                           center[ 0 ],
                                                           strand,
                                                           scored_context = scored_context,
                                                           unscored_context = unscored_context,
                                                        ),
                                         scored_context,
                                         rev_strand = rev_strand,
                                         unscored_context = unscored_context )
                    for center in centers ]

    #this will fail if any of the other variants are indels...
    #maybe I should add some functionality to the splai_score_variant fn to handle this...
    outdf = splai_score_wt( annots_df,
                                  models,
                                  refvarseqs,
                                  ref_name,
                                  chrom,
                                  centers,
                                  scored_context = scored_context,
                                  rev_strand = rev_strand,
                          )

    return outdf

def splai_score_wt( annots_df,
                    models,
                    refvarseqs,
                    ref_name,
                    chrom,
                    centers,
                    scored_context = 50,
                    rev_strand = False,
                ):

    outtbl = { 'ref_name': [ ref_name ]*len( refvarseqs ),
               'chrom': [ chrom ]*len( refvarseqs ),
               'pos': [],
               'ref': [],
               'wt_acc_pr': [],
               'wt_don_pr': [],
             }

    for idx, refvarseq in enumerate( refvarseqs ):

        refseq, _ = refvarseq

        x_ref = one_hot_encode( refseq )[ None, : ]
        y_ref = np.mean( [ models[ m ].predict( x_ref ) for m in range( 5 ) ], axis=0 )

        #flips the results so the positions are relative to the forward strand
        if rev_strand:
                y_ref = y_ref[:, ::-1]

        ref_acc = y_ref[0, :, 1]
        ref_don = y_ref[0, :, 2]

        outtbl[ 'pos' ].append( centers[ idx ][ 0 ] )
        outtbl[ 'ref' ].append( centers[ idx ][ 1 ] )
        outtbl[ 'wt_acc_pr' ].append( ref_acc[ scored_context ] )
        outtbl[ 'wt_don_pr' ].append( ref_don[ scored_context ] )

    outdf = pd.DataFrame( outtbl )

    return outdf

def splai_compute_ss_prob( annots_df,
                           models,
                           refvarseqs,
                           ref_name,
                           chrom,
                           center_var,
                           haplotypes,
                           ss_pos_l,
                           scored_context = 50,
                           rev_strand = False,
                         ):

    outtbl = { 'ref_name': [ ref_name ]*len( refvarseqs ),
               'chrom': [ chrom ]*len( refvarseqs ),
               'pos': [],
               'ref': [],
               'alt': [],
               'other_var': [],
             }

    for pos in ss_pos_l:
        outtbl[ 'ss_acc_prob_' + str( pos ) ] = []
        outtbl[ 'ss_don_prob_' + str( pos ) ] = []

    out_acc,out_don = np.empty( ( len( refvarseqs ), 2*scored_context + 1 ) ), np.empty( ( len( refvarseqs ), 2*scored_context + 1 ) )

    for idx, refvarseq in enumerate( refvarseqs ):

        refseq, varseq = refvarseq

        x_var = one_hot_encode( varseq )[ None, : ]
        y_var = np.mean( [ models[ m ].predict( x_var ) for m in range( 5 ) ], axis=0 )

        #flips the results so the positions are relative to the forward strand
        if rev_strand:
                y_var = y_var[:, ::-1]

        var_acc = y_var[0, :, 1]
        var_don = y_var[0, :, 2]

        #transforms variants into sequence position coords
        allvars_seqpos = [ ( scored_context + ( p - center_var[ idx ][ 0 ] ), r, a )
                           for p,r,a in [ center_var[ idx ] ] + haplotypes[ idx ] ]

        var_acc, var_don = adjust_for_indels( var_acc,
                                              var_don,
                                              allvars_seqpos,
                                             )

        out_acc[ idx, : ] = var_acc
        out_don[ idx, : ] = var_don

        ss_seqpos = [ scored_context + ( p - center_var[ idx ][ 0 ] ) for p in ss_pos_l ]

        outtbl[ 'pos' ].append( center_var[ idx ][ 0 ] )
        outtbl[ 'ref' ].append( center_var[ idx ][ 1 ] )
        outtbl[ 'alt' ].append( center_var[ idx ][ 2 ] )
        outtbl[ 'other_var' ].append( ';'.join( [ ':'.join( [ str( p ), '>'.join( [ r, a ] ) ] )
                                      for p,r,a in haplotypes[ idx ] ] ) )

        for ss_gen,ss_seq in zip( ss_pos_l, ss_seqpos ):

            if ss_seq >= 0 and ss_seq < 2*scored_context + 1:

                outtbl[ 'ss_acc_prob_' + str( ss_gen ) ].append( var_acc[ ss_seq ] )
                outtbl[ 'ss_don_prob_' + str( ss_gen ) ].append( var_don[ ss_seq ] )

            else:

                outtbl[ 'ss_acc_prob_' + str( ss_gen ) ].append( np.nan )
                outtbl[ 'ss_don_prob_' + str( ss_gen ) ].append( np.nan )

    outdf = pd.DataFrame( outtbl )
    outacc = pd.DataFrame( out_acc )
    outdon = pd.DataFrame( out_don )

    outacc = pd.concat( [ outdf[ [ 'chrom', 'pos', 'ref', 'alt', 'other_var' ] + [ col for col in outtbl if 'acc' in col ] ], outacc ],
                        axis = 1 )

    outdon = pd.concat( [ outdf[ [ 'chrom', 'pos', 'ref', 'alt', 'other_var' ] + [ col for col in outtbl if 'don' in col ] ], outdon ],
                        axis = 1 )

    return outdf, outacc, outdon

def splai_ss_prob_mult_variants_onegene( annots_df,
                                      models,
                                      refseq,
                                      ref_name,
                                      chrom,
                                      center_var,
                                      ss_pos_l,
                                      haplotypes = None,
                                      ref_vars = None,
                                      scored_context = 50,
                                      unscored_context = 5000,
                                      rev_strand = False ):

    strand = '-' if rev_strand else '+'

    flanks = scored_context + unscored_context

    if not haplotypes:

        haplotypes = [ [] for var in center_var ]

    if not ref_vars:

        ref_vars = [ [] for var in center_var ]

    else:

        for p,r,a in ref_vars:

            ref_name += '+' + str( p ) + ':' + r + '>' + a

    assert all( len(i) == len( center_var ) for i in [ haplotypes, ref_vars ] ), \
    'Haplotypes and ref_vars input must be either missing or the same length as the center_var list'

    #creates a giant list of ( reference seq, variant seq ) tuples
    refvarseqs = [ create_input_seq( refseq,
                                     center,
                                     hapref[ 0 ],
                                     hapref[ 1 ],
                                     get_gene_bds( annots_df,
                                                   chrom,
                                                   center[ 0 ],
                                                   strand,
                                                   scored_context = scored_context,
                                                   unscored_context = unscored_context,
                                                    ),
                                     scored_context,
                                     rev_strand = rev_strand,
                                     unscored_context = unscored_context )
                  for center,hapref in zip( center_var, zip( haplotypes, ref_vars ) ) ]

    #this will fail if any of the other variants are indels...
    #maybe I should add some functionality to the splai_score_variant fn to handle this...
    outdf, outacc, outdon = splai_compute_ss_prob( annots_df,
                                                   models,
                                                   refvarseqs,
                                                    ref_name,
                                                    chrom,
                                                    center_var,
                                                    haplotypes,
                                                    ss_pos_l,
                                                    scored_context = scored_context,
                                                    rev_strand = rev_strand,
                                                    )

    return outdf, outacc, outdon
