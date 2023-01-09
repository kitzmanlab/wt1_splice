import pandas as pd
import numpy as np
import scipy.stats as ss
import pysam
import hgvs.parser
import hgvs.dataproviders.uta
import hgvs.assemblymapper
import splanl.custom_splai_scores as css

def get_refseq( fa_file ):

    refseq = []

    with pysam.FastxFile( fa_file, 'r' ) as fa:
        refseq = { entry.name: entry.sequence for entry in fa }
        #for entry in fa:
            #refseq.append( entry.sequence )

    return refseq

def get_kmers(seq,
               k):

    return( [ seq[ i : i+k ] for i in range( len( seq ) - k + 1 ) ] )

def acceptors_donors(refseq,
                    byvartbl,
                    sdv_col,
                    out_col_suffix = '' ):

    tbv = byvartbl.copy()

    assert sdv_col in tbv, '%s is not in the dataframe columns' % sdv_col

    min_pos = tbv.pos.min()
    max_pos = tbv.pos.max()

    #accounts for 1 based numbering
    vec_seq = refseq.upper()[ min_pos-1: max_pos ]

    di_nts = get_kmers(vec_seq, 2 )

    for col in ['wt_acc','wt_don','psbl_snv_acc','psbl_snv_don']:
        tbv[ col ] = False

    for p, dnt in enumerate( di_nts ):

        #first check for cryptic exon locations
        if dnt == 'AG':
            tbv.loc[ ( tbv.pos == ( p + min_pos ) ) | ( tbv.pos == ( p + min_pos + 1 ) ), 'wt_acc' ] = True
        elif dnt == 'GT':
            tbv.loc[ ( tbv.pos == ( p + min_pos ) ) | ( tbv.pos == ( p + min_pos + 1 ) ), 'wt_don' ] = True
        #ok now look for possible acceptors and donors
        elif dnt.startswith( 'A' ):
            tbv.loc[ ( tbv.pos == ( p + min_pos + 1 ) ) & ( tbv.alt == 'G' ), 'psbl_snv_acc' ] = True
        elif dnt.endswith( 'G' ):
            tbv.loc[ ( tbv.pos == ( p + min_pos ) ) & ( tbv.alt == 'A' ), 'psbl_snv_acc' ] = True
        elif dnt.startswith( 'G' ):
            tbv.loc[ ( tbv.pos == ( p + min_pos + 1 ) ) & ( tbv.alt == 'T' ), 'psbl_snv_don' ] = True
        elif dnt.endswith( 'T' ):
            tbv.loc[ ( tbv.pos == ( p + min_pos ) ) & ( tbv.alt == 'G' ), 'psbl_snv_don' ] = True

    tbv[ 'snv_acc' + out_col_suffix ] = ( tbv[ sdv_col ] ) & ( tbv.psbl_snv_acc )
    tbv[ 'snv_don' + out_col_suffix ] = ( tbv[ sdv_col ] ) & ( tbv.psbl_snv_don )

    return tbv

def sdvs(byvartbl,
        sdv_col,
        sdv_thresh,
        abs_vals = True):

    tbv = byvartbl.copy()

    if abs_vals:
        tbv['sdv'] = np.abs( tbv[ sdv_col ] ) >= sdv_thresh
    else:
        tbv['sdv'] = tbv[ sdv_col ] >= sdv_thresh

    return tbv

def print_summary_info( byvartbl ):

    tbv = byvartbl.loc[ byvartbl.n_bc_passfilt > 0  ].copy()

    pos_var = 3*( tbv.pos.max() - tbv.pos.min() )
    seen_var = tbv.shape[0]
    seen_per = 100*( seen_var / pos_var )

    print( 'Out of %i possible variants, we see %i (%.2f%%).' % ( pos_var, seen_var, seen_per ) )

    sdvs = tbv.sdv.sum()
    sdv_per = 100*( sdvs / seen_var )

    print( 'Out of %i variants, %i (%.2f%%) are splice disrupting.' % ( seen_var, sdvs, sdv_per ) )

    if 'var_type' in tbv.columns:

        syn_tbv = tbv.query( 'var_type == "Synonymous"' ).copy()
        syn = syn_tbv.shape[0]
        syn_sdvs = syn_tbv.sdv.sum()
        syn_sdv_per = 100*( syn_sdvs / syn )

        print( 'Out of %i synononymous variants, %i (%.2f%%) are splice disrupting.' % ( syn, syn_sdvs, syn_sdv_per ) )

    if 'sdv_exon' in tbv.columns:

        ex_tbv = tbv.query( 'var_type != "Intronic"' ).copy()
        ex = ex_tbv.shape[0]
        ex_sdvs = ex_tbv.sdv_exon.sum()
        ex_sdv_per = 100*( ex_sdvs / ex )

        print( 'Out of %i exonic variants, %i (%.2f%%) are splice disrupting.' % ( ex, ex_sdvs, ex_sdv_per ) )

    if 'sdv_intron' in tbv.columns:

        intron_tbv = tbv.query( 'var_type == "Intronic"' ).copy()
        intron = intron_tbv.shape[0]
        intron_sdvs = intron_tbv.sdv_intron.sum()
        intron_sdv_per = 100*( intron_sdvs / intron )

        print( 'Out of %i intronic variants, %i (%.2f%%) are splice disrupting.' % ( intron, intron_sdvs, intron_sdv_per ) )

    pos_acc = tbv.psbl_snv_acc.sum()
    used_acc = tbv.snv_acc.sum()
    acc_per = 100*( used_acc / pos_acc )

    print( 'Out of %i possible alternate acceptors, %i (%.2f%%) have a high OTHER value.' % ( pos_acc, used_acc, acc_per ) )

    pos_don = tbv.psbl_snv_don.sum()
    used_don = tbv.snv_don.sum()
    don_per = 100*( used_don / pos_don )

    print( 'Out of %i possible alternate donors, %i (%.2f%%) have a high OTHER value.\n' % ( pos_don, used_don, don_per ) )

def stdize_cols_by_sample(  tbv,
                            std_cols ):

    out_tbl = tbv.copy()

    #if this is a long dataset
    if 'sample' in out_tbl.columns:

        for col in std_cols:
            #creates z score by sample while ignoring any missing values
            out_tbl[ 'z' + col ] = out_tbl.groupby( [ 'sample' ] )[ col ].transform( lambda x : ss.zscore( x, nan_policy='omit' ) )

    #if the data contains only one sample
    else:

        for col in std_cols:
            out_tbl[ 'z' + col ] = ss.zscore( out_tbl[ col ], nan_policy='omit' )

    return( out_tbl )

def across_sample_stats(ltbls,
                        lsampnames,
                        med_col_names):

    out_tbl = { 'sample_group':[],
                'sample':[],
                'psbl_var':[],
                'n_var':[],
                'n_var_ex':[],
                'n_var_in':[],
                'n_reads':[],
                'n_reads_passfilt':[],
                'n_usable_reads':[],
                'n_bc':[],
                'n_bc_passfilt':[],
                'n_unmapped':[],
                'n_badstart':[],
                'n_badend':[],
                'n_softclip':[],
                'n_otheriso':[],
                'n_sdv':[],
                'n_sdv_ex':[],
                'n_sdv_in':[],
                'psbl_alt_acc':[],
                'psbl_alt_don':[],
                'n_alt_acc':[],
                'n_alt_don':[]
                }

    for col in med_col_names:
        out_tbl['med_'+col] = []

    i=0
    for grp, _lsamp in lsampnames.items():
        for lsamp in _lsamp:

            lsamp_df = ltbls[ i ].query( 'sample=="%s"' % lsamp ).copy()
            lsamp_filt_df = lsamp_df.query( 'n_bc_passfilt > 0' )

            out_tbl['sample_group'].append( grp )
            out_tbl['sample'].append( grp+'_'+lsamp )
            out_tbl['psbl_var'].append( 3*( lsamp_df.pos.max() - lsamp_df.pos.min() ) )
            out_tbl['n_var'].append( int( lsamp_filt_df.shape[0] ) )
            out_tbl['n_reads'].append( int( lsamp_df.sum_reads.sum() ) )
            out_tbl['n_reads_passfilt'].append( int( lsamp_df.sum_reads_passfilt.sum() ) )
            out_tbl['n_usable_reads'].append( int( lsamp_df.sum_usable_reads.sum() ) )
            out_tbl['n_bc'].append( int( lsamp_df.n_bc.sum() ) )
            out_tbl['n_bc_passfilt'].append( int( lsamp_df.n_bc_passfilt.sum() ) )
            out_tbl['n_unmapped'].append( int( lsamp_df.sum_unmapped_reads.sum() ) )
            out_tbl['n_badstart'].append( int( lsamp_df.sum_bad_starts.sum() ) )
            out_tbl['n_badend'].append( int( lsamp_df.sum_bad_ends.sum() ) )
            out_tbl['n_softclip'].append( int( lsamp_df.sum_soft_clipped.sum() ) )
            out_tbl['n_otheriso'].append( int( lsamp_df.sum_other_isoform.sum() ) )

            if 'sdv' in lsamp_filt_df.columns:
                out_tbl['n_sdv'].append( int( lsamp_filt_df.sdv.sum() ) )
            else:
                out_tbl['n_sdv'].append( None )

            if 'psbl_snv' in lsamp_filt_df.columns:
                out_tbl['psbl_alt_acc'].append( int( lsamp_filt_df.psbl_snv_acc.sum() ) )
                out_tbl['psbl_alt_don'].append( int( lsamp_filt_df.psbl_snv_don.sum() ) )
                out_tbl['n_alt_acc'].append( int( lsamp_filt_df.snv_acc.sum() ) )
                out_tbl['n_alt_don'].append( int( lsamp_filt_df.snv_don.sum() ) )
            else:
                out_tbl['psbl_alt_acc'].append( None )
                out_tbl['psbl_alt_don'].append( None )
                out_tbl['n_alt_acc'].append( None )
                out_tbl['n_alt_don'].append( None )

            if 'var_type' in lsamp_filt_df.columns:
                out_tbl['n_var_ex'].append( int( lsamp_filt_df.query( 'var_type != "Intronic"' ).shape[0] ) )
                out_tbl['n_var_in'].append( int( lsamp_filt_df.query( 'var_type == "Intronic"' ).shape[0] ) )
                out_tbl['n_sdv_ex'].append( int( lsamp_filt_df.sdv_exon.sum() ) )
                out_tbl['n_sdv_in'].append( int( lsamp_filt_df.sdv_intron.sum() ) )
            else:
                out_tbl['n_var_ex'].append( None )
                out_tbl['n_var_in'].append( None )
                out_tbl['n_sdv_ex'].append( None )
                out_tbl['n_sdv_in'].append( None )

            for col in med_col_names:
                out_tbl['med_'+col].append( float( lsamp_df[ col ].median() ) )

        i+=1

    out_tbl = pd.DataFrame( out_tbl )

    out_tbl['per_var_seen'] = 100*( out_tbl.n_var / out_tbl.psbl_var )
    out_tbl['per_reads_passfilt'] = 100*( out_tbl.n_reads_passfilt / out_tbl.n_reads )
    out_tbl['per_bc_passfilt'] = 100*( out_tbl.n_bc_passfilt / out_tbl.n_bc )
    out_tbl['per_usable'] = 100*( out_tbl.n_usable_reads / out_tbl.n_reads_passfilt )
    out_tbl['per_unmapped'] = 100*( out_tbl.n_unmapped / out_tbl.n_reads_passfilt )
    out_tbl['per_badstart'] = 100*( out_tbl.n_badstart / out_tbl.n_reads_passfilt )
    out_tbl['per_badend'] = 100*( out_tbl.n_badend / out_tbl.n_reads_passfilt )
    out_tbl['per_softclip'] = 100*( out_tbl.n_softclip / out_tbl.n_reads_passfilt )
    out_tbl['per_otheriso'] = 100*( out_tbl.n_otheriso / out_tbl.n_reads_passfilt )
    out_tbl['per_sdv'] = 100*( out_tbl.n_sdv / out_tbl.n_var )
    out_tbl['per_sdv_ex'] = 100*( out_tbl.n_sdv_ex / out_tbl.n_var_ex )
    out_tbl['per_sdv_in'] = 100*( out_tbl.n_sdv_in / out_tbl.n_var_in )
    out_tbl['per_acc_used'] = 100*( out_tbl.n_alt_acc / out_tbl.psbl_alt_acc )
    out_tbl['per_don_used'] = 100*( out_tbl.n_alt_don / out_tbl.psbl_alt_don )

    return out_tbl

#This is the one letter universal translation table
#It handles cases of DNA ambiguity where the encoded amino acid is unambiguous.
#You need to deal with the missing cases where ambiguity codes would result in
#an ambiguous amino acid assignment. It is suggested that you use 'X' in these
#cases as this is the standard character for an unknown amino acid.
#Only Y (pyrimidine), R (purine) and N (any) degeneracy symbols are handled at
#this time. (need to add M,K,W,S,B,D,H,V where appropriate)
#Stop codons are symbolized as X
#Reassign TAA, TAG, TAR and TGA to change the stop codon symbol if desired.

transTab1L = {
'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
'TAT': 'Y', 'TAC': 'Y', 'TAA': 'X', 'TAG': 'X',
'TGT': 'C', 'TGC': 'C',  'TGA': 'X', 'TGG': 'W',
'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

def get_ref_amino( vartbl,
                    refseq,
                    exon_coords,
                    frame_shift = 0 ):

    assert isinstance( frame_shift, int) and frame_shift < 3, 'Frameshift must be a non-negative integer less than 3'

    tbv = vartbl.set_index( 'pos' ).copy()

    exon_seq = refseq[ exon_coords[0] - 1 + frame_shift : exon_coords[1] ].upper()

    daminos = { ( i + exon_coords[0] + frame_shift ) : transTab1L[ exon_seq[ i: i+3 ] ]
                for i in range( 0, len( exon_seq ), 3 )
                if len( exon_seq[ i: i+3 ] ) == 3  }

    #fills in the other positions
    for pos in list( daminos.keys() ):
        amino = daminos[pos]
        daminos[ pos + 1 ] = amino
        daminos[ pos + 2 ] = amino

    aa_df = pd.DataFrame.from_dict( daminos, orient='index' ).reset_index()
    aa_df = aa_df.rename( columns={'index':'pos', 0:'ref_aa'}).sort_values(by='pos').set_index('pos')

    out_tbl = pd.merge( tbv, aa_df, left_index=True, right_index=True, how = 'outer' ).reset_index()

    #drop rows that aren't associated with a variant, can't do an inner merge bc we'll lose any intronic variants
    out_tbl = out_tbl.dropna( subset=['varlist'] )

    return out_tbl

def get_snv_alt_amino( vartbl,
                        refseq,
                        exon_coords,
                        frame_shift = 0 ):

    assert isinstance( frame_shift, int) and frame_shift < 3, 'Frameshift must be a non-negative integer less than 3'

    tbv = vartbl.set_index( [ 'pos', 'alt' ] ).copy()

    exon_seq = refseq[ exon_coords[0] - 1 + frame_shift : exon_coords[1] ].upper()

    lcodons = [ exon_seq[ i: i+3 ] for i in range( 0, len( exon_seq ), 3 ) ]

    nt_sub = ['A', 'C', 'G', 'T']

    daminos = {}

    #handles the case in which the initial bases go to a codon in the upstream exon
    for i in range( frame_shift ):

        #adjust for 1 and 0 based coordinates
        ref = refseq[ exon_coords[0] - 1 + i ].upper()

        if ( i + exon_coords[0] ) not in daminos:
            daminos[ i + exon_coords[0] ] = {}

        for snv in nt_sub:

            if snv == ref:
                continue

            daminos[ i + exon_coords[0] ][ snv ] = 'Exonic - out of frame'

    for i in range( exon_coords[1] - exon_coords[0] - frame_shift ):

        #adjust for 1 and 0 based coordinates
        ref = refseq[ exon_coords[0] - 1 + frame_shift + i ].upper()
        codon = lcodons[ i // 3 ]
        snv_pos = i % 3

        if ( i + exon_coords[0] + frame_shift ) not in daminos:
            daminos[ i + exon_coords[0] + frame_shift ] = {}

        for snv in nt_sub:

            if snv == ref:
                continue

            if len( codon ) == 3:
                daminos[ i + exon_coords[0] + frame_shift ][ snv ] = transTab1L[ codon[ :snv_pos ] + snv + codon[ snv_pos + 1: ] ]
            #handles the case in which the final codon goes into the next downstream exon
            else:
                daminos[ i + exon_coords[0] + frame_shift ][ snv ] = 'Exonic - out of frame'

    #creates dataframe indexed on position and alt with alternate aminos as the only column
    aa_df = pd.concat( { k: pd.DataFrame.from_dict( v, 'index', columns = [ 'alt_aa' ] ) for k, v in daminos.items() }, axis=0 ).reset_index()
    aa_df = aa_df.rename( columns={ 'level_0':'pos', 'level_1':'alt' } ).set_index( [ 'pos', 'alt' ] )

    out_tbl = pd.merge( tbv, aa_df, left_index=True, right_index=True, how = 'outer' ).reset_index()

    #drop rows that aren't associated with a variant, can't do an inner merge bc we'll lose any intronic variants
    out_tbl = out_tbl.dropna( subset=['varlist'] )

    return out_tbl

def extract_var_type( vartbl,
                      refseq,
                        exon_coords,
                        frame_shift = 0,
                        overwrite = True ):

    assert isinstance( frame_shift, int) and frame_shift < 3, 'Frameshift must be a non-negative integer less than 3'

    tbv = vartbl.sort_values( by = 'pos' ).copy()

    if 'ref_aa' not in tbv.columns or overwrite:
        tbv = get_ref_amino( tbv, refseq, exon_coords, frame_shift )

    if 'alt_aa' not in tbv.columns or overwrite:
        tbv = get_snv_alt_amino( tbv, refseq, exon_coords, frame_shift )

    var_type = []

    for ref, alt in zip( tbv.ref_aa.values, tbv.alt_aa.values ):

        #checking if alt is missing but with a work around since np.isnan fails with strings
        if not isinstance( alt, str ):
            var_type.append( 'Intronic' )
        elif alt == 'Exonic - out of frame':
            var_type.append('Exonic - out of frame')
        elif ref == alt:
            var_type.append( 'Synonymous' )
        elif alt == 'X':
            var_type.append( 'Nonsense' )
        else:
            var_type.append( 'Missense' )

    tbv['var_type'] = var_type

    return tbv

def sdv_by_var_type( vartbl ):

    tbv = vartbl.copy()

    #check that variant type is in the data and not all missing
    assert ( 'var_type' in tbv.columns ) and ( tbv.shape[0] != tbv.var_type.isnull().sum() ), \
    'var_type must be a non-empty column in the dataframe'

    assert ( 'sdv' in tbv.columns ), 'sdv must be a column in the dataframe'

    tbv['sdv_exon'] = ( ( tbv.sdv ) & ( tbv.var_type != 'Intronic' ) )

    tbv['sdv_intron'] = ( ( tbv.sdv ) & ( tbv.var_type == 'Intronic' ) )

    return tbv

def identify_var( vartbl,
                 pat_pos_alt,
                 out_col ):

    tbv = vartbl.copy()

    tbv[ out_col ] = False

    for pos, alt in pat_pos_alt:

        tbv.loc[ ( tbv.pos == pos ) & ( tbv.alt == alt ), out_col ] = True

    return tbv

def frameshift( tbl_byvar,
                acc_don_bool,
                exon_bd,
                acc = True
              ):

    tbv = tbl_byvar.copy()

    if acc:
        col = 'frameshift_acc'
    else:
        col = 'frameshift_don'

    tbv[ col ] = False

    posl = tbv.loc[ tbv[ acc_don_bool ] ].pos
    altl = tbv.loc[ tbv[ acc_don_bool ] ].alt

    if acc:
        assert len( set( altl ) ) == 2 and 'C' not in set( altl ) and 'T' not in set( altl ), \
        'Alternate alleles for acceptor have bases other than A/G'
    else:
        assert len( set( altl ) ) and 'A' not in set( altl ) and 'C' not in set( altl ), \
        'Alternate alleles for donor have bases other than G/T'

    fs = []

    for p,a in zip( posl, altl ):

        if acc:

            #for this exon bd should be vec_coord of first exonic base
            if a.upper() == 'A':

                fs.append( ( ( ( exon_bd - ( p + 2 ) ) % 3 ) != 0 ) )

            elif a.upper() == 'G':

                fs.append( ( ( ( exon_bd - ( p + 1 ) ) % 3 ) != 0 ) )

        else:

            #for this exon bd should be vec_coord of last exonic base
            if a.upper() == 'G':

                fs.append( ( ( ( exon_bd - ( p + 1 ) ) % 3 ) != 0 ) )

            elif a.upper() == 'T':

                fs.append( ( ( ( exon_bd - p ) % 3 ) != 0 ) )

    tbv.loc[ tbv[ acc_don_bool ], col ] = fs

    return tbv

def bootstrap_null_distribution( n_bcs,
                                 wt_bc_df,
                                 byvar_df,
                                 seed = 1687,
                                 iso_names = None ):

    wt_psi = wt_bc_df.copy()
    tbv = byvar_df.copy()

    assert all( n > 0 for n in n_bcs ), 'Inputting a sample with 0 barcodes will lead to division by 0 errors'

    wt_bcs = wt_psi.index.tolist()

    if not iso_names:

        iso_names = [ col[:-4] for col in wt_psi if col.endswith( '_psi' ) ]

        assert len( iso_names ) > 0, 'Could not infer isoform names - please input names to use'

    np.random.seed( seed )

    wt_dfs = [ wt_psi.loc[ np.random.choice( wt_bcs, int( n ) ) ] for n in n_bcs ]

    for iso in iso_names:

        wmeans = [ ( wt_df.usable_reads * wt_df[ iso + '_psi' ] ).sum() / wt_df.usable_reads.sum()
                   for wt_df in wt_dfs ]

        tbv[ 'wmean_bs_WT_' + iso ] = np.mean( wmeans )
        tbv[ 'wstdev_bs_WT_' + iso ] = np.std( wmeans )

    return tbv

def compute_intron_null_allvar( tbl_byvar,
                                exon_cds,
                                intron_bds,
                                pos_col = 'pos',
                                iso_names = None
                              ):

    tbv = tbl_byvar.copy()

    if not iso_names:

        iso_names = [ col[ 6: ] for col in tbv if col.startswith( 'wmean_' ) ]

        assert iso_names, 'Isoform names could not be inferred, please enter them directly'

    introns = tbv.loc[ ( tbv[ pos_col ] < exon_cds[ 0 ] - intron_bds ) | ( tbv[ pos_col ] > exon_cds[ 1 ] + intron_bds ) ].copy()
    exons = tbv.loc[ ( tbv[ pos_col ] >= exon_cds[ 0 ] - intron_bds ) | ( tbv[ pos_col ] <= exon_cds[ 1 ] + intron_bds ) ].copy()

    for iso in iso_names:

        tbv[ 'wmean_int_' + iso ] = introns[ 'wmean_' + iso ].mean()
        tbv[ 'wstdev_int_' + iso ] = introns[ 'wmean_' + iso ].std()

        tbv[ 'sq_inv_eff_' + iso ] = ( ( len( introns ) * len( exons ) ) * ( len( introns ) + len( exons ) - 2 ) ) \
                                    / ( ( len( introns ) + len( exons ) ) \
                                    * ( ( ( len( introns ) - 1 ) * introns[ 'wmean_' + iso ].std()**2 ) \
                                    + ( ( len( exons ) - 1 ) * exons[ 'wmean_' + iso ].std()**2 ) ) )

    return tbv

def bootstrap_varsp_null_distribution( null_bc_df,
                                       byvar_df,
                                       seed = 1687,
                                       iso_names = None,
                                       bootstraps = 1000, ):

    null_psi = null_bc_df.copy()
    tbv = byvar_df.copy()

    n_bcs = tbv.n_bc_passfilt.tolist()

    assert all( n > 0 for n in n_bcs ), 'Inputting a sample with 0 barcodes will lead to division by 0 errors'

    null_bcs = null_psi.index.tolist()

    if not iso_names:

        iso_names = [ col[ : -4 ] for col in null_psi if col.endswith( '_psi' ) ]

        assert len( iso_names ) > 0, 'Could not infer isoform names - please input names to use'

    usable_reads = null_psi.usable_reads.to_numpy()

    sample_tbl = {}
    null_iso = {}

    for iso in iso_names:

        sample_tbl[ 'wmean_bs_null_' + iso ] = []
        sample_tbl[ 'wstdev_bs_null_' + iso ] = []

        null_iso[ iso ] = null_psi[ iso + '_psi' ].to_numpy()

    bcs_sampled = {}

    for i,n_bc in enumerate( n_bcs ):

        if n_bc in bcs_sampled:

            idx = bcs_sampled[ n_bc ]

            for iso in iso_names:

                sample_tbl[ 'wmean_bs_null_' + iso ].append( sample_tbl[ 'wmean_bs_null_' + iso ][ idx ] )
                sample_tbl[ 'wstdev_bs_null_' + iso ].append( sample_tbl[ 'wstdev_bs_null_' + iso ][ idx ] )

            continue

        bcs_sampled[ n_bc ] = i

        np.random.seed( seed )

        null_idx = np.random.randint( len( null_psi ), size = ( bootstraps, int( n_bc ) ) )

        for iso in iso_names:

            mus = ( usable_reads[ null_idx ] * null_iso[ iso ][ null_idx ] ).sum( axis = 1 ) / usable_reads[ null_idx ].sum( axis = 1 )

            sample_tbl[ 'wmean_bs_null_' + iso ].append( np.mean( mus ) )
            sample_tbl[ 'wstdev_bs_null_' + iso ].append( np.std( mus ) )

    #samp_df = pd.DataFrame( sample_tbl )

    for iso in iso_names:

        tbv[ 'wmean_bs_null_' + iso ] = sample_tbl[ 'wmean_bs_null_' + iso ]
        tbv[ 'wstdev_bs_null_' + iso ] = sample_tbl[ 'wstdev_bs_null_' + iso ]

    #tbv = pd.concat( [ tbv.reset_index(), samp_df.reset_index() ],
                      #axis = 1, )

    print( 'done' )

    return tbv

def compute_null_zscores( tbl_byvar,
                          null_stem,
                          iso_names ):

    tbv = tbl_byvar.copy()

    for iso in iso_names:

        tbv[ '_'.join( [ 'zwmean', null_stem, iso ] ) ] = ( tbv[ 'wmean_' + iso ] \
                                                           - tbv[ '_'.join( [ 'wmean', null_stem, iso ] ) ] ) \
                                                           / tbv[ '_'.join( [ 'wstdev', null_stem, iso ] ) ]

    return tbv

def stouffers_z( tbl_byvar_wide,
                 iso_names,
                 zcol = 'zmean_',
                 weight = False ):

    tbv = tbl_byvar_wide.copy()

    for iso in iso_names:

        if not weight:

            tbv[ zcol + iso ] = tbv[ [ col for col in tbv if col.endswith( zcol + iso ) ] ].sum( axis = 1 ) \
                                / np.sqrt( tbv[ [ col for col in tbv if col.endswith( zcol + iso ) ] ].notnull().sum( axis = 1 ) )

        else:

            tbv[ zcol[ 1: ] + iso ] = ( tbv[ [ col for col in tbv if col.endswith( zcol[ :-2 ] + '_' + iso ) ] ] * weight[ iso ] ).sum( axis = 1 ) \
                                / np.sqrt( ( weight[ iso ]**2 ).sum() )

    return tbv

def sdv_by_iso( tbl_byvar,
                iso_names,
                z_col_stem,
                z_thresh,
                fc_col_stem,
                fc_thresh,
                chg_null_col_stem = None,
                chg_meas_col_stem = None,
                chg_thresh = None,
                out_col_stem = 'sdv_',
                bi_directional = True ):

    tbv = tbl_byvar.copy()

    if chg_null_col_stem or chg_meas_col_stem:

        assert chg_thresh, 'To test for change in PSI please specify null column, measured column, and change in PSI threshold'

    for iso in iso_names:

        if not chg_thresh:

            if bi_directional:
                tbv[ out_col_stem + iso ] = ( np.abs( tbv[ z_col_stem + iso ] ) >= z_thresh ) \
                                            & ( ( tbv[ fc_col_stem + iso ] >= fc_thresh ) | ( tbv[ fc_col_stem + iso ] <= 1/fc_thresh ) )
            else:
                tbv[ out_col_stem + iso ] = ( tbv[ z_col_stem + iso ] >= z_thresh ) \
                                            & ( tbv[ fc_col_stem + iso ] >= fc_thresh )

        else:

            assert chg_null_col_stem and chg_meas_col_stem, 'To test for change in PSI please specify null column, measured column, and change in PSI threshold'

            if bi_directional:
                tbv[ out_col_stem + iso ] = ( np.abs( tbv[ z_col_stem + iso ] ) >= z_thresh ) \
                                            & ( ( tbv[ fc_col_stem + iso ] >= fc_thresh ) | ( tbv[ fc_col_stem + iso ] <= 1/fc_thresh ) ) \
                                            & ( np.abs( tbv[ chg_meas_col_stem + iso ] - tbv[ chg_null_col_stem + iso ] ) >= chg_thresh )
            else:
                tbv[ out_col_stem + iso ] = ( tbv[ z_col_stem + iso ] >= z_thresh ) \
                                            & ( tbv[ fc_col_stem + iso ] >= fc_thresh ) \
                                            & ( ( tbv[ chg_meas_col_stem + iso ] - tbv[ chg_null_col_stem + iso ] ) >= chg_thresh )

    return tbv

def compute_fold_change( tbl_byvar,
                         null_col_stem,
                         test_col_stem,
                         iso_names = None,
                         out_col = 'fc_' ):

    tbv = tbl_byvar.copy()

    if not iso_names:

        iso_names = [ col[ len( null_col_stem ): ] for col in tbv if col.startswith( null_col_stem ) ]

        assert len( iso_names ) > 0, 'Cannot infer isoform names - please provide them directly'

    for iso in iso_names:

        tbv[ out_col + iso ] = tbv[ test_col_stem + iso ] / tbv[ null_col_stem + iso ]

    return tbv

def get_transcripts( variant,
                     genome = 'GRCh37', ):

    hp = hgvs.parser.Parser()
    hdp = hgvs.dataproviders.uta.connect()
    am = hgvs.assemblymapper.AssemblyMapper( hdp,
                                             assembly_name = genome,
                                             alt_aln_method='splign',
                                             replace_reference=True )

    parsed_variant = hp.parse_hgvs_variant( variant )

    print( 'Variant', str( parsed_variant ) )

    print( 'Transcripts', am.relevant_transcripts ( parsed_variant ) )

def gDNA_to_protein( chrom_id,
                     transcript_id,
                      gdna_tbl,
                      var_col,
                      genome = 'GRCh37',
                      hgvs_col = 'hgvs_var',
                      protein_col = 'protein_var' ):

    out_tbl = gdna_tbl.copy()

    hp = hgvs.parser.Parser()
    hdp = hgvs.dataproviders.uta.connect()
    am = hgvs.assemblymapper.AssemblyMapper( hdp,
                                             assembly_name = genome,
                                             alt_aln_method='splign',
                                             replace_reference=True )

    out_tbl[ hgvs_col ] = [ str( am.g_to_c( hp.parse_hgvs_variant( ':'.join( [ chrom_id, var ] ) ), transcript_id ) ).split( ':' )[ 1 ]
                              for var in out_tbl[ var_col ] ]

    out_tbl[ protein_col ] = [ str( am.c_to_p( hp.parse_hgvs_variant( ':'.join( [ transcript_id, var ] ) ) ) ).split( ':' )[ 1 ]
                                for var in out_tbl[ hgvs_col ] ]

    return out_tbl

def gDNA_to_cDNA( chrom_id,
                  transcript_id,
                  gdna_tbl,
                  var_col,
                  genome = 'GRCh37',
                  hgvs_col = 'hgvs_var' ):

    out_tbl = gdna_tbl.copy()

    hp = hgvs.parser.Parser()
    hdp = hgvs.dataproviders.uta.connect()
    am = hgvs.assemblymapper.AssemblyMapper( hdp,
                                             assembly_name = genome,
                                             alt_aln_method='splign',
                                             replace_reference=True )

    out_tbl[ hgvs_col ] = [ str( am.g_to_c( hp.parse_hgvs_variant( ':'.join( [ chrom_id, var ] ) ), transcript_id ) ).split( ':' )[ 1 ]
                              for var in out_tbl[ var_col ] ]

    return out_tbl

def possible_ss( tbl_by_var,
                 refseq,
                 out_cols = ( 'acceptor_created', 'donor_created' ),
                 rev_strand = False ):

    tbv = tbl_by_var.copy()

    acc = []
    don = []

    for pos, ra in zip( tbv.hg19_pos, zip( tbv.ref, tbv.alt ) ):

        ref,alt = ra

        assert refseq[ pos -1: pos - 1 + len( ref ) ].upper() == ref, 'Reference does not match sequence at %i' % pos

        if rev_strand:
            alt = css.rev_complement( alt )

        if alt == 'C':

            acc.append( False )
            don.append( False )

        elif alt == 'A':

            don.append( False )

            if not rev_strand:

                acc.append( refseq[ pos ].upper() == 'G' )

            else:

                acc.append( css.rev_complement( refseq[ pos - 2 ].upper() ) == 'G' )

        elif alt == 'T':

            acc.append( False )

            if not rev_strand:

                don.append( refseq[ pos - 2 ].upper() == 'G' )

            else:

                don.append( css.rev_complement( refseq[ pos ].upper() ) == 'G' )

        elif alt == 'G':

            if not rev_strand:

                acc.append( refseq[ pos - 2 ].upper() == 'A' )
                don.append( refseq[ pos ].upper() == 'T' )

            else:

                acc.append( css.rev_complement( refseq[ pos ].upper() ) == 'A' )
                don.append( css.rev_complement( refseq[ pos - 2 ].upper() ) == 'T' )

        else:

            acc.append( False )
            don.append( False )

    tbv[ out_cols[ 0 ] ] = acc
    tbv[ out_cols[ 1 ] ] = don

    return tbv

def saturate_variants( tbl_by_var,
                       refseq,
                       pos_col,
                       exon_col,
                       intron_dist = 100,
                       rev_strand = False,
                       add_missing_introns = False ):

    tbv = tbl_by_var.copy()

    exons = tbv.loc[ tbv[ exon_col ].notnull() ][ exon_col ].unique()

    exon_bds = { int( ex ): ( int( tbv.loc[ tbv[ exon_col ] == ex ][ pos_col ].min() ) - intron_dist,
                              int( tbv.loc[ tbv[ exon_col ] == ex ][ pos_col ].max() ) + intron_dist )
                 for ex in exons }

    by_ex_d = { int( ex ): tbv.loc[ ( tbv[ pos_col ] >= exon_bds[ int( ex ) ][ 0 ] )
                                    & ( tbv[ pos_col ] <= exon_bds[ int( ex ) ][ 1 ] ) ].copy()
                for ex in exons }

    for ex in by_ex_d.keys():

        if add_missing_introns:
            min_bd = min( by_ex_d[ ex ][ pos_col ].min(), exon_bds[ ex ][ 0 ] )
            max_bd = max( by_ex_d[ ex ][ pos_col ].max(), exon_bds[ ex ][ 1 ] )
        else:
            min_bd, max_bd = by_ex_d[ ex ][ pos_col ].min(), by_ex_d[ ex ][ pos_col ].max(),

        merge_ex = pd.DataFrame( { pos_col: [ p for p in range( min_bd, max_bd + 1 ) for j in range( 3 ) ],
                                   'alt': [ a for p in range( min_bd, max_bd + 1 )
                                              for a in [ 'A', 'C', 'G', 'T' ] if a.upper() != refseq[ p - 1 ].upper() ],
                                   'ref': [ refseq[ p - 1 ].upper() for p in range( min_bd, max_bd + 1 )
                                            for j in range( 3 ) ] } )

        if rev_strand:

            merge_ex[ 'alt_c' ] = [ css.rev_complement( a ) for a in merge_ex.alt ]
            merge_ex[ 'ref_c' ] = [ css.rev_complement( r ) for r in merge_ex.ref ]
            #merge_ex[ 'pos' ] = -merge_ex[ pos_col ]


        idx = merge_ex.columns.tolist()

        by_ex_d[ int( ex ) ] = merge_ex.set_index( idx ).merge( by_ex_d[ int( ex ) ].set_index( idx ),
                                                                how = 'outer',
                                                                left_index = True,
                                                                right_index = True ).reset_index()

    return by_ex_d

def merge_clinvar( tbl_by_var,
                   clinvar,
                   index_cols = [ 'hg19_pos', 'ref', 'alt' ] ):

    tbv = tbl_by_var.set_index( index_cols ).copy()
    cv = clinvar.set_index( index_cols ).copy()

    tbv = tbv.merge( cv,
                     how = 'left',
                     left_index = True,
                     right_index = True ).reset_index()

    tbv[ 'clinvar' ] = tbv[ 'clinvar_interp' ].notnull()

    return tbv
