import pandas as pd
import numpy as np
from datetime import date
import time
import splanl.post_processing as pp
import splanl.custom_splai_scores as css

def create_all_SNVs( annot,
                     exon,
                     chr_seq ):

    ex_start = int( annot.loc[ annot.exon == exon ].start_hg19 ) + 1
    ex_end = int( annot.loc[ annot.exon == exon ].end_hg19 )

    snvs = [ ( p, chr_seq[ p - 1 ].upper(), base ) for p in range( ex_start, ex_end + 1 )
                                                     for base in [ 'A', 'C', 'G', 'T' ]
                                                     if base != chr_seq[ p - 1 ].upper() ]

    return snvs

def create_all_DNVs( snvs_tup ):

    dnvs_cent = []
    dnvs_hap = []

    for snv_i in snvs_tup:

        for snv_j in snvs_tup:

            if snv_i[ 0 ] >= snv_j[ 0 ]:
                continue

            dnvs_cent.append( snv_i )
            dnvs_hap.append( [ snv_j ] )

    return ( dnvs_cent, dnvs_hap )

def create_all_SNVs_DNVs( annot,
                          exon,
                          chr_seq ):

    snvs = create_all_SNVs( annot,
                            exon,
                            chr_seq )

    dnvs_c, dnvs_h = create_all_DNVs( snvs )

    return( snvs, ( dnvs_c, dnvs_h ) )

def splai_score_SNVs_DNVs( annots_bed,
                           splai_annots,
                           models,
                           refseq,
                           ref_name,
                           chrom,
                           exon,
                           snvs,
                           dnvs_tup,
                           mask_value = 0,
                           rev_strand = False ):

    t0 = time.time()

    scored_context = int( annots_bed.loc[ annots_bed.exon == exon ].exon_len_hg19 ) + 10

    snvs_scored = css.splai_score_mult_variants_onegene( splai_annots,
                                                          models,
                                                          refseq,
                                                          ref_name,
                                                          chrom,
                                                          snvs,
                                                          mask_value = mask_value,
                                                          scored_context = scored_context,
                                                          rev_strand = rev_strand )


    t1 = time.time()

    print( 'Done scoring SNVs after %.2f seconds' % ( t1 - t0 ) )

    dnvs_scored = css.splai_score_mult_variants_onegene( splai_annots,
                                                          models,
                                                          refseq,
                                                          ref_name,
                                                          chrom,
                                                          dnvs_tup[ 0 ],
                                                          haplotypes = dnvs_tup[ 1 ],
                                                          mask_value = mask_value,
                                                          scored_context = scored_context,
                                                          rev_strand = rev_strand )

    t2 = time.time()

    print( 'Done scoring DNVs after %.2f minutes' % ( ( t2 - t1 ) / 60 ) )

    return ( snvs_scored, dnvs_scored )

def locate_acceptors( seq,
                      acceptor_seq = 'AG',
                      offset = 2 ):

    seq_up = seq.upper()
    acceptor_up = acceptor_seq.upper()

    accept_loc = np.concatenate( ( [ 0 ]*offset,
                                   np.array( [ bp1 + bp2 == acceptor_up for bp1, bp2 in zip( seq_up, seq_up[ 1: ] ) ],
                                             dtype = int )[ : -offset + 1 ] ) )

    return accept_loc

def locate_donors( seq,
                   donor_seq = 'GT',
                   offset = 1 ):

    seq_up = seq.upper()
    donor_up = donor_seq.upper()

    donor_loc = np.concatenate( ( np.array( [ bp1 + bp2 == donor_up for bp1, bp2 in zip( seq_up, seq_up[ 1: ] ) ],
                                  dtype = int )[ offset: ],
                                  [ 0 ]*offset ) )

    return donor_loc

def classify_loss( event_pos,
                   canonical_bds,
                   var1_pos,
                   var2_pos = None,
                   distal_dist = 20 ):

    var1_distal = abs( event_pos - var1_pos ) >= distal_dist

    if var2_pos:

        var2_distal = abs( event_pos - var2_pos ) >= distal_dist

    else:

        var2_distal = True

    if event_pos in canonical_bds:

        if not ( var1_distal and var2_distal ):

            outcome = 'Native'

        else:

            outcome = 'Distal native'

    else:

        if not ( var1_distal and var2_distal ):

            outcome = 'Non-native'

        else:

            outcome = 'Distal non-native'

    return outcome

def classify_gain_snv( event_pos,
                   canonical_bds,
                   wt_ss,
                   var_ss,
                   var_pos,
                   distal_dist = 20 ):

    var_distal = abs( event_pos - var_pos ) >= distal_dist

    if event_pos in canonical_bds:

        if not var_distal:

            outcome = 'Native'

        else:

            outcome = 'Distal native'

    else:

        if wt_ss == 1 and var_ss == 1:

            if not var_distal:

                outcome = 'Existing cryptic'

            else:

                outcome = 'Existing distal cryptic'

        elif wt_ss == 0 and var_ss == 1:

            outcome = 'Created cryptic'

        elif var_ss == 0:

            outcome = 'Unknown NC cryptic'

    return outcome

def categorize_splai_snvs( tbl_by_var,
                          refseq,
                          annot,
                          exon_len,
                          rev_strand = False,
                          pos_col = 'POS_maxrw',
                          event_col = 'DS_maxrw_type',
                          out_col = 'splai_cat',
                          distal_dist = 20 ):

    tbv = tbl_by_var.copy()

    tbv[ pos_col + '_hg19' ] = tbv.pos + tbv[ pos_col ]

    cats = []

    c_vars = [ ( p, ra[ 0 ], ra[ 1 ] ) for p, ra in zip( tbv.pos, zip( tbv.ref, tbv.alt ) ) ]

    scored_context = exon_len + 10

    exon_starts = [ int( p ) + 1 for p in annot.EXON_START.tolist()[ 0 ].split( ',' ) ]
    exon_ends = [ int( p ) for p in annot.EXON_END.tolist()[ 0 ].split( ',' ) ]

    for c_var, event in zip( c_vars, zip( tbv[ event_col ], tbv[ pos_col ] ) ):

        if 'DS_AL' in event[ 0 ]:

            cats.append( ' '.join( [ classify_loss( event[ 1 ] + c_var[ 0 ],
                                                    exon_starts,
                                                    c_var[ 0 ],
                                                    distal_dist = distal_dist ),
                                     event[ 0 ][ 3: ] ] ) )

        elif 'DS_DL' in event[ 0 ]:

            cats.append( ' '.join( [ classify_loss( event[ 1 ] + c_var[ 0 ],
                                                    exon_ends,
                                                    c_var[ 0 ],
                                                    distal_dist = distal_dist ),
                                      event[ 0 ][ 3: ] ] ) )

        elif 'DS_AG' in event[ 0 ]:

            wt, var = css.create_input_seq( refseq,
                                            c_var,
                                            [],
                                            [],
                                            ( 0, 0 ),
                                            scored_context,
                                            rev_strand = rev_strand,
                                            unscored_context = 0, )

            wt_acc = locate_acceptors( wt )
            var_acc = locate_acceptors( var )

            cats.append( ' '.join( [ classify_gain_snv( event[ 1 ] + c_var[ 0 ],
                                                        exon_starts,
                                                        wt_acc[ scored_context + int( event[ 1 ] ) ],
                                                        var_acc[ scored_context + int( event[ 1 ] ) ],
                                                        c_var[ 0 ],
                                                        distal_dist = distal_dist ),
                                                        event[ 0 ][ 3: ] ] ) )

        elif 'DS_DG' in event[ 0 ]:

            wt, var = css.create_input_seq( refseq,
                                            c_var,
                                            [],
                                            [],
                                            ( 0, 0 ),
                                            scored_context,
                                            rev_strand = rev_strand,
                                            unscored_context = 0, )

            wt_don = locate_donors( wt )
            var_don = locate_donors( var )

            if var_don[ scored_context + int( event[ 1 ] ) ] == 1:

                cats.append( ' '.join( [ classify_gain_snv( event[ 1 ] + c_var[ 0 ],
                                                            exon_ends,
                                                            wt_don[ scored_context + int( event[ 1 ] ) ],
                                                            var_don[ scored_context + int( event[ 1 ] ) ],
                                                            c_var[ 0 ],
                                                            distal_dist = distal_dist ),
                                                            event[ 0 ][ 3: ] ] ) )

            elif var_don[ scored_context + int( event[ 1 ] ) ] == 0:

                wt_gc_don = locate_donors( wt, donor_seq = 'GC' )
                var_gc_don = locate_donors( var, donor_seq = 'GC' )

                if var_gc_don[ scored_context + int( event[ 1 ] ) ] == 1:

                    cats.append( ' '.join( [ classify_gain_snv( event[ 1 ] + c_var[ 0 ],
                                                                exon_ends,
                                                                wt_gc_don[ scored_context + int( event[ 1 ] ) ],
                                                                var_gc_don[ scored_context + int( event[ 1 ] ) ],
                                                                c_var[ 0 ],
                                                                distal_dist = distal_dist ),
                                                                'GC',
                                                                event[ 0 ][ 3: ] ] ) )

                elif var_gc_don[ scored_context + int( event[ 1 ] ) ] == 0:

                    cats.append( 'Unknown NC cryptic ' + event[ 0 ][ 3: ] )

    tbv[ out_col ] = cats

    return tbv

def classify_gain_dnv( event_pos,
                       canonical_bds,
                       wt_ss,
                       snv1_ss,
                       snv2_ss,
                       dnv_ss,
                       var1_pos,
                       var2_pos,
                       distal_dist = 20 ):

    var1_distal = abs( event_pos - var1_pos ) >= distal_dist
    var2_distal = abs( event_pos - var2_pos ) >= distal_dist

    if event_pos in canonical_bds:

        if not ( var1_distal and var2_distal ):

            outcome = 'Native'

        else:

            outcome = 'Distal native'

    else:

        if wt_ss == 1 and dnv_ss == 1:

            if not( var1_distal and var2_distal ):

                outcome = 'Existing cryptic'

            else:

                outcome = 'Existing distal cryptic'

        elif wt_ss == 0 and dnv_ss == 1 and ( snv1_ss == 1 or snv2_ss == 1 ):

            outcome = 'SNV created cryptic'

        elif wt_ss == 0 and dnv_ss == 1 and snv1_ss == 0 and snv2_ss == 0:

            outcome = 'DNV created cryptic'

        elif dnv_ss == 0:

            outcome = 'Unknown NC cryptic'

    return outcome

def categorize_splai_dnvs( tbl_by_var,
                          refseq,
                          annot,
                          exon_len,
                          rev_strand = False,
                          pos_col = 'POS_maxrw',
                          event_col = 'DS_maxrw_type',
                          out_col = 'splai_cat',
                          distal_dist = 20 ):

    tbv = tbl_by_var.copy()

    tbv[ pos_col + '_hg19' ] = tbv.pos + tbv[ pos_col ]

    cats = []

    c_vars = [ ( p, ra[ 0 ], ra[ 1 ] ) for p, ra in zip( tbv.pos, zip( tbv.ref, tbv.alt ) ) ]

    haps = [ [ ( int( var.split( ':' )[ 0 ] ),
                 var.split( ':' )[ 1 ].split( '>' )[ 0 ],
                 var.split( ':' )[ 1 ].split( '>' )[ 1 ] ) ] for var in tbv.other_var ]

    scored_context = exon_len + 10

    exon_starts = [ int( p ) + 1 for p in annot.EXON_START.tolist()[ 0 ].split( ',' ) ]
    exon_ends = [ int( p ) for p in annot.EXON_END.tolist()[ 0 ].split( ',' ) ]

    for c_hap, event in zip( zip( c_vars, haps ), zip( tbv[ event_col ], tbv[ pos_col ] ) ):

        if 'DS_AL' in event[ 0 ]:

            cats.append( ' '.join( [ classify_loss( event[ 1 ] + c_hap[ 0 ][ 0 ],
                                                    exon_starts,
                                                    c_hap[ 0 ][ 0 ],
                                                    c_hap[ 1 ][ 0 ][ 0 ],
                                                    distal_dist = distal_dist ),
                                     event[ 0 ][ 3: ] ] ) )

        elif 'DS_DL' in event[ 0 ]:

            cats.append( ' '.join( [ classify_loss( event[ 1 ] + c_hap[ 0 ][ 0 ],
                                                    exon_ends,
                                                    c_hap[ 0 ][ 0 ],
                                                    c_hap[ 1 ][ 0 ][ 0 ],
                                                    distal_dist = distal_dist ),
                                      event[ 0 ][ 3: ] ] ) )

        elif 'DS_AG' in event[ 0 ]:

            snv_dist = int( c_hap[ 0 ][ 0 ] ) - int( c_hap[ 1 ][ 0 ][ 0 ] )

            wt, var = css.create_input_seq( refseq,
                                            c_hap[ 0 ],
                                            c_hap[ 1 ],
                                            [],
                                            ( 0, 0 ),
                                            scored_context,
                                            rev_strand = rev_strand,
                                            unscored_context = 0, )

            _, snv2 = css.create_input_seq( refseq,
                                            c_hap[ 0 ],
                                            [],
                                            [],
                                            ( 0, 0 ),
                                            scored_context,
                                            rev_strand = rev_strand,
                                            unscored_context = 0, )

            _, snv1 = css.create_input_seq( refseq,
                                            c_hap[ 1 ][ 0 ],
                                            [],
                                            [],
                                            ( 0, 0 ),
                                            scored_context + abs( snv_dist ),
                                            rev_strand = rev_strand,
                                            unscored_context = 0, )

            wt_acc = locate_acceptors( wt )
            var_acc = locate_acceptors( var )
            snv1_acc = locate_acceptors( snv1 )
            snv2_acc = locate_acceptors( snv2 )

            cats.append( ' '.join( [ classify_gain_dnv( event[ 1 ] + c_hap[ 0 ][ 0 ],
                                                        exon_starts,
                                                        wt_acc[ scored_context + int( event[ 1 ] ) ],
                                                        snv1_acc[ scored_context + int( event[ 1 ] ) ],
                                                        snv2_acc[ scored_context + int( event[ 1 ] ) ],
                                                        var_acc[ scored_context + int( event[ 1 ] ) ],
                                                        c_hap[ 0 ][ 0 ],
                                                        c_hap[ 1 ][ 0 ][ 0 ],
                                                        distal_dist = distal_dist ),
                                                        event[ 0 ][ 3: ] ] ) )

        elif 'DS_DG' in event[ 0 ]:

            snv_dist = c_hap[ 0 ][ 0 ] - c_hap[ 1 ][ 0 ][ 0 ]

            wt, var = css.create_input_seq( refseq,
                                            c_hap[ 0 ],
                                            c_hap[ 1 ],
                                            [],
                                            ( 0, 0 ),
                                            scored_context,
                                            rev_strand = rev_strand,
                                            unscored_context = 0, )

            _, snv2 = css.create_input_seq( refseq,
                                            c_hap[ 0 ],
                                            [],
                                            [],
                                            ( 0, 0 ),
                                            scored_context,
                                            rev_strand = rev_strand,
                                            unscored_context = 0, )

            _, snv1 = css.create_input_seq( refseq,
                                            c_hap[ 1 ][ 0 ],
                                            [],
                                            [],
                                            ( 0, 0 ),
                                            scored_context + abs( snv_dist ),
                                            rev_strand = rev_strand,
                                            unscored_context = 0, )

            wt_don = locate_donors( wt )
            var_don = locate_donors( var )
            snv1_don = locate_donors( snv1 )
            snv2_don = locate_donors( snv2 )

            if var_don[ scored_context + int( event[ 1 ] ) ] == 1:

                cats.append( ' '.join( [ classify_gain_dnv( event[ 1 ] + c_hap[ 0 ][ 0 ],
                                                            exon_ends,
                                                            wt_don[ scored_context + int( event[ 1 ] ) ],
                                                            snv1_don[ scored_context + int( event[ 1 ] ) ],
                                                            snv2_don[ scored_context + int( event[ 1 ] ) ],
                                                            var_don[ scored_context + int( event[ 1 ] ) ],
                                                            c_hap[ 0 ][ 0 ],
                                                            c_hap[ 1 ][ 0 ][ 0 ],
                                                            distal_dist = distal_dist ),
                                                            event[ 0 ][ 3: ] ] ) )

            elif var_don[ scored_context + int( event[ 1 ] ) ] == 0:

                wt_gc_don = locate_donors( wt, donor_seq = 'GC' )
                snv1_gc_don = locate_donors( snv1, donor_seq = 'GC' )
                snv2_gc_don = locate_donors( snv2, donor_seq = 'GC' )
                var_gc_don = locate_donors( var, donor_seq = 'GC' )

                if var_gc_don[ scored_context + int( event[ 1 ] ) ] == 1:

                    cats.append( ' '.join( [ classify_gain_dnv( event[ 1 ] + c_hap[ 0 ][ 0 ],
                                                                exon_ends,
                                                                wt_gc_don[ scored_context + int( event[ 1 ] ) ],
                                                                snv1_gc_don[ scored_context + int( event[ 1 ] ) ],
                                                                snv2_gc_don[ scored_context + int( event[ 1 ] ) ],
                                                                var_gc_don[ scored_context + int( event[ 1 ] ) ],
                                                                c_hap[ 0 ][ 0 ],
                                                                c_hap[ 1 ][ 0 ][ 0 ],
                                                                distal_dist = distal_dist ),
                                                                'GC',
                                                                event[ 0 ][ 3: ] ] ) )

                elif var_gc_don[ scored_context + int( event[ 1 ] ) ] == 0:

                    cats.append( 'Unknown NC cryptic ' + event[ 0 ][ 3: ] )

    tbv[ out_col ] = cats

    return tbv


def process_SNVs( snvs_df,
                  annots_bed,
                  refseq,
                  gnomad2_df,
                  gnomad3_df,
                  exon,
                  splai_annots,
                  spliceai_col,
                  cat_out_col,
                  rev_strand,
                  common_af,
                  spliceai_thresh,
                  distal_dist ):

    snvs = snvs_df.copy()
    gnomad2 = gnomad2_df.copy()
    gnomad3 = gnomad3_df.copy()

    print( '%i SNVS input' % len( snvs ) )

    snvs = categorize_splai_snvs( snvs,
                                refseq,
                                splai_annots,
                                int( annots_bed.loc[ annots_bed.exon == exon ].exon_len ),
                                rev_strand = rev_strand,
                                pos_col = 'POS' + spliceai_col[ 2: ],
                                event_col = spliceai_col + '_type',
                                out_col = cat_out_col,
                                distal_dist = distal_dist )

    snvs[ 'hg19_pos' ] = snvs.pos
    snvs = hg19_to_hg38( snvs,
                         annots_bed,
                         exon )

    snvs = gn.merge_data_gnomad( snvs,
                                 gnomad2,
                                 indexcols = [ 'hg19_pos', 'ref', 'alt' ],
                                 suffix = 'v2' )

    snvs = gn.merge_data_gnomad( snvs,
                                 gnomad3,
                                 indexcols = [ 'hg38_pos', 'ref', 'alt' ],
                                 suffix = 'v3' )

    print( '%i SNVs exist in gnomAD' % ( ( snvs.af_v2.notnull() ) | ( snvs.af_v3.notnull() ) ).sum() )

    snvs = dich_at_thresh( snvs,
                           'af_v2',
                           common_af,
                           'ge',
                           'common_v2' )

    snvs = dich_at_thresh( snvs,
                           'af_v3',
                           common_af,
                           'ge',
                           'common_v3' )

    snvs[ 'common' ] = ( snvs.common_v2 ) | ( snvs.common_v3 )

    snvs = dich_at_thresh( snvs,
                           spliceai_col,
                           spliceai_thresh,
                           'ge',
                           'sdv' )

    if not rev_strand:

        snvs_ex = snvs.loc[ ( snvs.pos > int( annots_bed.loc[ annots_bed.exon == exon ].start_hg19 ) ) \
                            & ( snvs.pos <= int( annots_bed.loc[ annots_bed.exon == exon ].end_hg19 ) ) ]

    else:

        snvs_ex = snvs.loc[ ( snvs.pos >= int( annots_bed.loc[ annots_bed.exon == exon ].start_hg19 ) ) \
                            & ( snvs.pos < int( annots_bed.loc[ annots_bed.exon == exon ].end_hg19 ) ) ]

    print( '%i of SNVs are within exon %i ( %i expected )' %( len( snvs_ex),
                                                              exon,
                                                              annots_bed.loc[ annots_bed.exon == exon ].exon_len_hg19*3 ) )

    return ( snvs, snvs_ex )

def process_DNVs( dnvs_df,
                  annots_bed,
                  refseq,
                  exon,
                  spliceai_col,
                  splai_annots,
                  cat_out_col,
                  rev_strand,
                  spliceai_thresh,
                  distal_dist ):

    dnvs = dnvs_df.copy()

    print( '%i DNVS input' % len( dnvs ) )

    dnvs = categorize_splai_dnvs( dnvs,
                                refseq,
                                splai_annots,
                                int( annots_bed.loc[ annots_bed.exon == exon ].exon_len ),
                                rev_strand = rev_strand,
                                pos_col = 'POS' + spliceai_col[ 2: ],
                                event_col = spliceai_col + '_type',
                                out_col = cat_out_col,
                                distal_dist = distal_dist )

    dnvs = dnvs.rename( columns = { 'other_var': 'var_snv1',
                                    'pos': 'pos_snv2',
                                    'ref': 'ref_snv2',
                                    'alt': 'alt_snv2' } )
    dnvs[ 'var_snv2' ] = [ str( p ) + ':' + ra[ 0 ] + '>' + ra[ 1 ]
                           for p, ra in zip( dnvs.pos_snv2, zip( dnvs.ref_snv2, dnvs.alt_snv2 ) ) ]

    dnvs[ 'pos_snv1' ] = [ int( var.split( ':' )[ 0 ] ) for var in dnvs.var_snv1 ]
    dnvs[ 'ref_snv1' ] = [ var.split( ':' )[ 1 ].split( '>' )[ 0 ] for var in dnvs.var_snv1 ]
    dnvs[ 'alt_snv1' ] = [ var.split( ':' )[ 1 ].split( '>' )[ 1 ] for var in dnvs.var_snv1 ]

    dnvs[ 'hg19_pos_snv1' ] = dnvs.pos_snv1
    dnvs[ 'hg19_pos_snv2' ] = dnvs.pos_snv2
    dnvs = hg19_to_hg38( dnvs,
                         annots_bed,
                         exon,
                         'hg19_pos_snv1',
                         'hg38_pos_snv1' )
    dnvs = hg19_to_hg38( dnvs,
                         annots_bed,
                         exon,
                         'hg19_pos_snv2',
                         'hg38_pos_snv2' )

    dnvs = dich_at_thresh( dnvs,
                           spliceai_col,
                           spliceai_thresh,
                           'ge',
                           'sdv' )

    if not rev_strand:

        dnvs_ex = dnvs.loc[ ( ( dnvs.pos_snv1 > int( annots_bed.loc[ annots_bed.exon == exon ].start_hg19 ) ) \
                                & ( dnvs.pos_snv1 <= int( annots_bed.loc[ annots_bed.exon == exon ].end_hg19 ) ) ) \
                              & ( ( dnvs.pos_snv2 > int( annots_bed.loc[ annots_bed.exon == exon ].start_hg19 ) ) \
                                & ( dnvs.pos_snv2 <= int( annots_bed.loc[ annots_bed.exon == exon ].end_hg19 ) ) ) ]

    else:

        dnvs_ex = dnvs.loc[ ( ( dnvs.pos_snv1 >= int( annots_bed.loc[ annots_bed.exon == exon ].start_hg19 ) ) \
                                & ( dnvs.pos_snv1 < int( annots_bed.loc[ annots_bed.exon == exon ].end_hg19 ) ) ) \
                              & ( ( dnvs.pos_snv2 >= int( annots_bed.loc[ annots_bed.exon == exon ].start_hg19 ) ) \
                                & ( dnvs.pos_snv2 < int( annots_bed.loc[ annots_bed.exon == exon ].end_hg19 ) ) ) ]

    print( '%i of DNVs are within exon %i ( %.0f expected )' %( len( dnvs_ex ), exon,
                                                  ( ( 3*annots_bed.loc[ annots_bed.exon == exon ].exon_len_hg19*( 3*annots_bed.loc[ annots_bed.exon == exon ].exon_len_hg19 - 1 ) ) / 2 ) ) )

    return ( dnvs, dnvs_ex )

def epistasis_to_long( snvs,
                       dnvs,
                       ):

    long = pd.concat( [ snvs.copy(), dnvs.copy() ],
                      ignore_index = True )

    return long

def epistasis_to_wide( snvs,
                       dnvs ):

    s = snvs.copy()
    d = dnvs.copy()

    index_cols = [ 'pos_snv1', 'ref_snv1', 'alt_snv1', 'hg19_pos_snv1', 'hg38_pos_snv1' ]
    wide = d.set_index( index_cols ).merge( s.rename( columns = { 'pos': 'pos_snv1',
                                                                  'ref': 'ref_snv1',
                                                                  'alt': 'alt_snv1',
                                                                  'hg19_pos': 'hg19_pos_snv1',
                                                                  'hg38_pos': 'hg38_pos_snv1',
                                                                  'n_alt_v2': 'n_alt_v2_snv1',
                                                                  'n_allele_v2': 'n_allele_v2_snv1',
                                                                  'n_homo_v2': 'n_homo_v2_snv1',
                                                                  'af_v2': 'af_v2_snv1',
                                                                  'exon_v2': 'exon_v2_snv1',
                                                                  'common_v2': 'common_v2_snv1',
                                                                  'n_alt_v3': 'n_alt_v3_snv1',
                                                                  'n_allele_v3': 'n_allele_v3_snv1',
                                                                  'n_homo_v3': 'n_homo_v3_snv1',
                                                                  'af_v3': 'af_v3_snv1',
                                                                  'exon_v3': 'exon_v3_snv1',
                                                                  'common_v3': 'common_v3_snv1',
                                                                  'common': 'common_snv1' } ).set_index( index_cols ),
                                                          how = 'left',
                                                          left_index = True,
                                                          right_index = True,
                                                          suffixes = ( '', '_snv1' )
                                                         ).reset_index()

    index_cols = [ 'pos_snv2', 'ref_snv2', 'alt_snv2', 'hg19_pos_snv2', 'hg38_pos_snv2' ]
    wide = wide.set_index( index_cols ).merge( s.rename( columns = { 'pos': 'pos_snv2',
                                                                  'ref': 'ref_snv2',
                                                                  'alt': 'alt_snv2',
                                                                  'hg19_pos': 'hg19_pos_snv2',
                                                                  'hg38_pos': 'hg38_pos_snv2',
                                                                  'n_alt_v2': 'n_alt_v2_snv2',
                                                                  'n_allele_v2': 'n_allele_v2_snv2',
                                                                  'n_homo_v2': 'n_homo_v2_snv2',
                                                                  'af_v2': 'af_v2_snv2',
                                                                  'exon_v2': 'exon_v2_snv2',
                                                                  'common_v2': 'common_v2_snv2',
                                                                  'n_alt_v3': 'n_alt_v3_snv2',
                                                                  'n_allele_v3': 'n_allele_v3_snv2',
                                                                  'n_homo_v3': 'n_homo_v3_snv2',
                                                                  'af_v3': 'af_v3_snv2',
                                                                  'exon_v3': 'exon_v3_snv2',
                                                                  'common_v3': 'common_v3_snv2',
                                                                  'common': 'common_snv2' } ).set_index( index_cols ),
                                                          how = 'left',
                                                          left_index = True,
                                                          right_index = True,
                                                          suffixes = ( '', '_snv2' )
                                                         ).reset_index()

    assert len( wide ) > 0, 'Merging to wide failed'

    return wide

def process_epistasis( snvs_df,
                       dnvs_df,
                      annots_bed,
                      refseq,
                      gnomad2_df,
                      gnomad3_df,
                      exon,
                      splai_annots,
                      spliceai_col,
                      cat_out_col,
                      gene_name,
                      outdir,
                      rev_strand,
                      common_af,
                      spliceai_thresh,
                      distal_dist ):

    snvs = snvs_df.copy()
    dnvs = dnvs_df.copy()

    today = date.today()

    print( 'Processing SNVs...' )

    snvs, snvs_ex = process_SNVs( snvs,
                                  annots_bed,
                                  refseq,
                                  gnomad2_df,
                                  gnomad3_df,
                                  exon,
                                  splai_annots,
                                  spliceai_col,
                                  cat_out_col,
                                  rev_strand,
                                  common_af,
                                  spliceai_thresh,
                                  distal_dist )

    snvs.to_csv( outdir + '%s_ex%s_snvs.%s.txt' % ( gene_name, str( exon ), today.strftime("%Y-%m%d") ),
                 sep = '\t',
                 index = False )

    print( 'SNV processing complete' )

    print( 'Processing DNVs...')

    dnvs, dnvs_ex = process_DNVs( dnvs,
                                  annots_bed,
                                  refseq,
                                  exon,
                                  splai_annots,
                                  spliceai_col,
                                  cat_out_col,
                                  rev_strand = rev_strand,
                                  spliceai_thresh = spliceai_thresh,
                                  distal_dist = distal_dist )

    dnvs.to_csv( outdir + '%s_ex%s_dnvs.%s.txt' % ( gene_name, str( exon ), today.strftime("%Y-%m%d") ),
                 sep = '\t',
                 index = False )

    print( 'DNV processing complete' )

    long = epistasis_to_long( snvs_ex.rename( columns = { 'other_var': 'var_snv1',
                                                          'pos': 'pos_snv1',
                                                          'ref': 'ref_snv1',
                                                          'alt': 'alt_snv1' } ),
                               dnvs_ex )

    long[ 'snv' ] = long.pos_snv2.isnull()

    long.to_csv( outdir + '%s_ex%s_allvars_long.%s.txt' % ( gene_name, str( exon ), today.strftime("%Y-%m%d") ),
                 sep = '\t',
                 index = False )

    wide = epistasis_to_wide( snvs_ex,
                              dnvs_ex )

    wide.to_csv( outdir + '%s_ex%s_allvars_wide.%s.txt' % ( gene_name, str( exon ), today.strftime("%Y-%m%d") ),
                 sep = '\t',
                 index = False )

    return ( snvs, dnvs, long, wide )

def hg19_to_hg38( tbl_by_var,
                  annot,
                  exon,
                  hg19_col = 'hg19_pos',
                  hg38_col = 'hg38_pos' ):

    tbv = tbl_by_var.copy()

    assert int( annot.loc[ annot.exon == exon ].exon_len_hg19 ) == int( annot.loc[ annot.exon == exon ].exon_len_hg38 ), \
    'Exon lengths differ across genome assemblies'

    tbv[ hg38_col ] = tbv[ hg19_col ] + ( ( annot.loc[ annot.exon == exon ].start_hg38 - annot.loc[ annot.exon == exon ].start_hg19 ).values[ 0 ] )

    return tbv

def dich_at_thresh( tbl_by_var,
                    thresh_col,
                    thresh,
                    compare_op,
                    bool_col ):

    tbv = tbl_by_var.copy()

    assert compare_op.lower() in [ 'gt', 'ge', 'lt', 'le' ], 'Comparison operator must be one of: gt, ge, lt, le'

    if compare_op.lower() == 'gt':

        tbv[ bool_col ] = tbv[ thresh_col ] > thresh

    elif compare_op.lower() == 'ge':

        tbv[ bool_col ] = tbv[ thresh_col ] >= thresh

    elif compare_op.lower() == 'lt':

        tbv[ bool_col ] = tbv[ thresh_col ] < thresh

    elif compare_op.lower() == 'le':

        tbv[ bool_col ] = tbv[ thresh_col ] <= thresh

    if compare_op.lower()[ 0 ] == 'g':

        print( '%i rows have %s above %.3f' % ( tbv[ bool_col ].sum(), thresh_col, thresh ) )

    elif compare_op.lower()[ 0 ] == 'l':

        print( '%i rows have %s below %.3f' % ( tbv[ bool_col ].sum(), thresh_col, thresh ) )

    return tbv

def Epistasis_Main( annots_bed,
                    exon,
                    refseq,
                    splai_annots,
                    models,
                    ref_name,
                    chrom,
                    gnomad2_df,
                    gnomad3_df,
                    gene_name,
                    date,
                    outdir,
                    spliceai_col = 'DS_maxm',
                    cat_out_col = 'splai_cat',
                    mask_value = 0,
                    rev_strand = False,
                    common_af = .01,
                    spliceai_thresh = .2,
                    distal_dist = 20 ):

    ex_len = int( annots_bed.loc[ annots_bed.exon == exon ].exon_len_hg19 )

    ex_snvs, ex_dnvs = create_all_SNVs_DNVs( annots_bed,
                                             exon,
                                             refseq )

    print( 'Computed %i SNVs ( %i expected )' % ( len( ex_snvs ), ex_len*3 ) )
    print( 'Computed %i DNVs ( %i expected )' % ( len( ex_dnvs[ 0 ] ), ( ex_len*3*( ( ex_len - 1 )*3 ) ) / 2 ) )

    snvs_scored, dnvs_scored = splai_score_SNVs_DNVs( annots_bed,
                                                      splai_annots,
                                                      models,
                                                      refseq,
                                                      ref_name,
                                                      chrom,
                                                      exon,
                                                      ex_snvs,
                                                      ex_dnvs,
                                                      mask_value = 0,
                                                      rev_strand = False )

    snv, dnv, long, wide = process_epistasis( snvs_scored,
                                              dnvs_scored,
                                              annots_bed,
                                              refseq,
                                              gnomad2_df,
                                              gnomad3_df,
                                              exon,
                                              splai_annots,
                                              spliceai_col,
                                              gene_name,
                                              outdir,
                                              rev_strand,
                                              common_af,
                                              spliceai_thresh,
                                              distal_dist )


    return snv, dnv, long, wide
