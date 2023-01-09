import pysam
import os
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import copy
import itertools
import pybedtools as pbt
from ast import literal_eval
import time
import splanl.merge_bcs as mbcs
import splanl.plots as sp

def adjust_junctions(pysam_align,
                    refseq,
                    cnst_exons,
                    outfile):

    for bc, _reads in itertools.groupby( pysam_align, lambda _r: _r.get_tag( 'RX' ) ):

        reads = list(_reads)

        for read in reads:

            iso = read.get_blocks()

            #don't check junctions for unmapped reads or if there's only one block
            if read.is_unmapped or len( iso )==1:
                outfile.write( read )
                continue

            #if all of the ends of junctions are before (downstream) the end of the downstream constant exon
            #OR all of the starts of junctions are after (upstream) the start of the upstream constant exon
            #the read is garbage anyway so just skip it
            if all( jn[1] < cnst_exons[0][1] for jn in iso ) or all( jn[0] > cnst_exons[1][0] for jn in iso ):
                outfile.write( read )
                continue

            #check if the end of any junction is more upstream than the expected end of the upstream exon
            #check if the end of any junction is at the expected end of the upstream exon
            if any( jn[1] < cnst_exons[0][1] for jn in iso ) and not( any( jn[1] == cnst_exons[0][1] for jn in iso ) ):
                #grabs the last junction which is less than the expected end of the constant exon
                bad_jn = [ jn for jn in iso if jn[1] < cnst_exons[0][1] ][-1]
                #grabs junction following the screwed up junction
                next_jn = iso[ iso.index(bad_jn)+1 ]

                #if the next exonic region is smaller than the difference from the bad to expected junction
                #write the read as is and move to next read
                if (next_jn[1] - next_jn[0]) < (cnst_exons[0][1] - bad_jn[1]):
                    outfile.write( read )
                    continue

                #if its not also check if the suffix of whats missing from the constant exon matches the prefix of the next junction
                if refseq[ bad_jn[1]:cnst_exons[0][1] ] == refseq[ next_jn[0]:next_jn[0] + ( cnst_exons[0][1] - bad_jn[1] ) ]:
                    cigarlist = [ list(tup) for tup in read.cigartuples ]
                    #gets indices that are a match and need editing
                    match_idx = [ idx for idx, tup in enumerate( read.cigartuples ) if tup[0] == 0 ]
                    bad_jn_idx = match_idx[ iso.index(bad_jn) ]

                    assert len(match_idx) > iso.index(bad_jn), bc+'\nNo matched segments after bad junction'
                    next_jn_idx = match_idx[ iso.index(bad_jn) + 1 ]

                    #adds bases back to the bad junction to force it to end at the expected constant exon end
                    cigarlist[ bad_jn_idx ][1]+=cnst_exons[0][1] - bad_jn[1]
                    #subtract bases from the next matched segment
                    cigarlist[ next_jn_idx ][1]-=cnst_exons[0][1] - bad_jn[1]
                    read.cigartuples = tuple( tuple(l) for l in cigarlist )

            #check if the start of any junction is farther upstream than the expected start of the constant downstream exon
            #check if the start of any junction is at the expected start of the constant downstream exon
            elif any( jn[0] > cnst_exons[1][0] for jn in iso ) and not( any( jn[0] == cnst_exons[1][0] for jn in iso ) ):
                #grabs the first junction which is less than the expected end of the constant exon
                bad_jn = [ jn for jn in iso if jn[0] > cnst_exons[1][0] ][0]
                #grabs junction preceeding the screwed up junction
                prev_jn = iso[ iso.index(bad_jn)-1 ]

                #if the previous exonic region is smaller than the difference from the bad to expected junction
                #write the read as is and move to next read
                if (prev_jn[1] - prev_jn[0]) < (bad_jn[0] - cnst_exons[1][0]):
                    outfile.write( read )
                    continue

                #finally check if the prefix lost on the constant upstream matches the suffix on the previous junction
                if refseq[ cnst_exons[1][0]:bad_jn[0] ] == refseq[ prev_jn[1]-( bad_jn[0] - cnst_exons[1][0] ):prev_jn[1] ]:
                    cigarlist = [ list(tup) for tup in read.cigartuples ]
                    #gets indices that are a match and need editing
                    match_idx = [ idx for idx, tup in enumerate( read.cigartuples ) if tup[0] == 0 ]
                    bad_jn_idx = match_idx[ iso.index(bad_jn) ]

                    assert iso.index(bad_jn) > 0, bc+'\nNo matched segments before bad junction'

                    prev_jn_idx = match_idx[ iso.index(bad_jn) - 1 ]

                    #adds bases back to the bad junction to force it to start at the expected constant exon start
                    cigarlist[ bad_jn_idx ][1]+=bad_jn[0] - cnst_exons[1][0]
                    #removes bases from the previous matched segment
                    cigarlist[ prev_jn_idx ][1]-=bad_jn[0] - cnst_exons[1][0]
                    read.cigartuples = tuple( tuple(l) for l in cigarlist )

            #sanity check that we didn't alter the length of the read
            assert ( read.infer_query_length() == read.query_length ), bc+'\nWrong read length - invalid cigar'

            outfile.write( read )

    #make sure to flush the buffer!
    outfile.close()


def clean_jns( jns,
               cnst_exons,
               tol = 3,
               min_matches = 120 ):
    """Removes any junctions within the constant exons,
    joins adjacent junctions,
    and removes the last junction if its <= 3 bp long

    Args:
        jns (list of tuples): list of tuples from a PySam read_blocks() function call
        cnst_exons (list of tuples): coords of known constant exons

    Returns:
        jns_joined (list of tuples): same list of tuples as input with
        junctions within the constant exons removed
    """
    jns_joined = []

    prev_jn = None
    #used to check if downstream constant in read
    ds_cnst = soft_clipped = False

    #screens out reads with too many bases soft_clipped
    if sum( jn[1] - jn[0] + 1 for jn in jns ) < min_matches:
        soft_clipped = True

    #handles the case in which the read doesn't make it to the downstream constant
    #empty junctions implies the read was unmapped - also let that pass
    if jns == [] or jns[-1][1] < cnst_exons[1][0]:
        ds_cnst = True

    for _jn in jns:

        if not( ds_cnst ) and ( cnst_exons[1][0] - tol ) <= _jn[0] <= ( cnst_exons[1][0] + tol ):
            ds_cnst = True

        #changes to 1-based inclusive numbering
        jn = ( _jn[0] + 1, _jn[1] )

        if jn[0] >= cnst_exons[0][1] and jn[1] < ( cnst_exons[1][0]+1 ):
            jns_joined.append(jn)
            cur_jn = jn
            #joins adjacent junctions
            if prev_jn:
                if cur_jn[0] == prev_jn[1]+1:
                    jns_joined[-2:]=[ (prev_jn[0], cur_jn[1]) ]
            prev_jn=cur_jn

    #removes the last junction if its less than 3 bp long
    #hard to align a so few reads accurately so ignore them
    if len(jns_joined)>0 and ( jns_joined[-1][1] - jns_joined[-1][0] ) <= tol:
        jns_joined = jns_joined[:-1]

    jns_joined = tuple( jns_joined )

    if not ds_cnst:
        jns_joined = None
    elif soft_clipped:
        jns_joined = 0

    return jns_joined

def clean_jns_pe( read1,
                  read2,
                  cnst_exons,
                  spl_tol,
                  indel_tol,
                  min_matches_for,
                  min_matches_rev ):

    #would this be faster if I zipped two aligned files together?

    if read1.get_blocks()[ 0 ][ 0 ] > read2.get_blocks()[ 0 ][ 0 ]:
        temp_r1 = read1
        read1 = read2
        read2 = temp_r1

    r12_jns = [ read1.get_blocks(),read2.get_blocks() ]
    r12_cigs = [ read1.cigartuples, read2.cigartuples ]

    #if one of the reads is unmapped skip the pair
    if not all( r12_jns ):
        return 'unmapped'

    r12_jns = [ list( r12_jns[ 0 ] ), list( r12_jns[ 1 ] ) ]

    us_cnst = ds_cnst = False

    #screens out reads with too many bases soft_clipped
    if sum( jn[ 1 ] - jn[ 0 ] + 1 for jn in r12_jns[ 0 ] ) < min_matches_for \
    or sum( jn[ 1 ] - jn[ 0 ] + 1 for jn in r12_jns[ 1 ] ) < min_matches_rev:
        return 'soft_clipped'

    #this section will collapse any small indels
    for i,r_cig in enumerate( r12_cigs ):

        #there are no indels - all is well
        if not any( c_o == 1 or c_o == 2  for c_o,c_l in r_cig ):
            continue

        ri_jn = []
        j = 0
        indel = False

        for c_o,c_l in r_cig:

            if indel:
                indel = False
                continue

            #if its a match add to append jn
            #don't add any very short alignments since those are unreliable
            if c_o == 0 and c_l >= spl_tol:
                ri_jn.append( r12_jns[ i ][ j ] )
                j += 1
            #if its an indel within tolerance, add it to previous junction
            elif ri_jn and ( c_o == 1 or c_o == 2 ) and c_l <= indel_tol:
                indel = True
                ri_jn[ -1 ] = ( ri_jn[ -1 ][ 0 ], r12_jns[ i ][ j ][ 1 ] )
                j += 1

        r12_jns[ i ] = ri_jn

    #this section combines the reads
    #if the reads don't overlap combine them
    #so ( 1267, 1312 ), ( 1480, 1512 ) becomes ( 1267, 1512 )
    if r12_jns[ 0 ][ -1 ][ 1 ] <= r12_jns[ 1 ][ 0 ][ 0 ]:
        r12_jns[ 0 ][ -1 ] = ( r12_jns[ 0 ][ -1 ][ 0 ], r12_jns[ 1 ][ 0 ][ 1 ] )
        r12_jns = tuple( sorted( r12_jns[ 0 ] + r12_jns[ 1 ][ 1: ] ) )

    else:
        r12_jns = tuple( sorted( r12_jns[ 0 ] + r12_jns[ 1 ] ) )

    r12_jn_comb = []
    for _jn in r12_jns:

        #checks if read splices properly from upstream constant donor
        if _jn[ 1 ] >= cnst_exons[ 0 ][ 1 ] - spl_tol and _jn[ 1 ] <= cnst_exons[ 0 ][ 1 ] + spl_tol:
            us_cnst = True
            continue

        #checks if read splices properly into the downstream constant acceptor
        if _jn[ 0 ] >= cnst_exons[ 1 ][ 0 ] - spl_tol and _jn[ 0 ] <= cnst_exons[ 1 ][ 0 ] + spl_tol:
            ds_cnst = True
            continue

        #changes to 1-based inclusive numbering
        jn = ( _jn[ 0 ] + 1, _jn[ 1 ] )

        #if the jn is outside of the constant exons
        if jn[ 0 ] >= cnst_exons[ 0 ][ 1 ] and jn[ 1 ] < cnst_exons[ 1 ][ 0 ] + 1:

            if not r12_jn_comb or jn[ 0 ] > r12_jn_comb[ -1 ][ 1 ]:
                r12_jn_comb.append( jn )
            #if the forward and reverse reads cross - combine them
            #so ( 1267, 1312 ), ( 1297, 1512 ) becomes ( 1267, 1512 )
            else:
                r12_jn_comb[ -1 ] = ( r12_jn_comb[ -1 ][ 0 ], jn[ 1 ] )

    r12_jn_comb = tuple( r12_jn_comb )

    if not ds_cnst:
        r12_jn_comb = 'bad_ends'
    elif not us_cnst:
        r12_jn_comb = 'bad_starts'

    return r12_jn_comb

def get_all_isoforms( align_by_samp_dict,
                      cnst_exons = None,
                      tol = 3,
                      min_matches = 120 ):
    """Gets all isoforms within a pysam alignment file

    Args:
        align_dict (dictionary): dictionary with sample names as keys and pysam alignment files as values

    Returns:
        out_tbls (dictionary): dictionary with sample names as keys and pandas dataframes with isoform tuples as the
        index and number of reads representing each isoform as the only column
    """
    out_tbls = {}

    for samp in align_by_samp_dict:

        print(samp)

        if cnst_exons:
            #creates a counter object of all of the exon junctions
            all_isos_cnt = Counter( [ clean_jns( read.get_blocks(),
                                                 cnst_exons,
                                                 tol,
                                                 min_matches ) for read in align_by_samp_dict[ samp ] ] )
        else:
            all_isos_cnt = Counter( [ tuple( read.get_blocks() ) for read in align_by_samp_dict[ samp ] ] )

        #removes None type from counter - represents reads not mapping to the downstream constant
        del all_isos_cnt[None]
        #removes 0 from counter - represents reads with not enough matches
        del all_isos_cnt[0]

        iso_df = pd.DataFrame.from_dict( all_isos_cnt, orient='index' ).reset_index()
        iso_df = iso_df.rename(columns={'index':'isoform', 0:'read_count'}).sort_values(by='read_count', ascending=False)
        iso_df = iso_df.set_index('isoform')

        out_tbls[ samp ] = iso_df

    return( out_tbls )

def get_all_isoforms_pe(  pysam_align_file,
                          cnst_exons,
                          spl_tol = 3,
                          indel_tol = 20,
                          min_matches_for = 70,
                          min_matches_rev = 50,
                          softclip_outbam = None ):
    """Gets all isoforms within a pysam alignment file

    Args:
        align_dict (dictionary): dictionary with sample names as keys and pysam alignment files as values

    Returns:
        out_tbls (dictionary): dictionary with sample names as keys and pandas dataframes with isoform tuples as the
        index and number of reads representing each isoform as the only column
    """

    jns = []

    if softclip_outbam:
        softclip = pysam.AlignmentFile( softclip_outbam, "wb", template = pysam_align_file )

    for read in pysam_align_file:

        #if read.is_secondary or not read.is_paired or read.is_unmapped or read.mate_is_unmapped:
        if read.is_secondary:
            jns.append( 'secondary' )
            continue

        if not read.is_paired:
            jns.append( 'unpaired' )
            continue

        if read.is_unmapped or read.mate_is_unmapped:
            jns.append( 'unmapped' )
            continue

        if read.is_read2:
            read2 = read
        else:
            read1 = read
            read2 = None

        if read1 and read2 and read1.query_name == read2.query_name:

            jns.append( clean_jns_pe( read1,
                                      read2,
                                      cnst_exons,
                                      spl_tol,
                                      indel_tol,
                                      min_matches_for,
                                      min_matches_rev  ) )

            if softclip_outbam and jns[ -1 ] == 'soft_clipped':
                softclip.write( read1 )

    if softclip_outbam:
        softclip.close()

    pysam_align_file.close()

    all_isos_cnt = Counter( jns )

    #removes None type from counter - reads with pair unmapped
    #del all_isos_cnt[ -3 ]
    #removes 0 from counter - represents reads with not enough matches
    #del all_isos_cnt[ 0 ]
    #removes None type from counter - represents reads not mapping to the downstream constant
    #del all_isos_cnt[ -1 ]
    #removes 0 from counter - - represents reads not mapping to the upstream constant
    #del all_isos_cnt[ -2 ]

    iso_df = pd.DataFrame.from_dict( all_isos_cnt,
                                     orient = 'index' ).reset_index()

    iso_df = iso_df.rename( columns = { 'index': 'isoform',
                                         0: 'read_count' } ).sort_values( by = 'read_count',
                                                                          ascending = False )
    iso_df = iso_df.set_index( 'isoform' )

    return iso_df

def number_and_merge_isoforms( isodf_by_samp_dict,
                                existing_iso_df = None ):

    if existing_iso_df:
        cur_df = existing_iso_df.copy()

    out_tbl = { samp+'_read_count': [] for samp in isodf_by_samp_dict }

    out_tbl[ 'isoform' ] = list( set ( [ iso for samp in isodf_by_samp_dict
                                              for iso in isodf_by_samp_dict[ samp ].index ] ) )

    #this if/else not tested
    if existing_iso_df:
        out_tbl[ 'isonum' ] = []

        non_matches = 0
        for iso in out_tbl[ 'isoform' ]:

            if iso in cur_df.isoform:
                out_tbl[ 'isonum' ].append( cur_df.index[ cur_df.isoform == iso ] )
            else:
                out_tbl[ 'isonum' ].append( 'iso'+str( cur_df.shape[0] ) )
                non_matches += 1
    else:
        #number all isoforms so they are of the form iso0001
        out_tbl[ 'isonum' ] = [ 'iso'+str(i).zfill( len( str( len( out_tbl[ 'isoform' ] ) ) ) )
                                for i in range( len( out_tbl[ 'isoform' ] ) ) ]

    for samp in isodf_by_samp_dict:

        print(samp)

        #set all the read counts as zero
        out_tbl[ samp+'_read_count' ] = [ 0 for i in range( len( out_tbl[ 'isoform' ] ) ) ]

        for idx, iso in enumerate( isodf_by_samp_dict[ samp ].index ):

            out_tbl[ samp+'_read_count' ][ out_tbl[ 'isoform' ].index( iso ) ] = isodf_by_samp_dict[ samp ].iloc[ idx ].read_count

    out_tbl = pd.DataFrame( out_tbl )

    #reorder columns so isoform is first column
    ordered_col = ['isoform'] + [ col for col in out_tbl.columns if col != 'isoform' ]
    out_tbl = out_tbl[ ordered_col ]

    out_tbl = out_tbl.set_index('isonum')

    return out_tbl

def check_tolerance(var_pos,
                    unique_jns,
                    tol):

    within_tol = False

    for jn in unique_jns:
        if var_pos >= ( jn - tol ) and var_pos <= ( jn + tol ):
            within_tol = True
            break

    return within_tol

def cluster_vars(var_bc_counts,
                min_cluster_size,
                tol,
                clustered_bc_ave):

    clusters = {}
    i = 0

    for variants in var_bc_counts:
        for var in variants.split(','):
            var_pos =  int( var.split(':')[1] )
            if not clusters:
                clusters[ 'cluster'+str(i) ] = [ (var_pos,), var_bc_counts[ variants ] ]
                i+=1
                continue
            for cluster, info in clusters.items():
                var_clustered = False
                #if the variant is nearby first variant in cluster add to cluster
                if check_tolerance( var_pos, [ info[0][0] ], tol):
                    clusters[ cluster ][0]+= ( var_pos, )
                    clusters[ cluster ][1] += var_bc_counts[ variants ]
                    var_clustered = True
                    #if we've reached the minimum cluster count and the average bc count is at threshold exit
                    if len( clusters[ cluster ][0] ) == min_cluster_size and \
                    ( clusters[ cluster ][1] / len( clusters[ cluster ][0] ) ) >=  clustered_bc_ave:
                        return( True )
                    #either way we clustered the variant so break outta the loop
                    break
                #otherwise make a new cluster for it
            if not var_clustered:
                clusters[ 'cluster'+str(i) ] = [ (var_pos,), var_bc_counts[ variants ] ]
                i+=1

    return( False )

def append_unmapped( isos_df,
                     unmapped_m1,
                     unmapped_m2 = None ):

    isos = isos_df.copy()

    m1_reads = sum( 1 for entry in unmapped_m1 )

    if unmapped_m2 is not None:

        m2_reads = sum( 1 for entry in unmapped_m2 )

    else:

        m2_reads = 0

    unmapped = max( m1_reads, m2_reads )

    isos.loc[ 'unmapped' ] = unmapped

    unmapped_m1.close()

    if unmapped_m2 is not None:
        unmapped_m2.close()

    return isos

def summarize_isos_by_var_bc(align_by_samp_dict,
                             cnst_exons,
                                satbl,
                                iso_df,
                                unique_jns,
                                canonical_isos,
                                exising_iso_stats_df=None,
                                print_count=5,
                                min_maxbc_count=100,
                                tol=10,
                                min_cluster_size=3,
                                clustered_bc_ave=3,
                                min_bc_max_reads=(2,2)):
    """Gets counts for number of variants and barcodes per isoform. Prints variants with the most number of barcodes.

    Args:
        pysam_align (PySam AlignmentFile): a tag sorted pysam alignment file with all of the reads for the sample
        cnst_exons (list of tuples): coords of known constant exons
        satbl (pandas dataframe): subassembly dataframe which maps barcodes to variants
                                    Indexed on readgroupid
        iso_df (pandas dataframe): dataframe of isoforms, isoform groups, and read counts created from get_all_isoforms
                                    and create_isogrps
                                    Indexed on isogrp
        print_count (int): number of desired variants to print out per isoform - program will print the variants
                            with the highest number of barcodes

    Returns:
        iso_df (pandas dataframe): Original iso_df with the addition of number of variants per isoform and number
                                    of barcodes per isoform
    """

    if exising_iso_stats_df:
        cur_df = exising_iso_stats_df.copy()

    out_tbl = iso_df.copy()
    satbl_c = satbl.copy()

    satbl_c = satbl_c.dropna(subset=['variant_list'])

    for samp in align_by_samp_dict:

        print(samp)

        isogrp_to_bc = {}

        bc_cnt, read_cnt = 0,0

        for bc, _reads in itertools.groupby( align_by_samp_dict[ samp ], lambda _r: _r.get_tag( 'RX' ) ):

            if bc not in satbl_c.index:
                continue

            for iso, reads in itertools.groupby( _reads, lambda _r: clean_jns( _r.get_blocks(), cnst_exons ) ):

                #ignore read if it doesn't splice to downtstream constant or has too few matches
                if iso == None or iso == 0:
                    continue

                r = list( reads )

                isogrp = out_tbl.loc[out_tbl['isoform']==iso].index

                assert len(isogrp)==1, 'The isoform matches '+str( len( isogrp ) )+' isogroups instead of the expected 1'

                isogrp=isogrp[0]

                if isogrp not in isogrp_to_bc:
                    isogrp_to_bc[ isogrp ]={}

                if bc not in isogrp_to_bc[ isogrp ]:
                    isogrp_to_bc[ isogrp ][ bc ] = len(r)
                else:
                    isogrp_to_bc[ isogrp ][ bc ] += len(r)

                read_cnt+=len(r)

            bc_cnt+=1

            if bc_cnt%10000==0:
                print('Barcodes processed:',bc_cnt,'Reads processed:',read_cnt)

    #takes 16 minutes to get here
        isogrp_var_count = { isogrp: [ len( isogrp_to_bc[ isogrp ] ),
                                    #should count number of variants
                                    len( set( satbl_c.loc[ bc ].variant_list
                                    for bc in isogrp_to_bc[ isogrp ] ) ),
                                    max( isogrp_to_bc[ isogrp ].values() )
                                  ]
                        for isogrp in isogrp_to_bc
                        }

        chucked_bcs, chucked_reads = 0,0
        missed_bcs, missed_reads = 0,0

        for isogrp in isogrp_to_bc:

            bcs = [ bc for bc in isogrp_to_bc[ isogrp ] ]

            var_to_bc = {}
            for bc in bcs:
                var = satbl_c.loc[ bc ].variant_list
                if var in var_to_bc:
                    var_to_bc[ var ].add(bc)
                else:
                    var_to_bc[ var ] = { bc }

            var_bc_counts = { var: len(var_to_bc[ var ]) for var in var_to_bc }

            var_bc_counts_sort = { var: count for var, count in sorted(var_bc_counts.items(), key=lambda item: -item[1])}

            #get the variant with the highest number of barcodes
            top_var = list( itertools.islice( var_bc_counts_sort.items(), 1 ) ) [0]

            #get max bcs from one variant for each isogrp
            isogrp_var_count[ isogrp ].append( top_var[1] )

            print( 'For isoform:', out_tbl.loc[ isogrp ].isoform )
            print( 'The variants with the top', print_count, 'number of barcodes are:')
            print( list(itertools.islice( var_bc_counts_sort.items(), print_count ) ) )

            #get a list of all variants
            var_list = [ v for var in var_bc_counts_sort for v in var.split(',') ]
            var_list_unique = list( set( var_list ) )

            #this gets the first position where the isoform differs from expected junctions
            jn_diff = list( { jn for jns in out_tbl.loc[isogrp].isoform for jn in jns }
                        - set( unique_jns ) )
            if len( jn_diff )>1:
                jn_diff = [ min( jn_diff ) ]

            #lets keep the best isoforms
            #first check if the isoform exactly matches an isoform we're expecting to see
            if any( [ out_tbl.loc[ isogrp ].isoform == can_iso for can_iso in canonical_isos ] ):
                isogrp_var_count[ isogrp ].append( 1 )
            #then check if the max bc per isoform is greater than 100
            elif top_var[1] >= min_maxbc_count:
                isogrp_var_count[ isogrp ].append( 2 )
            #requires the top variant to have at least x barcodes and the top barcode to have at least y reads
            elif top_var[1] <= min_bc_max_reads[0] and isogrp_var_count[ isogrp ][2] <= min_bc_max_reads[1]:
                chucked_bcs += isogrp_var_count[ isogrp ][0]
                chucked_reads += out_tbl.loc[ isogrp ][ samp+'_read_count' ]
                isogrp_var_count[ isogrp ].append( 0 )
                continue
            #check if the var with the max bc is within tolerance of the where the isoform differs
            elif any( [ check_tolerance( int( var.split(':')[1] ),
                                 jn_diff,
                                 tol )
                    for var in top_var[0].split(',') ] ):
                isogrp_var_count[ isogrp ].append( 3 )
            #check if at least min_cluster_size of variants are within a tol of each other
            elif cluster_vars(  var_bc_counts_sort,
                            min_cluster_size,
                            tol,
                            clustered_bc_ave ):
                isogrp_var_count[ isogrp ].append( 4 )
            #check if a double variant is also listed as a single variant
            elif len( var_list ) != len( var_list_unique ):
                isogrp_var_count[ isogrp ].append( 5 )
            else:
                isogrp_var_count[ isogrp ].append( 0 )
                missed_bcs += isogrp_var_count[ isogrp ][0]
                missed_reads += out_tbl.loc[ isogrp ][ samp+'_read_count' ]

        print('%i (%.2f%%) barcodes failed the min_bc_max_reads filter' \
            % ( chucked_bcs, 100*(chucked_bcs/sum(i[0] for i in isogrp_var_count.values() ) ) ) )
        print('%i (%.2f%%) reads failed the min_bc_max_reads filter' \
            % ( chucked_reads, 100*(chucked_reads/out_tbl[ samp+'_read_count' ].sum() ) ) )
        print('%i (%.2f%%) barcodes did not fulfill any filter' \
            % ( missed_bcs, 100*(missed_bcs/sum(i[0] for i in isogrp_var_count.values() ) ) ) )
        print('%i (%.2f%%) reads did not fulfill any filter'\
            % ( missed_reads, 100*(missed_reads/out_tbl[ samp+'_read_count' ].sum() ) ) )

        cols = [ samp + suffix for suffix in ['_num_bcs','_num_vars','_max_reads_per_bc','_max_bc_per_var','_filter'] ]
        iso_cnts = pd.DataFrame.from_dict(isogrp_var_count, orient='index',
                                    columns=cols)

        out_tbl = pd.merge(out_tbl, iso_cnts, left_index=True, right_index=True, how="outer")

        out_tbl.index.name = 'isonum'

    #if iso_counts doesn't contain all the isoforms, the values will be null for those rows
    #change to zeros
    out_tbl = out_tbl.fillna(0)

    #get totals across samples
    for suffix in [ '_read_count', '_num_bcs', '_num_vars' ]:

        #sum of each row across the columns
        col = [ col for col in out_tbl.columns if suffix in col ]
        out_tbl[ 'total'+suffix ] = out_tbl[ col ].sum( axis=1 )

    #counts number of samples passing filter for each isoform
    #specifically, counting non zero filter values
    filter_col = [ col for col in out_tbl.columns if 'filter' in col ]
    out_tbl[ 'total_passfilt' ] = out_tbl[ filter_col ].astype( bool ).sum( axis=1 )

    if exising_iso_stats_df:
        out_tbl = pd.merge( cur_df, out_tbl, left_index=True, right_index=True, how="outer" )
        out_tbl = out_tbl.fillna(0)

    return(out_tbl)

def summarize_isos_by_var_bc_pe( align_by_samp_dict,
                                 cnst_exons,
                                satbl,
                                iso_df,
                                unique_jns,
                                canonical_isos,
                                spl_tol = 3,
                                min_matches_for = 70,
                                min_matches_rev = 50,
                                indel_tol = 20,
                                exising_iso_stats_df=None,
                                print_count=5,
                                min_maxbc_count=100,
                                tol = 10,
                                min_cluster_size=3,
                                clustered_bc_ave=3,
                                min_bc_max_reads=(2,2),
                                bc_tag = 'RX', ):
    """Gets counts for number of variants and barcodes per isoform. Prints variants with the most number of barcodes.

    Args:
        pysam_align (PySam AlignmentFile): a tag sorted pysam alignment file with all of the reads for the sample
        cnst_exons (list of tuples): coords of known constant exons
        satbl (pandas dataframe): subassembly dataframe which maps barcodes to variants
                                    Indexed on readgroupid
        iso_df (pandas dataframe): dataframe of isoforms, isoform groups, and read counts created from get_all_isoforms
                                    and create_isogrps
                                    Indexed on isogrp
        print_count (int): number of desired variants to print out per isoform - program will print the variants
                            with the highest number of barcodes

    Returns:
        iso_df (pandas dataframe): Original iso_df with the addition of number of variants per isoform and number
                                    of barcodes per isoform
    """

    if exising_iso_stats_df:
        cur_df = exising_iso_stats_df.copy()

    out_tbl = iso_df.copy()
    satbl_c = satbl.copy()

    satbl_c = satbl_c.dropna( subset = ['variant_list'] )

    out_tbl_d = out_tbl.reset_index().set_index( 'isoform' ).to_dict( 'index' )

    for samp in align_by_samp_dict:

        print(samp)

        unmapped = 0
        unpaired = 0
        secondary_reads = 0
        soft_clipped = 0
        bad_starts = 0
        bad_ends = 0
        bc_not_in_rna = 0
        reads_non_rna_bc = 0

        isogrp_to_bc = {}

        bc_cnt, read_cnt = 0,0

        for bc, _reads in itertools.groupby( align_by_samp_dict[ samp ], lambda _r: _r.get_tag( bc_tag ) ):

            reads = list( _reads )

            if bc not in satbl_c.index:
                bc_not_in_rna += 1
                bc_cnt += 1
                reads_non_rna_bc += sum( 1 for r in reads ) / 2
                read_cnt += sum( 1 for r in reads ) / 2
                continue

            for read in reads:

                if read.is_secondary:
                    secondary_reads += 1
                    continue

                elif read.is_unmapped:
                    unmapped += 1
                    continue

                elif not read.is_paired:
                    unpaired += 1
                    continue

                if read.is_read2:
                    read2 = read
                else:
                    read1 = read
                    read2 = None

                if read1 and read2 and read1.query_name == read2.query_name:

                    iso = clean_jns_pe( read1,
                                        read2,
                                        cnst_exons,
                                        spl_tol,
                                        indel_tol,
                                        min_matches_for,
                                        min_matches_rev )

                else:
                    continue

                #ignore read if it doesn't splice to downtstream constant or has too few matches
                if iso == 'unmapped':
                    unmapped += 1
                    #continue
                elif iso == 'bad_starts':
                    bad_starts += 1
                    #continue
                elif iso == 'bad_ends':
                    bad_ends += 1
                    #continue
                elif iso == 'soft_clipped':
                    soft_clipped += 1
                    #continue

                isogrp = out_tbl_d[ iso ][ 'isonum' ]

                if bc in satbl_c.index:

                    if isogrp not in isogrp_to_bc:
                        isogrp_to_bc[ isogrp ] = {}

                    if bc not in isogrp_to_bc[ isogrp ]:
                        isogrp_to_bc[ isogrp ][ bc ] = 1
                    else:
                        isogrp_to_bc[ isogrp ][ bc ] += 1

                read_cnt += 1

            bc_cnt += 1

            if bc_cnt % 1000 == 0:
                print( 'Barcodes processed: %i\nReads processed: %i' % ( bc_cnt, read_cnt ) )

        #takes 16 minutes to get here
        isogrp_var_count = { isogrp: [ len( isogrp_to_bc[ isogrp ] ),
                                    #should count number of variants
                                    len( set( satbl_c.loc[ bc ].variant_list
                                    for bc in isogrp_to_bc[ isogrp ] ) ),
                                    max( isogrp_to_bc[ isogrp ].values() ),
                                    sum( isogrp_to_bc[ isogrp ].values() )
                                  ]
                        for isogrp in isogrp_to_bc
                        }

        chucked_bcs, chucked_reads = 0,0
        missed_bcs, missed_reads = 0,0

        for isogrp in isogrp_to_bc:

            bcs = [ bc for bc in isogrp_to_bc[ isogrp ] ]

            var_to_bc = {}
            for bc in bcs:
                var = satbl_c.loc[ bc ].variant_list
                if var in var_to_bc:
                    var_to_bc[ var ].add( bc )
                else:
                    var_to_bc[ var ] = { bc }

            var_bc_counts = { var: len( var_to_bc[ var ] ) for var in var_to_bc }

            var_bc_counts_sort = { var: count for var, count in sorted( var_bc_counts.items(), key = lambda item: -item[ 1 ] ) }

            #get the variant with the highest number of barcodes
            top_var = list( itertools.islice( var_bc_counts_sort.items(), 1 ) )[ 0 ]

            #get max bcs from one variant for each isogrp
            isogrp_var_count[ isogrp ].append( top_var[ 1 ] )

            print( 'For isoform:', out_tbl.loc[ isogrp ].isoform )
            print( 'The variants with the top', print_count, 'number of barcodes are:')
            print( list( itertools.islice( var_bc_counts_sort.items(), print_count ) ) )

            #this gets rid of the QC counts like unmapped
            if ')' not in str( out_tbl.loc[ isogrp ].isoform ):
                isogrp_var_count[ isogrp ].append( 0 )
                continue

            #get a list of all variants
            var_list = [ v for var in var_bc_counts_sort for v in var.split(',') ]
            var_list_unique = list( set( var_list ) )

            #this gets the first position where the isoform differs from expected junctions
            jn_diff = list( { jn for jns in out_tbl.loc[ isogrp ].isoform for jn in jns }
                        - set( unique_jns ) )

            #lets keep the best isoforms
            #first check if the isoform exactly matches an isoform we're expecting to see
            if any( [ out_tbl.loc[ isogrp ].isoform == can_iso for can_iso in canonical_isos ] ):
                isogrp_var_count[ isogrp ].append( 1 )
            #then check if the max bc per isoform is greater than 100
            elif top_var[ 1 ] >= min_maxbc_count:
                isogrp_var_count[ isogrp ].append( 2 )
            #requires the top variant to have at least x barcodes and the top barcode to have at least y reads
            elif top_var[ 1 ] <= min_bc_max_reads[ 0 ] and isogrp_var_count[ isogrp ][ 2 ] <= min_bc_max_reads[ 1 ]:
                chucked_bcs += isogrp_var_count[ isogrp ][ 0 ]
                chucked_reads += out_tbl.loc[ isogrp ][ samp + '_read_count' ]
                isogrp_var_count[ isogrp ].append( 0 )
                continue
            #check if the var with the max bc is within tolerance of the where the isoform differs
            elif any( [ check_tolerance( int( var.split( ':' )[ 1 ] ),
                                         jn_diff,
                                         tol )
                        for var in top_var[0].split( ',' ) ] ):
                isogrp_var_count[ isogrp ].append( 3 )
            #check if at least min_cluster_size of variants are within a tol of each other
            elif cluster_vars(  var_bc_counts_sort,
                                min_cluster_size,
                                tol,
                                clustered_bc_ave ):
                isogrp_var_count[ isogrp ].append( 4 )
            #check if a double variant is also listed as a single variant
            elif len( var_list ) != len( var_list_unique ):
                isogrp_var_count[ isogrp ].append( 5 )
            else:
                isogrp_var_count[ isogrp ].append( 0 )
                missed_bcs += isogrp_var_count[ isogrp ][0]
                missed_reads += out_tbl.loc[ isogrp ][ samp+'_read_count' ]

        print( '%i total barcodes in bam file' % ( bc_cnt ) )
        print( '%i total reads in the bam file' % ( read_cnt ) )
        print( '%i (%.2f%%) barcodes had no variant in the subassembly' % ( bc_not_in_rna, 100*( bc_not_in_rna / bc_cnt ) ) )
        print( '%i (%.2f%%) reads were associated with barcodes without variants in the subassembly' \
            % ( reads_non_rna_bc, 100*( reads_non_rna_bc / read_cnt ) ) )
        print( 'From the remaining barcodes/reads in the subassembly...' )

        bc_sa_cnt = sum( i[ 0 ] for i in isogrp_var_count.values() )
        read_sa_cnt = sum( i[ 3 ] for i in isogrp_var_count.values() )

        print('%i (%.2f%%) barcodes failed the min_bc_max_reads filter' % ( chucked_bcs, 100*( chucked_bcs / bc_sa_cnt ) ) )
        print('%i (%.2f%%) reads failed the min_bc_max_reads filter' % ( chucked_reads, 100*( chucked_reads / read_sa_cnt ) ) )
        print('%i (%.2f%%) barcodes did not fulfill any filter' % ( missed_bcs, 100*( missed_bcs / bc_sa_cnt ) ) )
        print('%i (%.2f%%) reads did not fulfill any filter' % ( missed_reads, 100*( missed_reads / read_sa_cnt ) ) )
        print('%i (%.2f%%) reads were unmapped' % ( unmapped, 100*( unmapped / read_sa_cnt ) ) )
        print('%i (%.2f%%) reads were unpaired' % ( unpaired, 100*( unpaired / read_sa_cnt ) ) )
        print('%i (%.2f%%) reads were secondary alignments' % ( secondary_reads, 100*( secondary_reads / read_sa_cnt ) ) )
        print('%i (%.2f%%) reads were soft clipped' % ( soft_clipped, 100*( soft_clipped / read_sa_cnt ) ) )
        print( '%i (%.2f%%) reads were bad starts' % ( bad_starts, 100*( bad_starts / read_sa_cnt ) ) )
        print( '%i (%.2f%%) reads were bad ends' % ( bad_ends, 100*( bad_ends / read_sa_cnt ) ) )

        cols = [ samp + suffix for suffix in ['_num_bcs','_num_vars','_max_reads_per_bc','_sum_sa_reads','_max_bc_per_var','_filter'] ]
        iso_cnts = pd.DataFrame.from_dict( isogrp_var_count,
                                           orient = 'index',
                                           columns = cols )

        out_tbl = pd.merge( out_tbl, iso_cnts, left_index = True, right_index = True, how = "outer")

        out_tbl.index.name = 'isonum'

    #if iso_counts doesn't contain all the isoforms, the values will be null for those rows
    #change to zeros
    out_tbl = out_tbl.fillna( 0 )

    #get totals across samples
    for suffix in [ '_read_count', '_num_bcs', '_num_vars','_sum_sa_reads' ]:

        #sum of each row across the columns
        col = [ col for col in out_tbl.columns if suffix in col ]
        out_tbl[ 'total' + suffix ] = out_tbl[ col ].sum( axis = 1 )

    #counts number of samples passing filter for each isoform
    #specifically, counting non zero filter values
    filter_col = [ col for col in out_tbl.columns if 'filter' in col ]
    out_tbl[ 'total_passfilt' ] = out_tbl[ filter_col ].astype( bool ).sum( axis = 1 )

    if exising_iso_stats_df:
        out_tbl = pd.merge( cur_df, out_tbl, left_index = True, right_index = True, how="outer" )
        out_tbl = out_tbl.fillna( 0 )

    return out_tbl

def summarize_isos_wt_bc(align_by_samp_dict,
                             cnst_exons,
                                wt_bcs,
                                iso_df,
                                canonical_isos,
                                exising_iso_stats_df=None,
                                min_maxbc_count=100,
                                min_bc_max_reads=(2,2)):
    """Gets counts for number of variants and barcodes per isoform. Prints variants with the most number of barcodes.

    Args:
        pysam_align (PySam AlignmentFile): a tag sorted pysam alignment file with all of the reads for the sample
        cnst_exons (list of tuples): coords of known constant exons
        satbl (pandas dataframe): subassembly dataframe which maps barcodes to variants
                                    Indexed on readgroupid
        iso_df (pandas dataframe): dataframe of isoforms, isoform groups, and read counts created from get_all_isoforms
                                    and create_isogrps
                                    Indexed on isogrp
        print_count (int): number of desired variants to print out per isoform - program will print the variants
                            with the highest number of barcodes

    Returns:
        iso_df (pandas dataframe): Original iso_df with the addition of number of variants per isoform and number
                                    of barcodes per isoform
    """

    if exising_iso_stats_df:
        cur_df = exising_iso_stats_df.copy()

    out_tbl = iso_df.copy()

    for samp in align_by_samp_dict:

        print(samp)

        isogrp_to_bc = {}

        bc_cnt, read_cnt = 0,0

        for bc, _reads in itertools.groupby( align_by_samp_dict[ samp ], lambda _r: _r.get_tag( 'RX' ) ):

            if bc not in wt_bcs:
                continue

            for iso, reads in itertools.groupby( _reads, lambda _r: clean_jns( _r.get_blocks(), cnst_exons ) ):

                #ignore read if it doesn't splice to downtstream constant or has too few matches
                if iso == None or iso == 0:
                    continue

                r=list(reads)

                isogrp = out_tbl.loc[ out_tbl[ 'isoform' ] == iso ].index

                assert len(isogrp)==1, 'The isoform matches '+str( len( isogrp ) )+' isogroups instead of the expected 1'

                isogrp=isogrp[0]

                if isogrp not in isogrp_to_bc:
                    isogrp_to_bc[ isogrp ]={}

                if bc not in isogrp_to_bc[ isogrp ]:
                    isogrp_to_bc[ isogrp ][ bc ] = len(r)
                else:
                    isogrp_to_bc[ isogrp ][ bc ] += len(r)

                read_cnt+=len(r)

            bc_cnt+=1

            if bc_cnt%10000==0:
                print('Barcodes processed:',bc_cnt,'Reads processed:',read_cnt)

    #takes 16 minutes to get here
        isogrp_count = { isogrp: [ len( isogrp_to_bc[ isogrp ] ),
                                    max( isogrp_to_bc[ isogrp ].values() )
                                  ]
                        for isogrp in isogrp_to_bc
                        }

        chucked_bcs, chucked_reads = 0,0

        for isogrp in isogrp_to_bc:

            bcs = [ bc for bc in isogrp_to_bc[ isogrp ] ]

            bc_counts = len( bcs )

            print( 'For isoform:', out_tbl.loc[ isogrp ].isoform )
            print( 'There are', bc_counts, 'WT barcodes represented')

            #lets keep the best isoforms
            #first check if the isoform exactly matches an isoform we're expecting to see
            if any( [ out_tbl.loc[ isogrp ].isoform == can_iso for can_iso in canonical_isos ] ):
                isogrp_count[ isogrp ].append( 1 )
            #then check if the max bc per isoform is greater than 100
            elif bc_counts >= min_maxbc_count:
                isogrp_count[ isogrp ].append( 2 )
            #requires the top variant to have at least x barcodes and the top barcode to have at least y reads
            elif bc_counts > min_bc_max_reads[ 0 ] and isogrp_count[ isogrp ][ 1 ] > min_bc_max_reads[ 1 ]:
                isogrp_count[ isogrp ].append( 6 )
            else:
                isogrp_count[ isogrp ].append( 0 )
                chucked_bcs += isogrp_count[ isogrp ][0]
                chucked_reads += out_tbl.loc[ isogrp ][ samp+'_read_count' ]

        print('%i (%.2f%%) barcodes did not fulfill any filter' \
            % ( chucked_bcs, 100*(chucked_bcs/sum(i[0] for i in isogrp_count.values() ) ) ) )
        print('%i (%.2f%%) reads did not fulfill any filter' \
            % ( chucked_reads, 100*(chucked_reads/out_tbl[ samp+'_read_count' ].sum() ) ) )

        cols = [ samp + suffix for suffix in ['_num_bcs','_max_reads_per_bc','_filter'] ]
        iso_cnts = pd.DataFrame.from_dict(isogrp_count, orient='index',
                                           columns=cols)

        out_tbl = pd.merge( out_tbl, iso_cnts, left_index=True, right_index=True, how="outer")

        out_tbl.index.name = 'isonum'

    #if iso_counts doesn't contain all the isoforms, the values will be null for those rows
    #change to zeros
    out_tbl = out_tbl.fillna(0)

    #get totals across samples
    for suffix in [ '_read_count', '_num_bcs', ]:

        #sum of each row across the columns
        col = [ col for col in out_tbl.columns if suffix in col ]
        out_tbl[ 'total'+suffix ] = out_tbl[ col ].sum( axis=1 )

    #counts number of samples passing filter for each isoform
    #specifically, counting non zero filter values
    filter_col = [ col for col in out_tbl.columns if 'filter' in col ]
    out_tbl[ 'total_passfilt' ] = out_tbl[ filter_col ].astype( bool ).sum( axis=1 )

    if exising_iso_stats_df:
        out_tbl = pd.merge( cur_df, out_tbl, left_index=True, right_index=True, how="outer" )
        out_tbl = out_tbl.fillna(0)

    return(out_tbl)

def combine_isoforms(iso_df,
                    cryptic_exons,
                    data_from_csv = False):

    out_tbl = iso_df.copy()

    filter_cols = [ col for col in out_tbl.columns if 'filter' in col ]
    out_tbl.drop( columns = filter_cols, inplace=True )

    isoform_comb = []

    for iso in out_tbl.isoform:

        #this will make sure any strings are converted to tuples
        if data_from_csv:
            iso = literal_eval(iso)

        cur_iso = []
        prev_jn = None

        for jn in iso:

            #don't add junctions overlapping the cryptic_exons
            if not ( any( [ jn[0] in range( ce[0],ce[1] ) or jn[1] in range( ce[0],ce[1] )
                            for ce in cryptic_exons ] ) ):
                cur_iso.append( jn )
                cur_jn = jn
                if prev_jn:
                    #combine adjacent junctions
                    if cur_jn[0] == prev_jn[1]:
                        cur_iso[-2:]=[ (prev_jn[0], cur_jn[1]) ]
                prev_jn=cur_jn

        cur_iso = tuple( cur_iso )
        isoform_comb.append( cur_iso )

    out_tbl[ 'comb_isoform' ] = isoform_comb

    #sums across matching isoforms
    max_cols = [ col for col in out_tbl.columns if 'max' in col ]
    sums_df = out_tbl.groupby(['comb_isoform'],as_index=False).sum().drop(columns=max_cols)
    #max across matching isoforms
    other_cols = [ col for col in out_tbl.columns for name in ['read_count','num_bcs','num_vars','isoform'] if name in col and col!= 'comb_isoform' ]
    max_df = out_tbl.groupby(['comb_isoform'],as_index=False).max().drop(columns=other_cols)
    #counts number of isoforms per isogrp
    counts_df = out_tbl.groupby(['comb_isoform'],as_index=False).count()[['comb_isoform','isoform']].rename(columns={"isoform": "isoform_counts"})
    #concatenates isoform numbers to track from previous tables
    #iso_nums_df = out_tbl.reset_index().groupby(['comb_isoform'])['index'].apply(','.join).reset_index()
    iso_nums_df = out_tbl.reset_index().groupby(['comb_isoform'])['isonum'].apply(','.join).reset_index()
    #concatenates all isoforms to track from previous table
    isoform_df = out_tbl.reset_index().groupby(['comb_isoform'])['isoform'].apply(tuple).reset_index()

    out_tbl = pd.merge(sums_df,max_df,on='comb_isoform')
    out_tbl = pd.merge(out_tbl,counts_df,on='comb_isoform')
    out_tbl = pd.merge(out_tbl,isoform_df,on='comb_isoform')
    out_tbl = pd.merge(out_tbl,iso_nums_df,on='comb_isoform').rename(columns={"index": "comb_iso_nums"})
    out_tbl = out_tbl.sort_values(by='total_read_count', ascending=False)

    out_tbl['isogrp'] = ['isogrp'+str(i).zfill( len( str( out_tbl.shape[0] ) ) ) for i in range( out_tbl.shape[0] ) ]
    out_tbl.set_index('isogrp',inplace=True)

    return (out_tbl)

def create_iso_dict_no_cnst(iso_df):

    iso_df_c = iso_df.copy()

    isogrpdict = {}

    #for iso in iso_df_c.index:

        #try:
            #isogrpdict[ iso ] = literal_eval( iso_df_c.loc[ iso ].isoform )
        #except:
            #isogrpdict[ iso ] = iso_df_c.loc[ iso ].isoform

    #pandas hate tuples so this ensures any strings come out as tuples
    isogrpdict = { iso: literal_eval( iso_df_c.loc[ iso ].isoform )
                    for iso in iso_df_c.index }

    return( isogrpdict )

def make_junction_graph(exon_bed):

    """Create a graph of all possible traversals of specified exons

    Args:
        exon_bed (PyBedTool): bed file with exon records.  The first record is assumed to be the constant first exon,
            and the last record is assumed to be the constant final exon

    Returns:
        dict (str -> tuple): dictionary of named path (ie isoform) traversing exons.  Keys are path names,
            values are tuple (series) of tuples corresponding to exons on that path.  The exon coordinates returned are
            1-based, inclusive.

    """

    gr = nx.DiGraph()

    nex = len(exon_bed)

    gr.add_node( exon_bed[0].start+1 )
    gr.add_node( exon_bed[0].end )

    # add donors and acceptors
    for iex in range(1, len(exon_bed)):
        gr.add_node( exon_bed[iex].start+1 ) # +1 to make this 1-based,inclusive
        gr.add_node( exon_bed[iex].end )     # +0 to make this 1-based,inclusive

    # loop through donors in order
    for iex in range(len(exon_bed)):
        gr.add_edge( exon_bed[iex].start+1, exon_bed[iex].end )
        # at each donor, add a path to any acceptor which is >= donor position
        for jex in range(len(exon_bed)):
            if exon_bed[jex].start+1 > exon_bed[iex].end:
                gr.add_edge( exon_bed[iex].end, exon_bed[jex].start+1 )

    gr.add_node( exon_bed[nex-1].start+1 )
    gr.add_node( exon_bed[nex-1].end )

    lpaths = list( [path for path in
                      nx.all_simple_paths( gr, exon_bed[0].start+1, exon_bed[nex-1].end )] )

    # convert paths to series of (exonstart,exonend) for legibility

    lpaths = [ tuple( [ (path[i*2],path[i*2+1]) for i in range(int(len(path)/2)) ] )
               for path in lpaths ]

    lpathnames = ['iso{:02d}'.format( i ) for i in range(len(lpaths)) ]

    return dict(zip(lpathnames, lpaths))


def trunc_isoforms_by_readlayout_SE(
    pathdict,
    read_start,
    read_length ):

    """Create "compatibility" group of isoforms cropped by a given single end read of fixed length and
    fixed starting position

    Args:
        pathdict (dict<str> -> tuple): isoform name to path dictionary
        read_start (int): fixed starting position of a read; 1-based, inclusive
        read_length (int): fixed read length

    Returns:
        - dict (str -> tuple): dictionary of isoform truncated by read position and length. Keys = compatbility group name, values = exon (or partial exon) starts and ends, in 1-based inclusive coordiantes

        - dict (str -> list of str) - within each named compatbility group (keys) --> which named isoforms from pathdict are equivalent (values)

    """

    # compat_grp --> ( visible path, [ isonames from paths input ] )

    m_iso_vispath = {}

    for pathname in pathdict:
        path = pathdict[pathname]
        found_ex = False
        for iex in range(int(len(path))):
            coord_ex = ( path[iex][0], path[iex][1] )
            if read_start >= coord_ex[0] and read_start <= coord_ex[1]:
                found_ex=True
                break

        if not found_ex:
            # this read start position does not exist on this path.
            m_iso_vispath[ pathname ] = None
            continue

        # truncate path to what is 'visible' given the proposed layout
        path_vis = []
        read_bp_remain = read_length

        in_first_ex = True
        while read_bp_remain > 0 and iex < int(len(path)):
            coord_ex = ( path[iex][0], path[iex][1] )

            if in_first_ex :
                coord_start = read_start
                in_first_ex = False
            else:
                coord_start = coord_ex[0]

            coord_end = min( coord_ex[1], coord_start+read_bp_remain-1 )

            path_vis.append( (coord_start,coord_end) )

            read_bp_remain -= (coord_end - coord_start + 1)

            iex += 1

        path_vis = tuple(path_vis) # change to hashable type

        m_iso_vispath[ pathname ] = path_vis

    m_vispath_name = {}
    m_name_vispath = {}

    n_vispath = 0
    for vispath in set(m_iso_vispath.values()):
        vispathname = 'isogrp{:02d}'.format(n_vispath)
        m_name_vispath[vispathname] = vispath
        m_vispath_name[vispath] = vispathname
        n_vispath+=1

    m_vispath_liso = {}
    for pathname in pathdict:
        vispath = m_iso_vispath[ pathname ]
        vispathname = m_vispath_name[vispath]

        if vispathname not in m_vispath_liso:
            m_vispath_liso[ vispathname ] = [ pathname ]
        else:
            m_vispath_liso[ vispathname] .append( pathname )

    return m_name_vispath, m_vispath_liso


def compute_isoform_counts(
    bam,
    isogrpdict,
    cnst_exons,
    tol_first_last=0,
    min_reads=1,
    other_isos_usable=False
):
    """
    Create per-barcode counts of reads matching each isoform. Resulting dataset contains a count of total reads, unmapped
    reads, reads with a bad start (greater than 5 (default) basepairs away from most common start), how many reads
    match each isoform, and the psis for each isoform (matching reads/usable reads).

    Args:
        bam (pysam.AlignmentFile):  a barcode tag-grouped bam file of spliced alignments to reporter sequence
        isogrpdict (dict): isoform compatbility group name --> list of exon start, end coodinates (1-based, inclusive)
        read_start_coord (int): expected fixed read start position
        tol_first_last (int): allow up to this many bases tolerance at the first and last exons
        min_reads (int): require at least this many reads to be present per barcode group
        other_isos_usable (bool): should reads not matching any of the known isoforms be counted in the total?

    Returns:
        Pandas data frame with per-barcode read counts
    """

    rowList = []

    ctr_bcs,ctr_reads=0,0

    for tag, _reads in itertools.groupby( bam, lambda _r:_r.get_tag( 'RX' )):

        reads=list(_reads)

        if len(reads)<min_reads:
            continue

        n_umapped=0
        n_badstart=0
        n_badend=0
        n_softclip=0
        n_nomatch = 0
        ln_matches=[0] * len( isogrpdict )

        for read in reads:
            if read.is_unmapped:
                n_umapped+=1
            #if the end of the upstream constant exon isn't within the tolerance count it as a bad start
            elif not( any( jn[1] <= cnst_exons[ 0 ][ 1 ] + tol_first_last
                            or jn[1] > cnst_exons[ 0 ][ 1 ] - tol_first_last
                            for jn in read.get_blocks() ) ):
                n_badstart+=1
            else:
                cur_matches = check_junctions2( read, isogrpdict, cnst_exons, tol_first_last )

                if cur_matches == None:
                    n_badend+=1
                elif cur_matches == .1:
                    n_softclip+=1
                elif sum(cur_matches)==0:
                    n_nomatch+=1
                else:
                    assert sum(cur_matches) <= 1, ( str(read),str(cur_matches) )

                    # for now, not dealing with read groups matching multiple isoform groups - this should not happen

                    for i in range(len(cur_matches)):
                        ln_matches[i]+=cur_matches[i]

        total = n_umapped + n_badstart + n_badend + n_softclip + n_nomatch + sum(ln_matches)

        ctr_bcs+=1
        ctr_reads+=len(reads)

        # for debugging only go through a few
        # if ctr_bcs==10:break

        if ctr_bcs % 1000 == 0 :
            print('processed {} bcs, {} reads'.format( ctr_bcs, ctr_reads ))

        rowList.append( [tag,total,n_umapped,n_badstart,n_badend,n_softclip,n_nomatch]+ln_matches )

    # from IPython.core.debugger import set_trace
    # set_trace()

    psidf=pd.DataFrame(rowList, columns=['barcode','num_reads','unmapped_reads','bad_starts','bad_ends','soft_clipped','other_isoform']+list(isogrpdict.keys()))

    if not other_isos_usable:
        psidf['usable_reads']=psidf.num_reads-(psidf.unmapped_reads+psidf.bad_starts+psidf.bad_ends+psidf.soft_clipped+psidf.other_isoform)

        #comment this out to save on memory
        #for iso in isogrpdict:
            #psidf[iso+'_psi'] = psidf[iso] / psidf.usable_reads
    else:
        psidf['usable_reads']=psidf.num_reads-(psidf.unmapped_reads+psidf.bad_starts+psidf.bad_ends+psidf.soft_clipped)

        #comment this out to save on memory
        #for iso in isogrpdict:
            #psidf[iso+'_psi'] = psidf[iso] / psidf.usable_reads

        #psidf['other_isoform_psi'] = psidf['other_isoform'] / psidf.usable_reads

    #sets psi to 0 instead of nan if none of the reads map to that isoform'((1201, 1266),)'
    psidf = psidf.fillna(0)
    psidf = psidf.set_index('barcode')

    return psidf

def compute_isoform_counts_pe( bam,
                                isogrpdict,
                                cnst_exons,
                                spl_tol = 3,
                                indel_tol = 20,
                                min_matches_for = 70,
                                min_matches_rev = 50,
                                tol_first_last=0,
                                min_reads=1,
                                other_isos_usable=False,
                                bc_tag = 'RX' ):
    """
    Create per-barcode counts of reads matching each isoform. Resulting dataset contains a count of total reads, unmapped
    reads, reads with a bad start (greater than 5 (default) basepairs away from most common start), how many reads
    match each isoform, and the psis for each isoform (matching reads/usable reads).

    Args:
        bam (pysam.AlignmentFile):  a barcode tag-grouped bam file of spliced alignments to reporter sequence
        isogrpdict (dict): isoform compatbility group name --> list of exon start, end coodinates (1-based, inclusive)
        read_start_coord (int): expected fixed read start position
        tol_first_last (int): allow up to this many bases tolerance at the first and last exons
        min_reads (int): require at least this many reads to be present per barcode group
        other_isos_usable (bool): should reads not matching any of the known isoforms be counted in the total?

    Returns:
        Pandas data frame with per-barcode read counts
    """

    rowList = []

    ctr_bcs,ctr_reads = 0,0

    isogrps = list( isogrpdict.keys() )
    isoforms = list( isogrpdict.values() )

    for bc, _reads in itertools.groupby( bam, lambda _r:_r.get_tag( bc_tag ) ):

        bc_n_reads = 0

        reads = list( _reads )

        if len( reads ) < min_reads:
            continue

        n_unpaired = 0
        n_secondary = 0
        n_unmapped = 0
        n_badstart = 0
        n_badend = 0
        n_softclip = 0
        n_nomatch = 0
        ln_matches = [ 0 ] * len( isogrpdict )

        for read in reads:

            if not read.is_paired:

                n_unpaired += 1
                bc_n_reads += 1
                read1 = None
                read2 = None
                continue

            elif read.is_read2:
                read2 = read

            else:
                read1 = read
                read2 = None

            if read1 and read2 and read1.query_name == read2.query_name:

                if read1.is_secondary:

                    n_secondary += 1
                    continue

                bc_n_reads += 1

                iso = clean_jns_pe( read1,
                                    read2,
                                    cnst_exons,
                                    spl_tol,
                                    indel_tol,
                                    min_matches_for,
                                    min_matches_rev )

                if read1.is_unmapped or read2.is_unmapped:

                    n_unmapped += 1
                    continue

                elif iso == 'bad_starts':
                    n_badstart += 1
                    continue

                elif iso == 'bad_ends':
                    n_badend += 1
                    continue

                elif iso == 'soft_clipped':
                    n_softclip += 1
                    continue

                else:

                    if iso in isoforms:

                        ln_matches[ isoforms.index( iso ) ] += 1

                    else:

                        n_nomatch += 1
                        continue

        ctr_bcs += 1
        ctr_reads += bc_n_reads
        # for debugging only go through a few
        # if ctr_bcs==10:break

        if ctr_bcs % 1000 == 0 :
            print( 'processed {} bcs, {} reads'.format( ctr_bcs, ctr_reads ) )

        rowList.append( [ bc, bc_n_reads, n_unpaired, n_secondary, n_unmapped, n_badstart, n_badend, n_softclip, n_nomatch ] + ln_matches )

    bam.close()

    # from IPython.core.debugger import set_trace
    # set_trace()

    psidf = pd.DataFrame( rowList,
                         columns = [ 'barcode', 'num_reads', 'unpaired_reads', 'secondary_reads', 'unmapped_reads', 'bad_starts', 'bad_ends', 'soft_clipped', 'other_isoform' ] + isogrps )

    if not other_isos_usable:

        psidf[ 'usable_reads' ] = psidf.num_reads - ( psidf[ [ 'unpaired_reads', 'secondary_reads', 'unmapped_reads', 'bad_starts', 'bad_ends', 'soft_clipped', 'other_isoform' ] ].sum( axis = 1 ) )

        #comment this out to save on memory
        #for iso in isogrpdict:
            #psidf[iso+'_psi'] = psidf[iso] / psidf.usable_reads
    else:

        psidf[ 'usable_reads' ] = psidf.num_reads - ( psidf[ [ 'unpaired_reads', 'secondary_reads', 'unmapped_reads', 'bad_starts', 'bad_ends', 'soft_clipped' ] ].sum( axis = 1 ) )

        #comment this out to save on memory
        #for iso in isogrpdict:
            #psidf[iso+'_psi'] = psidf[iso] / psidf.usable_reads

        #psidf['other_isoform_psi'] = psidf['other_isoform'] / psidf.usable_reads

    #sets psi to 0 instead of nan if none of the reads map to that isoform'((1201, 1266),)'
    psidf = psidf.fillna( 0 )
    psidf = psidf.set_index( 'barcode' )

    return psidf

# check coverage of exon and of junctions
def check_junctions2( read, isogrpdict, cnst_exons, tol_first_last=0 ):

    """Check an individual aligned read vs a list of isoforms

    Args:
        read (pysam.AlignedSegment): a single aligned read
        isogrpdict (dict): isoform compatbility group name --> list of exon start, end coodinates (1-based, inclusive)
        tol_first_last (int): allow up to this many bases tolerance at the first and last exons

    Returns:
        list of bools indicating whether this alignment is consistent with each of the provided isoform compat. groups.
    """
    # get blocks of reference coverage from the read.
    l_refcoord_blks_cleaned = clean_jns( read.get_blocks(), cnst_exons )

    if l_refcoord_blks_cleaned == None:
        lmatches = None
    elif l_refcoord_blks_cleaned == 0:
        lmatches = .1
    else:
        lmatches=[]

        # now compare to isogrps
        for isogrpname in isogrpdict:
            isogrp = isogrpdict[isogrpname]
            #print(isogrpname,isogrp)

            match = False
            lblks = len( l_refcoord_blks_cleaned )
            if lblks == len( isogrp ) and isogrp == l_refcoord_blks_cleaned:
                match=True

            lmatches.append( match )

    return lmatches

def filter_on_barc_len( jxnbybctbl, max_len=35 ):
    li = jxnbybctbl.index.str.len() <= max_len
    subtbl = jxnbybctbl.loc[li].copy()
    return subtbl

def create_named_isogrps(iso_grp_dict,
                        iso_names_dict,
                        remove_exon_coords,
                        upstream_ex_len,
                        read_len,
                        tol = 0):

    named_iso_grps = { isoname: [] for isoname in iso_names_dict }
    #set up an other category for isoforms that don't match
    named_iso_grps['OTHER']=[]

    for isonum,iso in iso_grp_dict.items():

        iso_cleaned = []
        for jn in iso:
            #check if the junction overlaps the exons to remove
            if not( any ( jn[0] in range( r_exon[0], r_exon[1] ) or jn[1] in range( r_exon[0], r_exon[1] )
                    for r_exon in remove_exon_coords ) ):
                iso_cleaned.append(jn)

        #more likely to be off by a base if we are near the read length
        bases_used = sum( jn[1] - jn[0] +1 for jn in iso_cleaned ) + upstream_ex_len

        #this will fail if there's two possible isoform matches
        #would that ever happen?
        match=False
        for isoname, iso_to_match in iso_names_dict.items():
            #check if it perfectly matches the named isoform
            if iso_cleaned == iso_to_match:
                match = isoname
                break

            #if they're different lengths don't bother going through all the tolerance checks
            elif len( iso_to_match ) != len( iso_cleaned ):
                continue

            #if we have are using a tolerance and we are near the read length
            elif tol > 0 and bases_used in range( read_len - tol, read_len +tol ):
                #check if the final junction is within the tolerance (allows for deletions in the upstream exon)

                #gets index of all mismatched junctions
                mismatched_jn = [ iso_cleaned.index( jn_c )
                                for jn_m, jn_c in zip( iso_to_match, iso_cleaned )
                                if jn_m != jn_c ]

                #we only want to allow mismatches at the end of the last junction
                #so first check the only mismatch is the last junction
                #then check the start of the junction matches
                #then check if the final junction is within tolerance
                if mismatched_jn[0] == len( iso_cleaned ) - 1 \
                and iso_cleaned[-1][0] == iso_to_match[-1][0] \
                and iso_cleaned[-1][1] in range(iso_to_match[-1][1] - tol, iso_to_match[-1][1] + tol):
                    match = isoname
                    break

        if match:
            named_iso_grps[ match ].append( isonum )
        else:
            named_iso_grps[ 'OTHER' ].append( isonum )

    return(named_iso_grps)

def combine_isogrps( new_grpnames_to_old_grpnames,
                     jxnbybctbl,
                     keep_cols = ['num_reads','unmapped_reads','bad_starts','bad_ends','soft_clipped','other_isoform','usable_reads'],
                     ):
    """ combine barcode groups
    Arguments:
        new_grpnames_to_old_grpnames {[type]} -- [description]
    """

    chg_cols = [ isocol for isocol_l in new_grpnames_to_old_grpnames.values()
                        for isocol in isocol_l ]

    other_cols = [ col for col in jxnbybctbl.columns
                   if col not in chg_cols ]

    if keep_cols != other_cols:

        missing_cols = list( set( other_cols ).difference( set( keep_cols ) ) )

        print( '%i columns will be removed during this process.\nColumns: ' % len( missing_cols ), missing_cols )

    newtbl = pd.DataFrame()
    for c in keep_cols:
        newtbl[c] = jxnbybctbl[c]

    for newgrp in new_grpnames_to_old_grpnames:
        oldgrp = new_grpnames_to_old_grpnames[ newgrp ]

        if type(oldgrp)==str:
            oldgrp=[oldgrp]
        else:
            assert type(oldgrp)==list

        if len(oldgrp)==1:
            newtbl[newgrp] = jxnbybctbl[oldgrp]
        else:
            newtbl[newgrp] = jxnbybctbl[oldgrp].sum(axis=1)

    for newgrp in new_grpnames_to_old_grpnames:
        newtbl[newgrp+'_psi'] = newtbl[newgrp] / newtbl['usable_reads']

    #fills in missing PSI values with 0 so we can create sparse datasets
    newtbl = newtbl.fillna(0)

    return newtbl

def combine_isogrps_pe( new_grpnames_to_old_grpnames,
                        jxnbybctbl,
                        keep_cols = ['num_reads','secondary_reads', 'unpaired_reads', 'unmapped_reads','bad_starts','bad_ends','soft_clipped','other_isoform','usable_reads'] ):
    """ combine barcode groups
    Arguments:
        new_grpnames_to_old_grpnames {[type]} -- [description]
    """

    chg_cols = [ isocol for isocol_l in new_grpnames_to_old_grpnames.values()
                        for isocol in isocol_l ]

    other_cols = [ col for col in jxnbybctbl.columns
                   if col not in chg_cols ]

    if keep_cols != other_cols:

        missing_cols = list( set( other_cols ).difference( set( keep_cols ) ) )

        print( '%i columns will be removed during this process.\nColumns: ' % len( missing_cols ), missing_cols )

    newtbl = pd.DataFrame()
    for c in keep_cols:
        newtbl[c] = jxnbybctbl[c]

    for newgrp in new_grpnames_to_old_grpnames:
        oldgrp = new_grpnames_to_old_grpnames[ newgrp ]

        if type(oldgrp)==str:
            oldgrp=[oldgrp]
        else:
            assert type(oldgrp)==list

        if len(oldgrp)==1:
            newtbl[newgrp] = jxnbybctbl[oldgrp]
        else:
            newtbl[newgrp] = jxnbybctbl[oldgrp].sum(axis=1)

    for newgrp in new_grpnames_to_old_grpnames:
        newtbl[newgrp+'_psi'] = newtbl[newgrp] / newtbl['usable_reads']

    #fills in missing PSI values with 0 so we can create sparse datasets
    newtbl = newtbl.fillna(0)

    return newtbl

def merge_unmapped_bcs( tbl_by_bc,
                        unmapped_m1,
                        bc_split_char = '_BC=',
                        unmap_col = 'unmapped_reads',
                        read_tot_col = 'num_reads'
                        ):

    tbb = tbl_by_bc.drop( columns = unmap_col ).copy()

    bc_cnt_m1 = pd.Series( Counter( [ entry.name.split( bc_split_char )[ -1 ] for entry in unmapped_m1 ] ),
                           name = unmap_col )

    unmapped_m1.close()

    bc_intsect = bc_cnt_m1.index.intersection( tbb.index )

    bc_cnt_m1 = bc_cnt_m1.loc[ bc_intsect ].copy()

    tbb[ unmap_col ] = bc_cnt_m1

    tbb[ unmap_col ] = tbb[ unmap_col ].fillna( value = 0 ).astype( int )

    tbb[ read_tot_col ] += tbb[ unmap_col ]

    return tbb

def process_bcs_wrapper( sample,
                         pysam_align,
                         all_isogrpdict,
                         named_isogrpdict,
                         cnst_exons,
                         satbl,
                         spl_tol = 3,
                         indel_tol = 20,
                         min_matches_for = 70,
                         min_matches_rev = 50,
                         tol_first_last = 0,
                         min_reads = 1,
                         other_isos_usable = False,
                         bc_tag = 'RX',
                         max_bc_len = 30,
                         unmapped_pysam = None,
                         unmap_bc_split_char = '_BC=',
                         unmap_col = 'unmapped_reads',
                         read_tot_col = 'num_reads',
                         usable_read_col = 'usable_reads',
                         other_cols = [ 'secondary_reads', 'unpaired_reads', 'bad_starts', 'bad_ends', 'soft_clipped', 'other_isoform' ],
                         waterfall_thresh = [ 75, 95 ],
                         bc_read_cut_thresh = 95 ):

    assert bc_read_cut_thresh in waterfall_thresh, 'BC read cut off thresholds must be a subset of waterfall thresholds!'

    t0 = time.time()

    print( 'Processing sample %s' % sample )

    msamp_bcrnatbl = compute_isoform_counts_pe( pysam_align,
                                                all_isogrpdict,
                                                cnst_exons,
                                                spl_tol = spl_tol,
                                                indel_tol = indel_tol,
                                                min_matches_for = min_matches_for,
                                                min_matches_rev = min_matches_rev,
                                                bc_tag = bc_tag )

    pysam_align.close()

    msamp_bcrnatbl_flen = filter_on_barc_len( msamp_bcrnatbl,
                                              max_len = max_bc_len )

    if unmapped_pysam:

        msamp_bcrnatbl_flen = merge_unmapped_bcs( msamp_bcrnatbl_flen,
                                                  unmapped_pysam,
                                                  bc_split_char = unmap_bc_split_char,
                                                  unmap_col = unmap_col,
                                                  read_tot_col = read_tot_col )

    msamp_bcrnatbl_rename = combine_isogrps_pe( named_isogrpdict,
                                                msamp_bcrnatbl_flen,
                                                keep_cols = [ read_tot_col, usable_read_col, unmap_col ] + other_cols  )

    msamp_varbcrnatbl_flen_allisos = mbcs.merge_subasm_and_rna_tbls( satbl,
                                                                     msamp_bcrnatbl_flen )

    x_cuts,y_cuts = sp.waterfall_plot( msamp_bcrnatbl_flen,
                                       usable_read_col,
                                       waterfall_thresh )

    cut_d = { 'x': x_cuts, 'y': y_cuts }

    msamp_byvartbl_allisos = mbcs.summarize_byvar_singlevaronly_pe( satbl,
                                                                    msamp_bcrnatbl_flen,
                                                                    cut_d[ 'y' ][ bc_read_cut_thresh ],
                                                                    [ usable_read_col, unmap_col ] + other_cols )

    t1 = time.time()

    print( 'Finished processing sample %s in %.2f minutes' % ( sample, ( t1 - t0 ) / 60 ) )

    return { 'msamp_bcrnatbl_rename': msamp_bcrnatbl_rename,
             'msamp_varbcrnatbl_flen_allisos': msamp_varbcrnatbl_flen_allisos,
             'msamp_byvartbl_allisos': msamp_byvartbl_allisos,
             'bc_read_cutoffs': cut_d }

def create_read_count_df( bysamp_bc_dict,
                          thresholds,
                          read_cutoff_key = 'bc_read_cutoffs' ):

    out_dict = { 'sample': [] }

    for thresh in thresholds:

        out_dict[ str( thresh ) + '_x' ] = []
        out_dict[ str( thresh ) + '_y' ] = []

    for samp in bysamp_bc_dict:

        out_dict[ 'sample' ].append( samp )

        for thresh in thresholds:

            out_dict[ str( thresh ) + '_x' ].append( 10**( bysamp_bc_dict[ samp ][ read_cutoff_key ][ 'x' ][ thresh ] ) )
            out_dict[ str( thresh ) + '_y' ].append( bysamp_bc_dict[ samp ][ read_cutoff_key ][ 'y' ][ thresh ] )

    outdf = pd.DataFrame( out_dict )

    for col in outdf.columns:

        if col.endswith( '_x' ):

            outdf[ col + '_log10' ] = np.log10( outdf[ col ].tolist() )

    return outdf
