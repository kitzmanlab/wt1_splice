import pandas as pd
import numpy as np
import pysam
import itertools
import subprocess as subp

def check_individual_variants(pysam_align,
                                bc_list,
                                suspicious_isos=[]):

    iso_dict = {}
    full_reads = {}

    for bc, _reads in itertools.groupby( pysam_align, lambda _r: _r.get_tag( 'RX' ) ):

        if bc not in bc_list:
            continue

        reads = list(_reads)

        for r in reads:
            iso = tuple( r.get_blocks() )
            if iso not in iso_dict:
                iso_dict[ iso ] = {bc: 1}
            elif bc not in iso_dict[ iso ]:
                iso_dict[ iso ][ bc ] = 1
            else:
                iso_dict[ iso ][ bc ] += 1
            if iso in suspicious_isos:
                if iso not in full_reads:
                    full_reads[ iso ] = [ r ]
                else:
                    full_reads[ iso ].append(r)

    return([iso_dict, full_reads])

def extract_nonzero_isos_byvar(alliso_byvar_df,
                            isoform_df,
                            col_prefix='wmean_',
                            remove_prefix=True):

    byvar = alliso_byvar_df.copy()

    col = [ col for col in byvar if col.startswith( col_prefix ) ]
    max_psi = byvar[col].max()

    #grab all the columns > 0
    nonzero_max = max_psi[ max_psi>0 ]

    if remove_prefix:
        #remove column suffix to just leave the isoform name
        #needed to look up isoform number in isoform dataframe
        #creates dictionary with column as keys and the isoform as values
        nonzero_iso_dict = { iso_col: iso_lookup( isoform_df, iso_col.replace( col_prefix, '' ) )
                                for iso_col in nonzero_max.index }
    else:
        nonzero_iso_dict = { iso_col: iso_lookup( isoform_df, iso_col )
                                for iso_col in nonzero_max.index }

    return( nonzero_iso_dict )

def extract_nonzero_isos_bybc(alliso_bybc_df,
                            isoform_df,
                            col_prefix='iso'):

    bybc = alliso_bybc_df.copy()

    col = [ col for col in bybc if col_prefix in col ]

    max_psi = bybc[col].max()

    #grab all the columns > 0 and not missing
    nonzero_max = max_psi[ max_psi>0 ]

    nonzero_iso_dict = { }

    for iso_col in nonzero_max.index:

        isoform = iso_lookup( isoform_df, iso_col) if '_psi' not in iso_col else iso_lookup( isoform_df, iso_col.replace('_psi','') )

        if isoform in nonzero_iso_dict:
            if 'psi' in iso_col:
                nonzero_iso_dict[ isoform ].append( nonzero_max.loc[ iso_col ] )
            else:
                nonzero_iso_dict[ isoform ][ 0 ] = nonzero_max.loc[ iso_col ]

        else:
            if 'psi' in iso_col:
                nonzero_iso_dict[ isoform ] = [ None, nonzero_max.loc[ iso_col ] ]
            else:
                nonzero_iso_dict[ isoform ] = [ nonzero_max.loc[ iso_col ] ]

    return( nonzero_iso_dict )


def iso_lookup(isoform_df,
                isonum):

    iso_df = isoform_df.copy()

    return( iso_df.loc[ isonum ].isoform )

def grab_seq_by_bc( pysam_align,
                    mapping = 'mapped',
                    barcode = None):

    assert mapping in ['mapped', 'unmapped', 'both'], "Read mapping must be specified as 'mapped', 'unmapped', or 'both'"

    for bc, _reads in itertools.groupby( pysam_align, lambda _r: _r.get_tag( 'RX' ) ):

        if barcode and bc != barcode:
            continue

        reads = list(_reads)

        seq_dict = {}

        for read in reads:
            seq = read.get_forward_sequence()

            if mapping == 'mapped' and not read.is_unmapped:
                if seq in seq_dict:
                    seq_dict[seq]+=1
                else:
                    seq_dict[seq]=1

            elif mapping == 'unmapped' and read.is_unmapped:
                if seq in seq_dict:
                    seq_dict[seq]+=1
                else:
                    seq_dict[seq]=1

            elif mapping == 'both':
                if seq in seq_dict:
                    seq_dict[seq]+=1
                else:
                    seq_dict[seq]=1

    return seq_dict

def edit_distance(s1, s2):

    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]

def dist_from_ref( refseq,
                    altseq,
                    dcan_isos):

    dcan_isos_dist = {}
    bases_used = 0
    for can_iso, iso_coords in dcan_isos.items():

        #subtract 1 from refseq coords to adjust for 1 and 0 based coords
        dcan_isos_dist[ can_iso ] = edit_distance( refseq[ iso_coords[0] - 1: iso_coords[1] ],
                                                    altseq[ bases_used: bases_used + ( iso_coords[1] - iso_coords[0] + 1 ) ] )

        bases_used += iso_coords[1] - iso_coords[0] + 1

    return dcan_isos_dist

def get_refseqs( fa_file ):

    with pysam.FastxFile( fa_file ) as fa:

        refseq = [ entry.sequence for entry in fa ]

    return refseq

def create_rownames_align_bctbl( bc,
                                 hisat_bam,
                                 print_err = False ):

    #gets all read names associated with the barcode
    out = subp.run( 'samtools view %s | grep RX:Z:%s | cut -f 1 | sort | uniq ' % ( hisat_bam, bc ),
                    stdout = subp.PIPE,
                    stderr = subp.PIPE,
                    shell = True,
                )

    if print_err:
        #the formatting of the output is annoying - trying to make it look nice
        err_nl = out.stderr.decode('utf-8').split( '\n' )
        print(*err_nl, sep='\n')

    read_names = out.stdout.decode('utf-8').split( '\n' )[ : -1 ]
    barcode = [ bc ]*len( read_names )
    empty_col = [ None ]*len( read_names )

    outtbl = pd.DataFrame( list( zip( barcode, empty_col, empty_col, empty_col, empty_col ) ),
                           columns = [ 'barcode', 'hisat_prim_cigar', 'hisat_alt_cigar', 'gmap_prim_cigar', 'gmap_alt_cigar' ],
                           index = read_names )

    outtbl.index.name = 'readname'

    return outtbl

def primary_secondary_align_bctbl( bc,
                                   hisat_bam,
                                   gmap_bam,
                                   print_err = False
                                 ):

    #creates a table with readnames as the index
    outtbl = create_rownames_align_bctbl( bc,
                                          hisat_bam,
                                          print_err )

    with pysam.AlignmentFile( hisat_bam, 'rb' ) as hisat:

        for read in hisat:

            if read.get_tag( 'RX' ) == bc:

                if read.is_unmapped:
                    outtbl.loc[ read.query_name ].hisat_prim_cigar = '*'
                elif not( read.is_secondary ):
                    outtbl.loc[ read.query_name ].hisat_prim_cigar = read.cigarstring
                else:
                    outtbl.loc[ read.query_name ].hisat_alt_cigar = read.cigarstring

    with pysam.AlignmentFile( gmap_bam, 'rb' ) as gmap:

        for read in gmap:

            if read.get_tag( 'RX' ) == bc:

                #read is unmapped
                if read.is_unmapped:
                    outtbl.loc[ read.query_name ].gmap_prim_cigar = '*'
                elif read.get_tag( 'HI' ) == 1:
                    outtbl.loc[ read.query_name ].gmap_prim_cigar = read.cigarstring
                else:
                    outtbl.loc[ read.query_name ].gmap_alt_cigar = read.cigarstring

    outtbl[ 'same_primary' ] = outtbl.hisat_prim_cigar == outtbl.gmap_prim_cigar

    return outtbl
