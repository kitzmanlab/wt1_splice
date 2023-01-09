import pandas as pd
import numpy as np
import random
from datetime import date

def vcf_header( build = 'hg19' ):

    today = date.today().strftime("%Y%m%d")

    hg19_chrom = [ ( 1, 249250621 ),
                   ( 2, 243199373 ),
                   ( 3, 198022430 ),
                   ( 4, 191154276 ),
                   ( 5, 180915260 ),
                   ( 6, 171115067 ),
                   ( 7, 159138663 ),
                   ( 8, 146364022 ),
                   ( 9, 141213431 ),
                   ( 10, 135534747 ),
                   ( 11, 135006516 ),
                   ( 12, 133851895 ),
                   ( 13, 115169878 ),
                   ( 14, 107349540 ),
                   ( 15, 102531392 ),
                   ( 16, 90354753 ),
                   ( 17, 81195210 ),
                   ( 18, 78077248 ),
                   ( 19, 59128983 ),
                   ( 20, 63025520 ),
                   ( 21, 48129895 ),
                   ( 22, 51304566 ),
                   ( 'X', 155270560),
                   ( 'Y', 59373566) ]

    header = '##' + '\n##'.join( [ 'fileformat=VCFv4.2',
                                    'fileDate=' + today,
                                    'reference=GRCh37/hg19'] )

    chrom_lengths = '##contig=' + '\n##contig='.join( [ '<ID=' + str( chrom ) + ',length=' + str( bp ) + '>'
                                                          for chrom, bp in hg19_chrom ] )

    ts_colhead = '#' + '\t'.join( [ 'CHROM',
                                    'POS',
                                    'ID',
                                    'REF',
                                    'ALT',
                                    'QUAL',
                                    'FILTER',
                                    'INFO' ] ) + '\n'

    return '\n'.join( [ header, chrom_lengths, ts_colhead ] )

def tup_to_vcf( tuples_l ):

    rows = [ '\t'.join( ( str( tup[ 0 ] ), str( tup[ 1 ] ), '.', tup[ 2 ], tup[ 3 ], '.', '.', '.' ) )
             for tup in tuples_l ]

    outvars = '\n'.join( rows ) + '\n'

    return outvars

def create_input_vcf( header,
                      varlist,
                      outfiledir ):

    with open( outfiledir, 'w' ) as outfile:
        outfile.write( header )
        outfile.write( varlist )

def vcf_header_chr( build = 'hg19' ):

    today = date.today().strftime("%Y%m%d")

    hg19_chrom = [ ( 'chr1', 249250621 ),
                   ( 'chr2', 243199373 ),
                   ( 'chr3', 198022430 ),
                   ( 'chr4', 191154276 ),
                   ( 'chr5', 180915260 ),
                   ( 'chr6', 171115067 ),
                   ( 'chr7', 159138663 ),
                   ( 'chr8', 146364022 ),
                   ( 'chr9', 141213431 ),
                   ( 'chr10', 135534747 ),
                   ( 'chr11', 135006516 ),
                   ( 'chr12', 133851895 ),
                   ( 'chr13', 115169878 ),
                   ( 'chr14', 107349540 ),
                   ( 'chr15', 102531392 ),
                   ( 'chr16', 90354753 ),
                   ( 'chr17', 81195210 ),
                   ( 'chr18', 78077248 ),
                   ( 'chr19', 59128983 ),
                   ( 'chr20', 63025520 ),
                   ( 'chr21', 48129895 ),
                   ( 'chr22', 51304566 ),
                   ( 'chrX', 155270560),
                   ( 'chrY', 59373566) ]

    header = '##' + '\n##'.join( [ 'fileformat=VCFv4.2',
                                    'fileDate=' + today,
                                    'reference=GRCh37/hg19'] )

    chrom_lengths = '##contig=' + '\n##contig='.join( [ '<ID=' + str( chrom ) + ',length=' + str( bp ) + '>'
                                                          for chrom, bp in hg19_chrom ] )

    ts_colhead = '#' + '\t'.join( [ 'CHROM',
                                    'POS',
                                    'ID',
                                    'REF',
                                    'ALT',
                                    'QUAL',
                                    'FILTER',
                                    'INFO' ] ) + '\n'

    return '\n'.join( [ header, chrom_lengths, ts_colhead ] )

def sample_variants( gtf_file,
                     var_per_trans,
                     refseq_d,
                     prop_ex = .5,
                     prop_can = .1,
                     seed = 1005 ):

    gtf = gtf_file.sort_values( by = 'chrom' ).copy()

    n_ex = int( var_per_trans*prop_ex )
    n_int = var_per_trans - n_ex
    n_canon = int( n_int*prop_can )

    outd = { 'chrom': [],
             'strand': [],
             'transcript': [],
             'pos': [],
             'exonic': [] }

    random.seed( seed )

    prev_chrom = None

    for t_id in gtf.transcript_id.unique():

        trans = gtf.loc[ gtf.transcript_id == t_id ].copy()

        chrom = str( trans.iloc[ 0 ].chrom )
        strand = str( trans.iloc[ 0 ].strand )

        cds_bd = ( int( trans.loc[ trans.feature == 'CDS' ].start.min() ),
                   int( trans.loc[ trans.feature == 'CDS' ].end.max() ) )

        cds = list( np.arange( cds_bd[ 0 ], cds_bd[ 1 ] ) )

        exons = trans.loc[ ( trans.feature == 'exon' ) ].copy()

        if len( exons ) == 0:

            print( 'Transcript %s skipped since no exons are coding' % t_id )
            continue

        ex_cds = list( set( np.concatenate( [ np.arange( start, end + 1 )
                                              for start, end in zip( exons.start, exons.end ) ] ) ).intersection( set( cds ) ) )

        if len( ex_cds ) < n_ex:

            print( 'Transcript %s will have only %i exonic positions selected' % ( t_id, len( ex_cds ) ) )

        if len( ex_cds ) > 0:

            ex_filt = min( n_ex, len( ex_cds ) )

            outd[ 'chrom' ].extend( [ chrom ]*ex_filt )
            outd[ 'strand' ].extend( [ strand ]*ex_filt )
            outd[ 'transcript' ].extend( [ t_id ]*ex_filt )
            outd[ 'pos' ].extend( random.sample( ex_cds, ex_filt ) )
            outd[ 'exonic' ].extend( [ True ]*ex_filt )

        int_cds = set( cds ).difference( set( ex_cds ) )

        if len( int_cds ) < n_int:

            print( 'Transcript %s will have only %i intronic positions selected' % ( t_id, len( int_cds ) ) )

        if len( int_cds ) > 0:

            int_filt = min( n_int, len( int_cds ) )

            outd[ 'chrom' ].extend( [ chrom ]*int_filt )
            outd[ 'strand' ].extend( [ strand ]*int_filt )
            outd[ 'transcript' ].extend( [ t_id ]*int_filt )
            outd[ 'pos' ].extend( random.sample( int_cds, int_filt ) )
            outd[ 'exonic' ].extend( [ False ]*int_filt )

    outdf = pd.DataFrame( outd ).sort_values( by = [ 'chrom', 'strand', 'pos' ] )

    alt_nts = { 'A', 'C', 'G', 'T' }

    outdf[ 'ref' ] = [ refseq_d[ chrom ][ pos - 1 ].upper()
                       for chrom, pos in zip( outdf.chrom, outdf.pos ) ]

    outdf[ 'alt' ] = [ list( alt_nts.difference( r ) )[ random.randint( 0, 2 ) ] for r in outdf.ref ]

    return outdf

def select_transcript( gtf,
                       selected_genes,
                       n_transcripts = 1,
                       seed = 1005 ):

    trans = gtf.loc[ gtf.feature == 'transcript' ].copy()

    random.seed( seed )

    select_trans = { gene: random.sample( trans.loc[ trans.gene_id == gene ].transcript_id.unique().tolist(),
                                          n_transcripts )
                     for gene in selected_genes }

    return select_trans

def sample_genes( gtf,
                  n_genes,
                  seed = 1005 ):

    genes = gtf.copy()

    gene_ids = genes.gene_id.unique().tolist()

    random.seed( seed )

    selected_genes = random.sample( gene_ids, 100 )

    return selected_genes

def random_vcf( gtf_file,
                refseq_dir,
                n_genes,
                n_variants,
                n_transcripts = 1,
                seed = 1005,
                prop_ex = .5,
                prop_can = .1, ):

    gtf = gtf_file.copy()

    gtf[ 'gene_id' ] = [ a.split( ' ' )[ 1 ].replace( '"', '' )
                         for att in gtf.attribute
                         for a in att.split( '; ' )
                         if a.split( ' ' )[ 0 ] == 'gene_id'  ]

    gtf.loc[ gtf.feature != 'gene', 'transcript_id' ] = [ a.split( ' ' )[ 1 ].replace( '"', '' )
                                                          for att in gtf.attribute
                                                          for a in att.split( '; ' )
                                                          if a.split( ' ' )[ 0 ] == 'transcript_id'  ]

    select_genes = sample_genes( gtf,
                                 n_genes,
                                 seed )

    genes = gtf.set_index( 'gene_id' ).loc[ select_genes ].reset_index().copy()

    select_trans_d = select_transcript( genes,
                                        select_genes,
                                        n_transcripts,
                                        seed )

    trans = genes.set_index( 'transcript_id' ).loc[ [ val[ 0 ] for val in select_trans_d.values() ] ].reset_index().copy()

    var_per_trans = int( n_variants / len( trans.transcript_id.unique() ) )

    refseq_d = { chrom: pp.get_refseq( refseq_dir + chrom + '.fa' )[ 0 ] for chrom in trans.chrom.unique().tolist() }

    tbv = sample_variants( trans,
                           var_per_trans,
                           refseq_d,
                           prop_ex,
                           prop_can,
                           seed )

    out_gtf = pd.concat( [ trans, genes.loc[ genes.feature == 'gene' ].copy() ],
                         ignore_index = True ).sort_values( by = [ 'chrom', 'start' ] )

    out_gtf = out_gtf.drop( columns = [ 'gene_id', 'transcript_id' ] )

    return tbv, out_gtf

def vcf_to_tbv( pysam_vcf,
                exon_annot ):

    exons = exon_annot.copy()

    out = { 'chrom': [],
            'pos': [],
            'ref': [],
            'alt': [],
            'strand': [],
            'nearest_bd': [],
            'exon': [],
            'exon_start': [],
            'exon_end': [] }

    chrs_annot = { chrom: exons.loc[ exons.chrom == chrom ].copy()
                   for chrom in exons.chrom.unique() }

    counter = 0

    for var in pysam_vcf:

        chrom = var.chrom
        pos = int( var.pos )

        out[ 'chrom' ].append( chrom )
        out[ 'pos' ].append( pos )
        out[ 'ref' ].append( var.ref )

        alts = var.alts

        assert len( alts ) == 1, \
        'More alts than expected for variant %s:%i%s!' % ( chrom, pos, out[ 'ref' ][ -1 ] )

        out[ 'alt' ].append( alts[ 0 ] )
        #this looks to complicated but its hard to find the right exon for the intronic variants
        out[ 'strand' ].append( chrs_annot[ 'chr' + chrom ].loc[ chrs_annot[ 'chr' + chrom ][ [ 'start', 'end' ] ].apply( lambda x: abs( x - pos )
                                                                                                                       ).min( axis = 1
                                                                                                                       ).idxmin()
                                                               ].strand )

        out[ 'nearest_bd' ].append( int( chrs_annot[ 'chr' + chrom ][ [ 'start', 'end' ] ].apply( lambda x: abs( x - pos )
                                                                                                ).min( axis =1
                                                                                                ).min() ) )


        exon = chrs_annot[ 'chr' + chrom ].loc[ ( chrs_annot[ 'chr' + chrom ].start <= pos ) \
                                                & ( chrs_annot[ 'chr' + chrom ].end >= pos ) ].copy()

        assert len( exon ) <= 1, 'Position is intersecting more than one exon for variant %s:%i%s>%s!' % ( chrom, pos, out[ 'ref' ][ -1 ], alts[ 0 ] )

        #so its an intronic variant
        if len( exon ) == 0:

            out[ 'exon' ].append( False )
            out[ 'exon_start' ].append( np.nan )
            out[ 'exon_end' ].append( np.nan )

        else:

            out[ 'exon' ].append( True )
            out[ 'exon_start' ].append( int( exon.start ) - 1 )
            out[ 'exon_end' ].append( int( exon.end ) )

        counter += 1

        if counter % 1000 == 0:

            print( '%i variants processed...' % counter )

    outdf = pd.DataFrame( out )

    return outdf
