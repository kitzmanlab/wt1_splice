import pandas as pd
import numpy as np
import pysam
import subprocess as subp
from os import path
import itertools

def trim_Tn5_adaptors( infq1,
                       infq2,
                       outdir = None,
                       outfiles = None,
                       print_out = True,
                       print_err = False ):

    #if outdirectory is unspecified, save in same directory as input
    if not outdir:
        outdir = ''.join( fq1.split( '/' )[ : -1 ] )

    #sets outfile names
    if not outfiles:
        #if not provided same as input file with the indication it was trimmed
        outfq1 = '.'.join( infq1.split( '/' )[ -1 ].split( '.' )[ :-1 ] ) + '.trim.fq.gz'
        outfq2 = '.'.join( infq2.split( '/' )[ -1 ].split( '.' )[ :-1 ] ) + '.trim.fq.gz'
    else:
        outfq1 = outfiles[ 0 ]
        outfq2 = outfiles[ 1 ]

    out = subp.run( 'cutadapt \
                    -a CTGTCTCTTATACACATCTCCGAGCCCACGAGAC \
                    -A CTGTCTCTTATACACATCTGACGCTGCCGACGA \
                    --minimum-length 20 \
                    -q 15 \
                    -O 12 \
                    -e 0.1 \
                    -o %s \
                    -p %s \
                    %s %s' % ( outdir+outfq1, outdir+outfq2, infq1, infq2 ),
                    stdout = subp.PIPE,
                    stderr = subp.PIPE,
                    shell = True,
                )

    if print_out:
        #the formatting of the output is annoying - trying to make it look nice
        out_nl = out.stdout.decode('utf-8').split( '\n' )
        print(*out_nl, sep='\n')

    if print_err:
        #the formatting of the output is annoying - trying to make it look nice
        err_nl = out.stderr.decode('utf-8').split( '\n' )
        print(*err_nl, sep='\n')

def filter_by_snv( satbl ):

    sa = satbl.copy()

    sa_singlevar = sa.query( 'n_variants_passing == 1' ).copy()

    sa_singlevar[ 'ref' ] = [ str( s ).split( ':' )[ 2 ] for s in sa_singlevar.variant_list ]
    sa_singlevar[ 'alt' ] = [ str( s ).split( ':' )[ 3 ] for s in sa_singlevar.variant_list ]

    sa_snvs = sa_singlevar.loc[ (sa_singlevar.ref.str.len() == 1) & (sa_singlevar.alt.str.len() == 1) ].copy()

    return sa_snvs

def unzip( infile,
            outfile = 'temp',
           directory = None ):

    file_ext = infile.split( '.' )[ -2 ]

    subp.run( 'zcat %s > %s' % ( infile, directory + outfile + '.' + file_ext  ),
                    shell = True,
                )

def extract_snv_bcs( satbl ):

    sa = satbl.reset_index().copy()

    var = sa.variant_list.tolist()
    bc = sa.readgroupid.tolist()

    var_to_bc_d = {}
    for v, b in zip(var, bc):
        if v in var_to_bc_d:
            var_to_bc_d[v].append(b)
        else:
            var_to_bc_d[v] = [ b ]

    return var_to_bc_d

def create_varseq( refseq,
                    variant ):

    pos = int( variant.split( ':' )[ 1 ] )
    ref = str( variant.split( ':' )[ 2 ] )
    alt = str( variant.split( ':' )[ 3 ] )

    assert refseq[ pos - 1 ] == ref, 'Reference sequence does not match variant reference allele'

    return refseq[ : ( pos -1 ) ] + alt + refseq[ pos : ]

def write_temp_fa( refseq,
                    variant,
                    tempdir ):

    with open( tempdir + 'temp.fa', 'w' ) as fa:
            fa.write( '>' + variant + '\n' )

            if variant == 'WT':
                fa.write( refseq )
            else:
                fa.write( create_varseq( refseq,
                                         variant ) )

def build_index( tempdir,
                variant,
                reffile = 'temp.fa',
                print_out = True,
                print_err = False ):

    out = subp.run( '/home/smithcat/bin/gmap_build -d %s -D %s -k 8 -w 0 %s' % ( variant, tempdir + 'indices/', tempdir + reffile ),
                    stdout = subp.PIPE,
                    stderr = subp.PIPE,
                    shell = True,
                )

    if print_out:
        #the formatting of the output is annoying - trying to make it look nice
        out_nl = out.stdout.decode('utf-8').split( '\n' )
        print(*out_nl, sep='\n')

    if print_err:
        #the formatting of the output is annoying - trying to make it look nice
        err_nl = out.stderr.decode('utf-8').split( '\n' )
        print(*err_nl, sep='\n')

def convert_fq_to_bam( fq_file1,
                       fq_file2,
                       sample_name,
                       tempdir,
                       append_bc = True ):

    if append_bc:

        #convert forward fq to unaligned bam
        subp.run( 'java -Xmx8g -jar /nfs/kitzman2/lab_software/platform_indep/picard-tools-2.9.0/picard.jar \
                    FastqToSam F1=%s SM=%s O=%s' % ( fq_file1, sample_name, tempdir + 'temp.bam' ),
             shell = True,
             stdout = subp.PIPE,
             stderr = subp.PIPE )

        append_bcs( tempdir, fq_file2 )

        subp.run( 'samtools sort -@8 -m8G -o %s -t RX %s' % ( tempdir + sample_name + '_unaligned.bam', tempdir + 'temp_bc.bam' ),
            shell = True,
            stdout = subp.PIPE,
            stderr = subp.PIPE )

    else:

        #convert forward fq to unaligned bam
        subp.run( 'java -Xmx8g -jar /nfs/kitzman2/lab_software/platform_indep/picard-tools-2.9.0/picard.jar \
                    FastqToSam F1=%s SM=%s O=%s' % ( fq_file1, sample_name, tempdir + sample_name + '_unaligned1.bam' ),
             shell = True,
             stdout = subp.PIPE,
             stderr = subp.PIPE )

        subp.run( 'java -Xmx8g -jar /nfs/kitzman2/lab_software/platform_indep/picard-tools-2.9.0/picard.jar \
                    FastqToSam F1=%s SM=%s O=%s' % ( fq_file2, sample_name, tempdir + sample_name + '_unaligned2.bam' ),
             shell = True,
             stdout = subp.PIPE,
             stderr = subp.PIPE )


def write_temp_fq(  pysam_align_in,
                    reads,
                    tempdir,
                    print_err = False ):

    with pysam.AlignmentFile( tempdir + 'temp.bam', 'wb', template = pysam_align_in ) as bam_out:

        for read in reads:
            bam_out.write( read )

    out = subp.run( 'samtools fastq --threads 8 %s > %s'
                    % ( tempdir + 'temp.bam', tempdir + 'temp.fq' ) ,
                    shell = True,
                    stderr = subp.PIPE
                    )

    if print_err:
        #the formatting of the output is annoying - trying to make it look nice
        err_nl = out.stderr.decode('utf-8').split( '\n' )
        print(*err_nl, sep='\n')

def align_reads( tempdir,
                 variant,
                  fqfile = 'temp.fq',
                  print_err = False,
                  outbam = 'temp.bam'
                   ):


    out = subp.run( '/home/smithcat/bin/gmap -d %s -D %s -t 8 -f samse --microexon-spliceprob=1.0 --allow-close-indels=2 %s > %s'
                        % ( variant, tempdir + 'indices/', fqfile, tempdir + 'temp.sam' ) ,
                        stderr = subp.PIPE,
                        shell = True,
                        )

    subp.run( 'samtools view -S -b %s > %s'
                    % ( tempdir + 'temp.sam', tempdir + outbam ) ,
                    stderr = subp.PIPE,
                    shell = True,
                    )

    if print_err:
        #the formatting of the output is annoying - trying to make it look nice
        err_nl = out.stderr.decode('utf-8').split( '\n' )
        print(*err_nl, sep='\n')

def append_bcs( tempdir,
                fq_file2,
                bam_in = 'temp.bam',
                bcbam_out = 'temp_bc.bam' ):

    #add BCs from reverse fq to unaligned bam
    out = subp.run( 'java -Xmx8g -jar /nfs/kitzman2/lab_software/platform_indep/fgbio-0.8.1/fgbio-0.8.1.jar \
                    AnnotateBamWithUmis --fail-fast=true --input=%s --fastq=%s --output=%s'
                    % ( tempdir + bam_in, fq_file2, tempdir + bcbam_out ),
                    shell = True,
                    stdout = subp.PIPE,
                    stderr = subp.PIPE )

def coordsort_bam( tempdir,
                    bamfile = 'temp.bam',
                    print_err = False ):

    out = subp.run( 'samtools sort %s > %s ' % ( tempdir + 'temp.sam', tempdir + bamfile ),
                    stderr = subp.PIPE,
                    shell = True,
                    )

    if print_err:
        #the formatting of the output is annoying - trying to make it look nice
        err_nl = out.stderr.decode('utf-8').split( '\n' )
        print(*err_nl, sep='\n')

def align_pe_reads( tempdir,
                    variant,
                    fqfile1 = 'for.temp.fq',
                    fqfile2 = 'rev.temp.fq',
                    print_err = False,
                    outbam = 'temp.bam'
                   ):


    out = subp.run( '/home/smithcat/bin/gsnap -d %s -D %s -N 1 -t 8 --format sam %s %s > %s'
                        % ( variant, tempdir + 'indices/', fqfile1, fqfile2, tempdir + 'temp.sam' ) ,
                        stderr = subp.PIPE,
                        shell = True,
                        )

    subp.run( 'samtools view -S -b %s > %s'
                    % ( tempdir + 'temp.sam', tempdir + outbam ) ,
                    stderr = subp.PIPE,
                    shell = True,
                    )

    if print_err:
        #the formatting of the output is annoying - trying to make it look nice
        err_nl = out.stderr.decode('utf-8').split( '\n' )
        print(*err_nl, sep='\n')

def align_sample( satbl,
                  refseq,
                  tempdir,
                  fq_file1,
                  fq_file2,
                  chrom_name,
                  sample_name = 'sample',
                  create_indices = False,
                  print_err = False
                   ):

    sa_snvs = filter_by_snv( satbl )

    var_to_bc_d = extract_snv_bcs( sa_snvs )

    #don't need to recreate indices for every sample
    if create_indices:

        #create folder to store indices
        subp.run( 'mkdir %s' % ( tempdir + 'indices/' ),
                        shell = True,
                        )

        #for each variant create a temporary fa file
        #then create an index within indices directory named by the variant
        for var in var_to_bc_d.keys():

                write_temp_fa( refseq, var, tempdir )
                build_index( tempdir, var )

    convert_fq_to_bam( fq_file1, fq_file2, sample_name, tempdir )

    #import with check_sq as False since this bam is unaligned
    sample_bam = pysam.AlignmentFile( tempdir + sample_name + '_unaligned.bam', 'rb', check_sq = False )

    #fake header junk
    header = { 'HD': {'VN': '1.5'},
            'SQ': [{'LN': 6490, 'SN': chrom_name}] }

    with pysam.AlignmentFile( tempdir + sample_name + '.bam', 'wb', header = header ) as bam_out:

        for bc, _reads in itertools.groupby( sample_bam, lambda _r: _r.get_tag( 'RX' ) ):

            if bc not in sa_snvs.index:
                #print( 'Barcode:', bc, 'not in the subassembly table')
                continue

            var = satbl.loc[ bc ].variant_list

            #possible that _reads will work.... dunno
            reads = list(_reads)

            write_temp_fq( sample_bam, reads, tempdir, print_err )

            align_reads( tempdir, var, tempdir + 'temp.fq' )

            with pysam.AlignmentFile( tempdir + 'temp.bam', 'rb' ) as bc_bam:

                for read in bc_bam:
                    bam_out.write( read )

    append_bcs( tempdir, fq_file2, bam_in = sample_name + '.bam', bcbam_out = sample_name + '_bc.bam' )

    sample_bam.close()

def filter_secondary_alignments( tempdir,
                                 infile ):

    outfile = '.'.join( infile.split( '.' )[ :-1 ] )

    with pysam.AlignmentFile( tempdir + infile, 'rb' ) as unfiltered_bam:

        with pysam.AlignmentFile( tempdir + outfile + '_filt.bam', 'wb', template = unfiltered_bam ) as filt_bam:

            for read in unfiltered_bam:

                if not( read.has_tag( 'HI' ) ) or read.get_tag( 'HI' ) == 1:
                    filt_bam.write( read )


def align_indiv_sample( variant,
                        refseq,
                        tempdir,
                        fq_file1,
                        chrom_name,
                        fq_file2 = None,
                        sample_name = 'sample',
                        create_indices = False,
                        print_err = False
                      ):

    #don't need to recreate indices for every sample
    if create_indices:

        #create folder to store indices
        subp.run( 'mkdir %s' % ( tempdir + 'indices/' ),
                        shell = True,
                        )

        #for the variant create a temporary fa file
        #then create an index within indices directory named by the variant
        write_temp_fa( refseq, variant, tempdir )
        build_index( tempdir, variant, print_err = print_err )

    if not fq_file2:

        align_reads( tempdir,
                     variant,
                     fq_file1,
                     outbam = sample_name + '.bam',
                     print_err = print_err )

    else:

        align_pe_reads( tempdir,
                        variant,
                        fq_file1,
                        fq_file2,
                        outbam = sample_name + '.bam',
                        print_err = print_err )

def align_WT_sample( wt_bcs,
                     refseq,
                     tempdir,
                     fq_file1,
                     fq_file2,
                     chrom_name,
                     sample_name = 'sample',
                     create_indices = False,
                     print_err = False
                   ):

    var_to_bc_d = { 'WT': wt_bcs }

    #don't need to recreate indices for every sample
    if create_indices:

        #create folder to store indices
        subp.run( 'mkdir %s' % ( tempdir + 'indices/' ),
                        shell = True,
                        )

        #for each variant create a temporary fa file
        #then create an index within indices directory named by the variant
        for var in var_to_bc_d.keys():

                write_temp_fa( refseq, var, tempdir )
                build_index( tempdir, var )

    convert_fq_to_bam( fq_file1, fq_file2, sample_name , tempdir )

    #import with check_sq as False since this bam is unaligned
    sample_bam = pysam.AlignmentFile( tempdir + sample_name + '_unaligned.bam', 'rb', check_sq = False )

    #fake header junk
    header = { 'HD': {'VN': '1.5'},
            'SQ': [{'LN': 6490, 'SN': chrom_name}] }

    with pysam.AlignmentFile( tempdir + sample_name + '_wt.bam', 'wb', header = header ) as bam_out:

        for bc, _reads in itertools.groupby( sample_bam, lambda _r: _r.get_tag( 'RX' ) ):

            if bc not in wt_bcs:
                #print( 'Barcode:', bc, 'not in the subassembly table')
                continue

            var = 'WT'

            reads = list(_reads)

            write_temp_fq( sample_bam, reads, tempdir, print_err )

            align_reads( tempdir, var, tempdir + 'temp.fq' )

            with pysam.AlignmentFile( tempdir + 'temp.bam', 'rb' ) as bc_bam:

                for read in bc_bam:
                    bam_out.write( read )

    append_bcs( tempdir, fq_file2, bam_in = sample_name + '_wt.bam', bcbam_out = sample_name + '_bc_wt.bam' )

    sample_bam.close()
