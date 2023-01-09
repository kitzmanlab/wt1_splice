import altair as alt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as tfrms
import matplotlib.patches as patches
from matplotlib.artist import Artist as art
from matplotlib import gridspec
from collections import Counter
import splanl.coords as cds
import splanl.merge_bcs as mbcs
import splanl.post_processing as pp
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, matthews_corrcoef
import scipy.stats as ss
import upsetplot as up
import random
from scipy.integrate import simps

github_colors = '3182bd6baed69ecae1c6dbefe6550dfd8d3cfdae6bfdd0a231a35474c476a1d99bc7e9c0756bb19e9ac8bcbddcdadaeb636363969696bdbdbdd9d9d9'
light_colors = [ '#' + github_colors[i:i+6] for i in range( 0, len( github_colors ), 6 ) ]

def waterfall_plot( bctbl,
                    col_read_counts,
                    percentile_l,
                    title='',
                    x_ax_title='Barcode Rank Log10',
                    y1_ax_title='Read Count Log10',
                    y2_ax_title='Cumulative Read Count Percentile' ):

    bc_ranks = bctbl.copy()

    bc_ranks = bc_ranks.sort_values( by = col_read_counts,
                                      ascending = False, )

    bc_ranks[ col_read_counts + '_rank' ] = np.arange( bc_ranks.shape[ 0 ] )
    bc_ranks[ col_read_counts + '_log10' ] = np.log10( bc_ranks[ col_read_counts ] + .1 )
    bc_ranks[ col_read_counts + '_rank_log10' ] = np.log10( bc_ranks[ col_read_counts + '_rank' ] + 1 )
    bc_ranks[ 'cumulative_read_percentage' ] = 100*( bc_ranks[ col_read_counts ].cumsum() / bc_ranks[ col_read_counts ].sum() )

    percentile_cutoff = {}
    read_count_cutoff = {}
    for per in percentile_l:

        percentile_cutoff[ per ] = bc_ranks.loc[ bc_ranks.cumulative_read_percentage >= per ][ col_read_counts + '_rank_log10' ].reset_index( drop = True )[ 0 ]
        read_count_cutoff[ per ] = bc_ranks.loc[ bc_ranks.cumulative_read_percentage >= per ][ col_read_counts ].reset_index( drop = True )[ 0 ]

    fig, ax1 = plt.subplots()

    color = 'blue'
    ax1.set_xlabel( x_ax_title )
    ax1.set_ylabel( y1_ax_title ,
                    color=color)
    ax1.plot( bc_ranks[ col_read_counts + '_rank_log10' ],
              bc_ranks[ col_read_counts + '_log10' ],
              color = color )
    ax1.tick_params( axis = 'y', labelcolor = color )

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'red'
    ax2.set_ylabel( y2_ax_title, color = color )  # we already handled the x-label with ax1
    ax2.plot( bc_ranks[ col_read_counts + '_rank_log10' ],
              bc_ranks.cumulative_read_percentage,
              color = color)
    ax2.tick_params( axis = 'y', labelcolor = color)

    plt.vlines( list( percentile_cutoff.values() ),
                ymin = -1,
                ymax = 100,
                color = 'black' )

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    for per in percentile_l:

        print( 'The read count cut off at the', per, 'th percentile is', read_count_cutoff[ per ] )

    return percentile_cutoff, read_count_cutoff


##########################################################################################
##########################################################################################
##########################################################################################
##
##  psi by coordinate plots
##


##########################################################################################
## altair-style, one sample

def altplot_psi_by_pos_onesamp(
    tblbyvar,
    col_iso_y,
    addl_cols_tooltip,
    fill_col='alt',
    lposrng_exon=None,
    yscale=None,
    height=100,
    width=800,
    title='',
    x_ax_title='pos',
    y_ax_title=None
):
    assert (tblbyvar.ref.str.len() == 1).all()
    assert (tblbyvar.alt.str.len() == 1).all()

    tbv = tblbyvar.copy()
    tbv['pos_display'] = tbv['pos']
    for ofs,mut in zip( (0.1,0.3,0.5,0.7), 'ACGT' ):
        tbv.loc[ tbv.alt==mut, 'pos_display' ]+=ofs

    if yscale is None:
        y = alt.Y('{}:Q'.format(col_iso_y), axis=alt.Axis(title=y_ax_title) )
    else:
        y = alt.Y('{}:Q'.format(col_iso_y), scale=yscale, axis=alt.Axis(title=y_ax_title) )

    if y_ax_title is None:
        y_ax_title=col_iso_y

    points = alt.Chart( tbv , title=title ).mark_circle(
        size=15
    ).encode(
        x=alt.X('pos_display:Q', scale=alt.Scale(zero=False) ,
        axis=alt.Axis(title=x_ax_title)),#, sort=list( pos_display_categ.categories )),
        y=y,
        tooltip=['pos','ref','alt']+addl_cols_tooltip,
        fill=alt.Color(fill_col+':N',scale=alt.Scale(scheme='category10'))
    ).properties( height=height, width=width )

    gr = points

    if lposrng_exon is not None:
        exonbox=pd.DataFrame( {'x':[lposrng_exon[i][0] for i in range(len(lposrng_exon))],
                              'x2':[lposrng_exon[i][1] for i in range(len(lposrng_exon))],
                             'y':tblbyvar[col_iso_y].min(), 'y2':tblbyvar[col_iso_y].max()} )
        ex = alt.Chart( exonbox ).mark_rect( color='yellow', opacity=0.1 ).encode( x='x', x2='x2', y='y', y2='y2' )
        gr = ex + gr

    return gr

##########################################################################################
## altair-style, one sample w/ zoom

def altplot_psi_by_pos_onesamp_wzoom(
    tblbyvar,
    col_iso_y,
    addl_cols_tooltip,
    fill_col='alt',
    lposrng_exon=None,
    yscale=None,
    height=100,
    width=800,
    zoom_frac = 0.5,
    title=''
):

    zoom_frac = max(0.1, min(0.9, zoom_frac))

    assert (tblbyvar.ref.str.len() == 1).all()
    assert (tblbyvar.alt.str.len() == 1).all()

    tbv = tblbyvar.copy()
    tbv['pos_display'] = tbv['pos']
    for ofs,mut in zip( (0.1,0.3,0.5,0.7), 'ACGT' ):
        tbv.loc[ tbv.alt==mut, 'pos_display' ]+=ofs

    if yscale is None:
        y = alt.Y('{}:Q'.format(col_iso_y))
    else:
        y = alt.Y('{}:Q'.format(col_iso_y), scale=yscale )

    selbrush = alt.selection(type='interval', encodings=['x'])

    points = alt.Chart( tbv, title=title ).mark_point(
        size=15
    ).encode(
        x=alt.X('pos_display:Q', scale=alt.Scale(zero=False, domain=selbrush)),#, sort=list( pos_display_categ.categories )),
        y=y,
        tooltip=['pos','ref','alt']+addl_cols_tooltip,
        # fill='alt:N'
        fill=alt.Color(fill_col+':N',scale=alt.Scale(scheme='category10'))
    ).properties( height=height, width=int(1-zoom_frac)*width )

    gr = points

    if lposrng_exon is not None:
        exonbox=pd.DataFrame( {'x':[lposrng_exon[i][0] for i in range(len(lposrng_exon))],
                              'x2':[lposrng_exon[i][1] for i in range(len(lposrng_exon))],
                             'y':np.min(tblbyvar.col_iso_y), 'y2':1} )
        ex = alt.Chart( exonbox ).mark_rect( color='yellow', opacity=0.1 ).encode( x='x', x2='x2', y='y', y2='y2' )
        gr = ex + gr


    pointszoom = points.properties( width=int(zoom_frac)*width ).add_selection( selbrush )

    gr_ret = pointszoom | gr

    return gr_ret

def altplot_psi_by_pos_onesamp_multiiso_wzoom(
    tblbyvar,
    l_col_iso_y,
    addl_cols_tooltip,
    lposrng_exon=None,
    yscale=None,
    height=100,
    width=800,
    zoom_frac = 0.5,
    title='' ):

    zoom_frac = max(0.1, min(0.9, zoom_frac))

    assert (tblbyvar.ref.str.len() == 1).all()
    assert (tblbyvar.alt.str.len() == 1).all()

    tbv = tblbyvar.copy()
    tbv['pos_display'] = tbv['pos']
    for ofs,mut in zip( (0.1,0.3,0.5,0.7), 'ACGT' ):
        tbv.loc[ tbv.alt==mut, 'pos_display' ]+=ofs

    selbrush = alt.selection(type='interval', encodings=['x'])

    # hack to get a title over everything
    tpl = alt.Chart( {'values':[{'text':title}]} ).mark_text( size=14 ).encode( text='text:N' )

    lpoints = [tpl]

    for col_iso_y in l_col_iso_y:
        if yscale is None:
            y = alt.Y('{}:Q'.format(col_iso_y))
        else:
            y = alt.Y('{}:Q'.format(col_iso_y), scale=yscale )

        points = alt.Chart( tbv ).mark_point(
            size=15
        ).encode(
            x=alt.X('pos_display:Q', scale=alt.Scale(zero=False, domain=selbrush)),#, sort=list( pos_display_categ.categories )),
            y=y,
            tooltip=['pos','ref','alt']+addl_cols_tooltip,
            # fill='alt:N'
            fill=alt.Color('alt:N',scale=alt.Scale(scheme='category10'))
        ).properties( height=height, width=int(1-zoom_frac)*width )

        pointszoom = points.properties( width=int(zoom_frac)*width ).add_selection( selbrush )

        if lposrng_exon is not None:
            exonbox=pd.DataFrame( {'x':[lposrng_exon[i][0] for i in range(len(lposrng_exon))],
                                  'x2':[lposrng_exon[i][1] for i in range(len(lposrng_exon))],
                                 'y':0, 'y2':1} )
            ex = alt.Chart( exonbox ).mark_rect( color='yellow', opacity=0.1 ).encode( x='x', x2='x2', y='y', y2='y2' )
            points = ex + points

        lpoints.append( pointszoom | points )

    gr = alt.vconcat( *lpoints ).configure_view( stroke=None ).configure_concat( spacing=1 )

    return gr

def altplot_scatter_onesamp(
    tblbyvar,
    col_iso_y,
    col_x,
    addl_cols_tooltip=[],
    fill_col='var_type',
    yscale=None,
    height=300,
    width=300,
    title=''
):

    tbv = tblbyvar.copy()

    if yscale is None:
        y = alt.Y('{}:Q'.format(col_iso_y))
    else:
        y = alt.Y('{}:Q'.format(col_iso_y), scale=yscale )

    points = alt.Chart( tbv , title=title ).mark_circle(
        size=20
    ).encode(
        x=alt.X(col_x+':Q', scale=alt.Scale(zero=False)),#, sort=list( pos_display_categ.categories )),
        y=y,
        tooltip=addl_cols_tooltip,
        fill=alt.Color(fill_col+':N',scale=alt.Scale(scheme='category10'))
    ).properties( height=height, width=width )

    gr = points

    return gr

def altplot_scatter_nofill(
    tblbyvar,
    col_iso_y,
    col_x,
    addl_cols_tooltip=[],
    yscale=None,
    height=300,
    width=300,
    title=''
):

    tbv = tblbyvar.copy()

    if yscale is None:
        y = alt.Y('{}:Q'.format(col_iso_y))
    else:
        y = alt.Y('{}:Q'.format(col_iso_y), scale=yscale )

    points = alt.Chart( tbv , title=title ).mark_circle(
        size=20
    ).encode(
        x=alt.X(col_x+':Q', scale=alt.Scale(zero=False)),#, sort=list( pos_display_categ.categories )),
        y=y,
        tooltip=addl_cols_tooltip
    ).properties( height=height, width=width )

    gr = points

    return gr

def altplot_violin_onesamp(
    tblbyvar,
    col_y,
    col_cat_x,
    yscale=None,
    height=400,
    width=400,
    title=''
):
    assert (tblbyvar.ref.str.len() == 1).all()
    assert (tblbyvar.alt.str.len() == 1).all()

    tbv = tblbyvar.copy()

    points = alt.Chart( tbv , title=title ).transform_density(
        col_y,
        as_=[col_y,'density'],
        groupby=[col_cat_x]
    ).mark_area(
        orient='horizontal'
    ).encode(
        y=col_y+':Q',
        color=col_cat_x+':N',
        x=alt.X(
            'density:Q',
            stack='center',
            axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
        ),
        column=alt.Column(
            col_cat_x+':N',
            header=alt.Header(
                titleOrient='bottom',
                labelOrient='bottom',
                labelPadding=0,
            )
        ),
    ).properties( height=height, width=width )

    gr = points

    return gr

def PlotPSIByPos(vardf,
                 col_y_isoform,
                 shade_exons,
                 gene_name,
                 fig_size = (20,7),
                 coords = 'cdna',
                vec_corange_cloned = None,
                vec_corange_exons = None,
                cdna_corange_exons = None,
                 zoom=None,
                 tick_spacing=10,
                 legend_loc='best',
                 y_ax_title='',
                 legend_title='Nucleotide Substitution from WT',
                 y_ax_lim = (0,1),
                 invert_x = False,
                 tight = True,
                 print_ex_count = False,
                 scale_bar = False,
                 rev_trans = False,
                 hlines = None
                ):

    tbv=vardf.copy()

    tbv.sort_values( by = ['pos'], inplace = True )
    bcs_tbl = tbv.pivot( index = 'pos', columns = 'alt', values = col_y_isoform )

    if zoom:
        assert zoom[1] > zoom[0], 'Your final zoom coordinate must be larger than the first zoom coordinate'
        bcs_tbl = bcs_tbl.loc[ zoom[0]:zoom[1] ]

    #check if plot will lose bars due to not enough space for all the pixels
    dpi = 100
    if ( bcs_tbl.shape[0]*6 ) > ( fig_size[0]*dpi ):
        fig_height = fig_size[1]
        fig_size = ( ( bcs_tbl.shape[0]*6 / dpi ), fig_height )
        print('Adjusting figure width to accomodate all pixels...')

    if print_ex_count:
        print('This figure shows %i exonic bases.' % sum( c[1]-c[0] for c in shade_exons ) )

    #usually want to represent alternate as seen on the forward strand so reverse complement columns
    if rev_trans:
        bcs_tbl = bcs_tbl.rename( columns = { "A": "T", "C": "G", "G": "C", "T": "A" } )

    bcs_tbl.loc[:,['A', 'C', 'G', 'T']].plot.bar( color = [ '#C00001', '#00AD4F', '#FFCF07', '#002966' ],
                                                 align='center',
                                                 width=1,
                                                figsize= fig_size )

    plt.title(
        col_y_isoform+' '+y_ax_title+' by Position for Single Nucleotide Variants in $\it{%s}$'%gene_name,
        fontsize=24)

    if y_ax_lim:
        plt.ylim( y_ax_lim )

    plt.ylabel(col_y_isoform+' '+y_ax_title,fontsize=22)
    plt.yticks(fontsize=18)

    if coords.lower() == 'cdna':
        plt.xlabel('cDNA Position',fontsize=22)
        bcs_tbl['hgvs_pos'] = cds.pos_to_hgvspos( bcs_tbl.index,
                               vec_corange_cloned,
                               vec_corange_exons,
                               cdna_corange_exons
                             )
        plt.xticks( [idx for idx,p in enumerate(bcs_tbl.index) if idx%tick_spacing==0],
                       [c for idx,c in enumerate(bcs_tbl.hgvs_pos) if idx%tick_spacing==0],
                       fontsize=18,
                       rotation='vertical' )

    elif coords.lower() == 'gdna':
        plt.xlabel('gDNA Position',fontsize=22)
        plt.xticks( [idx for idx,p in enumerate(bcs_tbl.index) if idx%tick_spacing==0],
                   [c for idx,c in enumerate(bcs_tbl.index) if idx%tick_spacing==0],
                   fontsize=18,
                   rotation='vertical' )

    elif coords.lower() == 'vector':
        plt.xlabel('Vector Position',fontsize=22)
        plt.xticks( [idx for idx,p in enumerate(bcs_tbl.index) if idx%tick_spacing==0],
                   [c for idx,c in enumerate(bcs_tbl.index) if idx%tick_spacing==0],
                   fontsize=18,
                   rotation='vertical' )

    if hlines:
        for line in hlines:
            plt.axhline( line, c = 'black', ls = '--', alpha = .6 )

    legend = plt.legend( title = legend_title,
                         ncol = 2,
                         loc = legend_loc,
                         fontsize = 14)
    plt.setp(legend.get_title(),fontsize=14)

    for ex in shade_exons:
        plt.axvspan( bcs_tbl.index.get_loc( ex[0] ) - .5,
                    bcs_tbl.index.get_loc( ex[1] ) + .5,
                    facecolor = 'gray',
                    alpha = 0.15)

    if invert_x:
        plt.gca().invert_xaxis()

    if scale_bar:
        ax = plt.gca()
        trans = tfrms.blended_transform_factory( ax.transData, ax.transAxes )
        plt.errorbar( tick_spacing, 0.96, xerr=tick_spacing/2, color='black', capsize=3, transform=trans)
        plt.text( tick_spacing, 0.94, str( tick_spacing )+' bases',  horizontalalignment='center',
        verticalalignment='top', transform=trans, fontsize = 14 )

    if tight:
        plt.tight_layout()

    plt.show()

def PlotBCsByPos(vardf,
                 vec_corange_cloned,
                 vec_corange_exons,
                 cdna_corange_exons,
                 shade_exons,
                 gene_name,
                 y_ax_lim=None,
                 fig_size = (20,7) ):

    tbv=vardf.copy()

    tbv.sort_values(by=['pos'],inplace=True)
    bcs_tbl=tbv.pivot(index='pos',columns='alt',values='n_bc_passfilt')
    bcs_tbl['hgvs_pos'] = cds.pos_to_hgvspos( bcs_tbl.index,
                           vec_corange_cloned,
                           vec_corange_exons,
                           cdna_corange_exons
                         )

    #check if plot will lose bars due to not enough space for all the pixels
    dpi = 100
    if ( bcs_tbl.shape[0]*1.5 ) > ( fig_size[0]*dpi ):
        fig_height = fig_size[1]
        fig_size = ( ( bcs_tbl.shape[0]*1.5 / dpi ), fig_height )
        print('Adjusting figure width to accomodate all pixels...')

    bcs_tbl.loc[:,['A','C', 'G', 'T']].plot.bar( stacked = True,
                                                color = [ '#C00001', '#00AD4F', '#FFCF07', '#002966' ],
                                                figsize = fig_size )

    plt.title(
        'Number of Distinct Barcodes Present by Position for Single Nucleotide Variants in $\it{%s}$'%gene_name,
        fontsize=24)

    if y_ax_lim:
        plt.ylim(0,y_ax_lim)

    plt.ylabel('Number of Distinct Barcodes',fontsize=22)
    plt.yticks(fontsize=18)

    plt.xlabel('cDNA Position',fontsize=22)
    plt.xticks( [idx for idx,p in enumerate(bcs_tbl.index) if idx%10==0],
               [c for idx,c in enumerate(bcs_tbl.hgvs_pos) if idx%10==0],
               fontsize=18,
               rotation='vertical' )

    legend = plt.legend(title='Nucleotide Substitution from WT',
                        ncol=2,
                        loc='upper left',
                        fontsize=14)
    plt.setp(legend.get_title(),fontsize=14)

    for ex in shade_exons:
        plt.axvspan(bcs_tbl.index.get_loc(ex[0])-.5,
                    bcs_tbl.index.get_loc(ex[1])+.5,
                    facecolor='gray',
                    alpha=0.15)

    plt.tight_layout()
    plt.show()

    #counts entries that are missing - one is missing per position as its reference
    missing = sum( bcs_tbl.isnull().sum() ) - bcs_tbl.shape[0]
    #counts number of entries with 0 barcodes passing the filter
    zero = 4*bcs_tbl.shape[0] - sum( bcs_tbl[[ 'A', 'C', 'G', 'T' ]].astype( bool ).sum() )
    print( 100 * ( 1- ( ( missing + zero ) / (3*bcs_tbl.shape[0] ) ) ), '% of all possible mutations present' )

def plot_corr_waterfalls(benchmark_df,
                        compare_df_long,
                        corr_col,
                        benchmark_samp_name,
                        merge_idx = ['chrom','pos','ref','alt','varlist'],
                        sample_col = 'sample'):

    rank_df = benchmark_df.copy()[ merge_idx + [corr_col] ]
    rank_df[ corr_col+'_rank' ] =  np.argsort( np.argsort( np.array( -rank_df[ corr_col ] ) ) )
    rank_df = rank_df.set_index( merge_idx )

    compare_df = compare_df_long.copy().set_index( merge_idx )

    merge_df = compare_df.merge( rank_df[ corr_col+'_rank' ], left_index=True, right_index=True )

    for samp in list( set( merge_df[ sample_col ] ) ):
        print(samp)
        plt.scatter(merge_df.query('sample == "%s"' %samp)[ corr_col+'_rank' ],
                    merge_df.query('sample == "%s"' %samp)[ corr_col ],
                    s=5)

        plt.ylabel(samp + ' ' + corr_col)
        plt.xlabel( benchmark_samp_name + ' ' + corr_col + ' rank')
        plt.show()

def barplot_allisos( allisos_df,
                    isoform_df,
                    psi_cols,
                    stat,
                    title = ''):

    plot_df = allisos_df.copy()
    iso_df = isoform_df.copy()

    if stat == 'max':
        plot_df[ psi_cols ].max().plot( kind = 'bar',
                                        figsize = ( 40, 5 ),
                                        title = title,
                                        ylim = ( 0, 1 ) )

    if stat == 'min':
        plot_df[ psi_cols ].min().plot( kind = 'bar',
                                        figsize = ( 40, 5 ),
                                        title = title,
                                        ylim = ( 0, 1 ) )

    if stat == 'mean':
        plot_df[ psi_cols ].mean().plot( kind = 'bar',
                                        figsize = ( 40, 5 ),
                                        title = title,
                                        ylim = ( 0, 1 ) )

    loc, labels = plt.xticks()

    plt.xticks( loc,
                iso_df.loc[ [ iso.split('_')[1] for iso in psi_cols ] ].isoform.tolist(),\
               fontsize=14,
               rotation='vertical' )

plt.show()

def barplot_across_samples( byvartbl_long,
                            bar_col,
                            y_scale = None,
                            y_label = None,
                            title = '',
                            y_lim = None,
                            color_col = None,
                            color_dict = None):

    tbv = byvartbl_long.copy()

    assert ( color_col and color_dict ) or not( color_col or color_dict ), \
    "color_col and color_dict must either both be none or both entered"

    if color_col:
        tbv.set_index('sample')[ bar_col ].plot.bar( color=[ color_dict[i] for i in tbv[ color_col ] ] )
    else:
        tbv.set_index('sample')[ bar_col ].plot.bar()

    if y_label:
        plt.ylabel( y_label )

    if y_lim:
        plt.ylim( y_lim )

    if y_scale:
        plt.yscale( y_scale )

    plt.title( title )

    plt.show()

def per_sdv_by_thresh( byvartbl,
                     sdv_col,
                     thresh_range = None,
                     abs_vals = True,
                     num_pts = 20,
                     title = '',
                     y_lim = None,
                     vlines = ( 1.96, 3 ),
                     fig_size = ( 12, 9.5 ) ):

    tbv = byvartbl.loc[ byvartbl.n_bc_passfilt > 0  ].copy()

    samp = 'sample' in tbv.columns

    if thresh_range:
        thresh = np.arange( thresh_range[0],
                            thresh_range[1],
                            ( thresh_range[1] - thresh_range[0] ) / num_pts )
    else:
        min_thresh = 0 if tbv[ sdv_col ].min() <= 0 else tbv[ sdv_col ].min()
        thresh = np.arange( min_thresh,
                            tbv[ sdv_col ].max(),
                            ( tbv[ sdv_col ].max() - min_thresh ) / num_pts )

    #add vline points into array while maintaining sort order
    for pt in vlines:
        thresh = np.insert( thresh, thresh.searchsorted( float( pt ) ), float( pt ) )

    plt.figure( figsize = fig_size )

    if samp:
        for smpl in set( tbv['sample'] ):
            sdv_per = []
            for t in thresh:
                tbv_samp = tbv.query( 'sample == "%s"' % smpl ).copy()
                tbv_samp = pp.sdvs( tbv_samp, sdv_col, t, abs_vals = abs_vals )
                sdv_per.append( 100*( tbv_samp.sdv.sum() / tbv_samp.shape[0] ) )

            plt.plot( thresh, sdv_per, label = smpl )

            print( smpl )
            for pt in vlines:
                per = sdv_per[ np.where( thresh == pt )[0][0] ]
                print( 'At a threshold of %.2f, %.2f%% of variants are splice disrupting.' % ( pt, per ) )

        plt.legend()

    else:
        sdv_per = []
        for t in thresh:
            tbv = pp.sdvs( tbv.query( 'sample == "%s"' % smpl ), sdv_col, t, abs_vals = abs_vals )
            sdv_per.append( 100*( tbv.sdv.sum() / tbv.shape[0] ) )

        plt.plot( thresh, sdv_per, label = smpl )

        for pt in vlines:
            per = sdv_per[ np.where( thresh == pt ) ]
            print( 'At a threshold of %.2f, %.2f%% of variants are splice disrupting.' % ( pt, per ) )

    for pt in vlines:
        plt.axvline( x = pt, color = 'gray', linestyle = '--' )

    if y_lim:
        plt.ylim( y_lim )
    else:
        plt.ylim( ( 0, 100 ) )

    plt.ylabel( 'Percentage splice disrupting variants' )
    plt.xlabel( sdv_col + ' threshold' )

    plt.title( title )

    plt.show()

def per_repeat_sdv( byvartbl,
                     sdv_col = None,
                     thresh = None,
                     abs_vals = True,
                     title = '',
                     y_lim = None, ):

    tbv = byvartbl.loc[ byvartbl.n_bc_passfilt > 0  ].copy()

    n_var = len( set( tbv.varlist ) )

    assert ( sdv_col and thresh ) or ( 'sdv' in tbv.columns ), \
    'Please specify a column and threshold to determine splice disrupting variants'

    if ( sdv_col and thresh ):
        tbv = pp.sdvs( tbv, sdv_col, thresh, abs_vals = abs_vals )

    n_samp = len( set( tbv[ 'sample' ] ) )

    assert n_samp > 1, 'Please provide a dataset with more than one sample'

    sdvs = tbv.loc[ tbv.sdv ]

    #this gets you a counter { n_samples: n_sdvs }
    repeat_counts = Counter( Counter( sdvs.varlist ).values() )

    n_sdvs = sum( repeat_counts.values() )

    print( '%i of %i variants (%.2f%%) are splice disrupting in at least one sample.' \
            % ( n_sdvs, n_var, 100*( n_sdvs / n_var )  ) )

    #adds zero counts to create plot with full range of possible values
    for i in range( 1, n_samp + 1 ):
        if i not in repeat_counts:
            repeat_counts[ i ] = 0

    print( '%i of %i variants (%.2f%%) are splice disrupting in all samples.' \
            % ( repeat_counts[ n_samp ], n_var, 100*( repeat_counts[ n_samp ] / n_var )  ) )

    #gives us a list sorted by n_samples in ascending order
    repeat_counts = sorted( repeat_counts.items() )

    labels, counts = zip( *repeat_counts )

    per = [ 100*( count / n_sdvs ) for count in counts ]

    indexes = np.arange( n_samp )

    plt.bar( indexes, per )

    plt.xticks(indexes, labels)

    if y_lim:
        plt.ylim( y_lim )
    else:
        plt.ylim( ( 0, 100 ) )

    plt.ylabel( 'Percent of splice disrupting variants' )
    plt.xlabel( 'Number of samples' )

    plt.title( title )

    plt.show()

def plot_stacked_psi(    var_df,
                          zcols,
                          pos_col,
                          colors,
                          color_col = 'alt',
                          fig_size = ( 20, 5 ),
                          shade_exons = False,
                          zoom=None,
                          tick_spacing = 10,
                          alt_labels = False,
                          bar_labels = False,
                          title = '',
                          y_ax_title='',
                          y_ax_lim = None,
                          x_ax_title = '',
                          legend = True,
                          legend_title = '',
                          legend_loc = 'best',
                          legend_labels = None,
                          tight = True,
                          print_ex_count = False,
                          scale_bar = False,
                          rev_trans = False,
                          hlines = None,
                          savefile = None ):

    tbv = var_df.sort_values( by = [ 'pos', 'alt' ] ).copy()

    if zoom:
        assert zoom[1] > zoom[0], 'Your final zoom coordinate must be larger than the first zoom coordinate'
        tbv_filt = tbv.set_index( 'pos' ).loc[ zoom[0]:zoom[1] ].reset_index()
    else:
        tbv_filt = tbv

    #check if plot will lose bars due to not enough space for all the pixels
    dpi = 100
    if ( tbv_filt.shape[0]*1.5 ) > ( fig_size[0]*dpi ):
        fig_height = fig_size[1]
        fig_size = ( ( tbv_filt.shape[0]*1.5 / dpi ), fig_height )
        print('Adjusting figure width to accomodate all pixels...')

    if print_ex_count:
        print('This figure shows %i exonic bases.' % sum( c[1]-c[0] for c in shade_exons ) )

    col_pivot = tbv_filt.set_index( ['pos', 'alt'] )[ zcols ]

    col_pivot.plot.bar( color = colors,
                        stacked = True,
                        align = 'center',
                        width = 1,
                        figsize = fig_size )

    ax = plt.gca()

    if alt_labels or bar_labels:

        rects = ax.patches

        heights = [ rect.get_height() for rect in rects ]

        n_bars = int( len( heights ) / len( zcols ) )

        pos_ht_sum = [ sum( heights[ i ] for i in range( j, len( heights ), n_bars ) if heights[ i ] > 0 )
                       for j in range( n_bars ) ]
        neg_ht_sum = [ sum( heights[ i ] for i in range( j, len( heights ), n_bars ) if heights[ i ] < 0 )
                       for j in range( n_bars ) ]

    if alt_labels:

        labels = [ idx[1] for idx in col_pivot.index ]

        min_ht = min( neg_ht_sum )

        for rect, label in zip( rects, labels ):
            height = rect.get_height()
            ax.text( rect.get_x() + rect.get_width() / 2,
                     min_ht - 1.5,
                     label,
                     fontsize = 14,
                     fontweight = 'bold',
                     ha='center',
                     va='bottom' )

    if bar_labels:

        for iidx, colmark in enumerate( bar_labels ):

            col, marker = colmark

            #true false vector for where to put the marker
            locs = tbv_filt[ col ]

            for jidx, rloc in enumerate( zip( rects, locs ) ):

                rect, loc = rloc

                if loc:

                    ax.text( rect.get_x() + rect.get_width() /2,
                             pos_ht_sum[ jidx ] + .1 + .9*iidx,
                             marker,
                             fontsize = 10,
                             ha = 'center',
                             va = 'bottom' )

    plt.title( title, fontsize = 24 )

    if not( y_ax_lim ) and ( bar_labels or alt_labels ):

        col_pivot[ 'y_max' ] = col_pivot[ col_pivot > 0 ].sum( axis = 1 )
        col_pivot[ 'y_min' ] = col_pivot[ col_pivot < 0 ].sum( axis = 1 )

        if bar_labels and alt_labels:
            y_ax_lim = ( col_pivot.y_min.min() - 1.6,
                         col_pivot.y_max.max() + .1 + .9*len( bar_labels ) )
        elif bar_labels:
            y_ax_lim = ( col_pivot.y_min.min()*1.01,
                         col_pivot.y_max.max() + .1 + .9*len( bar_labels ) )
        else:
            y_ax_lim = ( col_pivot.y_min.min() - 1.6,
                         col_pivot.y_max.max()*1.01 )

    if y_ax_lim:
        plt.ylim( y_ax_lim )

    plt.ylabel( y_ax_title, fontsize = 22 )
    plt.yticks( fontsize = 18 )

    plt.xlabel( x_ax_title, fontsize = 22 )
    plt.xticks( [ idx for idx,p in enumerate( tbv_filt.index ) if idx%( 3*tick_spacing ) == 1 ],
                [ c for idx,c in enumerate( tbv_filt[ pos_col ] ) if idx%( 3*tick_spacing ) == 1 ],
                fontsize=18,
                rotation='vertical' )

    if hlines:
        for line in hlines:
            plt.axhline( line, c = 'black', ls = '--', alpha = .6 )

    if legend:

        if legend_labels:
            legend = plt.legend( title = legend_title,
                                 ncol = 2,
                                 loc = legend_loc,
                                 labels = legend_labels,
                                 fontsize = 14 )
        else:
            legend = plt.legend( title = legend_title,
                                 ncol = 2,
                                 loc = legend_loc,
                                 fontsize = 14 )
        plt.setp( legend.get_title(), fontsize=14 )
    else:
        ax.legend_ = None
        plt.draw()

    if shade_exons:
        for ex in shade_exons:
            plt.axvspan( col_pivot.index.get_loc( ex[0] ).start - .5,
                         col_pivot.index.get_loc( ex[1] ).stop,
                         facecolor = 'gray',
                         alpha = 0.15 )

    if scale_bar:
        trans = tfrms.blended_transform_factory( ax.transData, ax.transAxes )
        plt.errorbar( 3*tick_spacing, 0.96, xerr = ( 3*tick_spacing / 2 ), color='black', capsize=3, transform=trans)
        txt = str( tick_spacing ) + ' bases' if tick_spacing > 1 else str( tick_spacing ) + ' base'
        plt.text( 3*tick_spacing, 0.94, txt,  horizontalalignment='center',
        verticalalignment='top', transform=trans, fontsize = 14 )

    if tight:
        plt.tight_layout()

    if savefile:
        plt.savefig( savefile )

    plt.show()

def pr_curves( var_df,
               truth_col,
               pred_cols,
               cmap,
               fig_size = ( 5, 5 ),
               grid = False,
               x_ax_label = 'Recall\n(%)',
               y_ax_label = 'Precision\n(%)',
               add_point = False,
               legend = True,
               legend_loc = 'best',
               bbox = False,
               savefile = None,
               **kwargs
             ):

    #pr curve function hates missing values
    tbv = var_df.dropna(subset = [ truth_col ] ).copy()

    if tbv.shape[ 0 ] != var_df.shape[ 0 ]:
        print( 'Missing values in truth column.', str( var_df.shape[ 0 ] - tbv.shape[ 0 ] ), 'rows removed.' )

    plt.figure( figsize = fig_size )

    max_scores = []

    for i, column in enumerate( pred_cols ):

        col_df = tbv.dropna( subset = [ column ] ).copy()

        if tbv.shape[ 0 ] != col_df.shape[ 0 ]:
            print( 'Missing values in', column, 'column.', \
                   str( tbv.shape[ 0 ] - col_df.shape[ 0 ] ), 'rows removed.' )

        precision_raw, recall_raw, _ = precision_recall_curve( col_df[ truth_col ],
                                                                col_df[ column ] )
        auc_raw = auc( recall_raw, precision_raw )

        precision_neg, recall_neg, _ = precision_recall_curve( col_df[ truth_col ],
                                                               -1 * col_df[ column ] )

        auc_neg = auc( recall_neg, precision_neg )

        precision_abs, recall_abs, _ = precision_recall_curve( col_df[ truth_col ],
                                                               np.abs( col_df[ column ] ) )

        auc_abs = auc( recall_abs, precision_abs )

        max_scores.append( max( [ auc_raw, auc_neg, auc_abs ] ) )
        max_fn = [ 'raw', 'neg', 'abs' ][ np.argmax( [ auc_raw, auc_neg, auc_abs ] ) ]

        print( column, 'auc_raw: %0.4f auc_neg: %0.4f auc_abs %0.4f' % ( auc_raw, auc_neg, auc_abs ) )
        print( 'Maximum is %.4f from %s' % ( max_scores[ -1 ], max_fn ) )

        recall = [ recall_raw, recall_neg, recall_abs ][ np.argmax( [ auc_raw, auc_neg, auc_abs ] ) ]
        precision = [ precision_raw, precision_neg, precision_abs ][ np.argmax( [ auc_raw, auc_neg, auc_abs ] ) ]

        plt.plot( 100*recall,
                  100*precision,
                  color = cmap( i ),
                  label = column + ' prAUC = %0.3f' % max_scores[ -1 ],
                  **kwargs )

    if add_point:
        plt.plot( add_point[ 0 ],
                  add_point[ 1 ],
                  color = 'black',
                  marker = add_point[ 2 ],
                  markersize = add_point[ 3 ]
                 )

    plt.xlabel( x_ax_label, fontsize = 24 )
    plt.xticks( fontsize = 20 )
    plt.xlim( ( 0, 100 ) )

    plt.ylabel( y_ax_label, fontsize = 24 )
    plt.yticks( fontsize = 20 )
    plt.ylim( ( 0, 100 ) )

    if legend:

        if not bbox:

            plt.legend( loc = legend_loc )

        else:

            plt.legend( loc = legend_loc,
                        bbox_to_anchor = bbox )

    plt.grid( grid )

    if savefile:
        plt.savefig( savefile,
                     dpi = 300,
                     bbox_inches = 'tight' )

    plt.show()

    return max_scores

def bal_pr_curves( var_df,
                   truth_col,
                   pred_cols,
                   cmap,
                   bootstraps = None,
                   seed = 1124,
                   fig_size = ( 5, 5 ),
                   grid = False,
                   x_ax_label = 'Recall\n(%)',
                   y_ax_label = 'Precision\n(%)',
                   add_point = False,
                   legend = True,
                   legend_loc = 'best',
                   bbox = False,
                   savefile = None,
                   **kwargs
                 ):

    #pr curve function hates missing values
    tbv = var_df.dropna(subset = [ truth_col ] ).copy()

    if tbv.shape[ 0 ] != var_df.shape[ 0 ]:
        print( 'Missing values in truth column.', str( var_df.shape[ 0 ] - tbv.shape[ 0 ] ), 'rows removed.' )

    plt.figure( figsize = fig_size )

    auc_vals = []

    for i, column in enumerate( pred_cols ):

        col_df = tbv.dropna( subset = [ column ] ).copy()

        if tbv.shape[ 0 ] != col_df.shape[ 0 ]:
            print( 'Missing values in', column, 'column.', \
                   str( tbv.shape[ 0 ] - col_df.shape[ 0 ] ), 'rows removed.' )

        if len( col_df ) == 0:

            print( 'Skipping %s since all values are missing' % column )
            continue

        prior = col_df[ truth_col ].sum() / len( col_df )

        precision, recall, _ = precision_recall_curve( col_df[ truth_col ],
                                                           col_df[ column ] )

        precision = precision*( 1 - prior ) / ( precision*( 1- prior ) + ( 1 - precision )*prior )

        if bootstraps:

            random.seed( seed )

            idx_l = [ i for i in range( len( col_df ) ) ]

            boot_dfs = [ col_df.iloc[ random.choices( idx_l, k = len( col_df ) ) ].copy() for i in range( bootstraps ) ]

            pr_vals = [ precision_recall_curve( boot_dfs[ i ][ truth_col ],
                                                boot_dfs[ i ][ column ] )
                        for i in range( bootstraps ) ]

            pr_vals = [ ( p*( 1- prior ) / ( p*( 1 - prior ) + ( 1 - p )*prior ), r, _ ) for p,r,_ in pr_vals ]

            auc_vals.append( np.array( [ auc( pr[ 1 ], pr[ 0 ] ) for pr in pr_vals ] ) )

            auc_mean = np.nanmean( auc_vals[ -1 ] )
            auc_std = np.nanstd( auc_vals[ -1 ] )

            #sometimes with smaller datasets we get missing values
            nonmissing = np.count_nonzero( ~np.isnan( auc_d[ 'pr_auc' ][ -1 ] ) )

            auc_ci = ( auc_mean - 1.96*auc_std / np.sqrt( nonmissing ), auc_mean + 1.96*auc_std / np.sqrt( nonmissing ) )

            leg_label = column.split( '_' )[ 0 ] + ' prAUC = %0.3f; 95%% CI = ( %0.3f, %0.3f )' % ( auc_mean, auc_ci[ 0 ], auc_ci[ 1 ] )

        else:

            auc_vals.append( auc( recall, precision ) )

            leg_label = column.split( '_' )[ 0 ] + ' prAUC = %0.3f' % auc_d[ 'pr_auc' ][ -1 ]

        plt.plot( 100*recall,
                  100*precision,
                  color = cmap( i ),
                  label = leg_label,
                  **kwargs )

    if add_point:
        plt.plot( add_point[ 0 ],
                  add_point[ 1 ],
                  color = 'black',
                  marker = add_point[ 2 ],
                  markersize = add_point[ 3 ]
                 )

    plt.xlabel( x_ax_label, fontsize = 24 )
    plt.xticks( fontsize = 20 )
    plt.xlim( ( 0, 100 ) )

    plt.ylabel( y_ax_label, fontsize = 24 )
    plt.yticks( fontsize = 20 )
    plt.ylim( ( 0, 100 ) )

    if legend:

        if not bbox:

            plt.legend( loc = legend_loc )

        else:

            plt.legend( loc = legend_loc,
                        bbox_to_anchor = bbox )

    plt.grid( grid )

    if savefile:
        plt.savefig( savefile,
                     dpi = 300,
                     bbox_inches = 'tight' )

    plt.show()

    return auc_vals

def bal_pr_curves_subplot( datasets,
                           categories,
                           cat_col,
                           truth_col,
                           pred_cols,
                           cmap,
                           bootstraps = None,
                           seed = 1124,
                           figsize = ( 15, 30 ),
                           sharex = True,
                           sharey = True,
                           grid = False,
                           bbox = None,
                           add_point = False,
                           savefile = None,
                           **kwargs
                         ):

    data = { name: datasets[ name ].loc[ datasets[ name ][ truth_col ].notnull() ].copy() for name in datasets.keys() }

    for name in data.keys():

        if len( data[ name ] ) != len( datasets[ name ] ):

            missing = len( datasets[ name ] ) - len( data[ name ] )

            print( '%i missing values in %s in dataset %s' % ( missing, truth_col, name ) )

    fig, ax = plt.subplots( len( data.keys() ),
                            len( categories ),
                            figsize = figsize,
                            sharex = sharex,
                            sharey = sharey )

    auc_d = { 'data': [],
              'group': [],
               'pr_auc': [] }

    for i, name in enumerate( data.keys() ):

        ax[ i ][ 0 ].set_ylabel( '%s Precision\n(%%)' % name,
                                 fontsize = 20 )

        df = data[ name ]

        for j, cat in enumerate( categories ):

            auc_d[ 'data' ].append( name )
            auc_d[ 'group' ].append( cat )

            if cat != 'Transcriptome':

                cat_df = df.loc[ df[ cat_col ] == cat ].copy()

            else:

                cat_df = df.copy()

            if len( cat_df ) == 0 or len( cat_df[ truth_col ].unique() ) == 1:

                print( 'Unable to plot category %s for dataset %s' % ( cat, name ) )
                ax[ i ][ j ].set_visible( False )

                for k in range( len( pred_cols ) ):
                    auc_d[ 'pr_auc' ].append( np.nan )
                continue

            if cat_df[ truth_col ].sum() < 2 or ( ~( cat_df[ truth_col ] ) ).sum() < 2:

                print( 'Unable to plot category %s for dataset %s' % ( cat, name ) )
                ax[ i ][ j ].set_visible( False )
                for k in range( len( pred_cols ) ):
                    auc_d[ 'pr_auc' ].append( np.nan )
                continue

            for k, col in enumerate( pred_cols ):

                col_df = cat_df.dropna( subset = [ col ] ).copy()

                if len( cat_df ) != len( col_df ):

                    missing = len( cat_df ) - len( col_df )
                    print( '%i missing values in %s column for dataset %s in category %s.' \
                           % ( missing, col, name, cat ) )

                if len( col_df ) == 0:

                    print( 'No non-missing values in %s column for dataset %s in category %s - skipping...' \
                           % ( col, name, cat ) )
                    auc_d[ 'pr_auc' ].append( np.nan )
                    continue

                if col_df[ truth_col ].sum() < 2 or ( ~( col_df[ truth_col ] ) ).sum() < 2:

                    print( 'Only one event in %s column for dataset %s in category %s - skipping...' \
                           % ( col, name, cat ) )
                    auc_d[ 'pr_auc' ].append( np.nan )
                    continue

                prior = col_df[ truth_col ].sum() / len( col_df )

                precision, recall, _ = precision_recall_curve( col_df[ truth_col ],
                                                               col_df[ col ] )

                precision = precision*( 1 - prior ) / ( precision*( 1- prior ) + ( 1 - precision )*prior )

                if bootstraps:

                    random.seed( seed )

                    idx_l = [ i for i in range( len( col_df ) ) ]

                    boot_dfs = [ col_df.iloc[ random.choices( idx_l, k = len( col_df ) ) ].copy() for i in range( bootstraps ) ]

                    pr_vals = [ precision_recall_curve( boot_dfs[ i ][ truth_col ],
                                                        boot_dfs[ i ][ col ] )
                                for i in range( bootstraps ) ]

                    pr_vals = [ ( p*( 1- prior ) / ( p*( 1 - prior ) + ( 1 - p )*prior ), r, _ ) for p,r,_ in pr_vals ]

                    auc_d[ 'pr_auc' ].append( np.array( [ auc( pr[ 1 ], pr[ 0 ] ) for pr in pr_vals ] ) )

                    auc_mean = np.nanmean( auc_vals[ -1 ] )
                    auc_std = np.nanstd( auc_vals[ -1 ] )

                    #sometimes with smaller datasets we get missing values
                    nonmissing = np.count_nonzero( ~np.isnan( auc_vals[ -1 ] ) )

                    auc_ci = ( auc_mean - 1.96*auc_std / np.sqrt( nonmissing ), auc_mean + 1.96*auc_std / np.sqrt( nonmissing ) )

                    leg_label = col.split( '_' )[ 0 ] + ' prAUC = %0.3f\n95%% CI = ( %0.3f, %0.3f )' % ( auc_mean, auc_ci[ 0 ], auc_ci[ 1 ] )

                else:

                    auc_d[ 'pr_auc' ].append( auc( recall, precision ) )

                    if col.split( '_' )[ 0 ] != 'DS':

                        leg_label = col.split( '_' )[ 0 ] + ' prAUC = %0.3f' % auc_vals[ -1 ]

                    else:

                        leg_label = 'SpliceAI prAUC = %0.3f' % auc_vals[ -1 ]

                ax[ i ][ j ].plot( 100*recall,
                                   100*precision,
                                   color = cmap( k ),
                                   label = leg_label,
                                   **kwargs )

            ax[ i ][ j ].set_xlabel( 'Recall\n(%)', fontsize = 12 )
            ax[ 0 ][ j ].set_title( cat, fontsize = 24 )

            ax[ i ][ j ].set_ylim( ( 0, 100 ) )
            ax[ i ][ j ].set_xlim( ( 0, 100 ) )

            if bbox:

                ax[ i ][ j ].legend( loc = 'center left',
                                     bbox_to_anchor = bbox )

            else:

                ax[ i ][ j ].legend()

    plt.tight_layout()

    if savefile:
        plt.savefig( savefile,
                     dpi = 300,
                     #bbox_inches = 'tight'
                   )

    plt.show()

    auc_vals = pd.DataFrame( auc_d )

    return auc_vals

def roc_curves( var_df,
               truth_col,
               pred_cols,
               cmap,
               fig_size = ( 5, 5 ),
               grid = False,
               x_ax_label = 'False Positive Rate\n(%)',
               y_ax_label = 'True Positive Rate\n(%)',
               add_point = False,
               legend = True,
               legend_loc = 'best',
               bbox = False,
               savefile = None,
               **kwargs
             ):

    #pr curve function hates missing values
    tbv = var_df.dropna(subset = [ truth_col ] ).copy()

    if tbv.shape[ 0 ] != var_df.shape[ 0 ]:
        print( 'Missing values in truth column.', str( var_df.shape[ 0 ] - tbv.shape[ 0 ] ), 'rows removed.' )

    plt.figure( figsize = fig_size )

    max_scores = []

    for i, column in enumerate( pred_cols ):

        col_df = tbv.dropna( subset = [ column ] ).copy()

        if tbv.shape[ 0 ] != col_df.shape[ 0 ]:
            print( 'Missing values in', column, 'column.', \
                   str( tbv.shape[ 0 ] - col_df.shape[ 0 ] ), 'rows removed.' )

        fpr_raw, tpr_raw, thresh_raw = roc_curve( col_df[ truth_col ],
                                         col_df[ column ] )
        auc_raw = auc( fpr_raw, tpr_raw )

        fpr_neg, tpr_neg, thresh_neg = roc_curve( col_df[ truth_col ],
                                         -1 * col_df[ column ] )

        auc_neg = auc( fpr_neg, tpr_neg )

        fpr_abs, tpr_abs, thresh_abs = roc_curve( col_df[ truth_col ],
                                         np.abs( col_df[ column ] ) )

        auc_abs = auc( fpr_abs, tpr_abs )

        max_scores.append( max( [ auc_raw, auc_neg, auc_abs ] ) )
        max_fn = [ 'raw', 'neg', 'abs' ][ np.argmax( [ auc_raw, auc_neg, auc_abs ] ) ]

        print( column, 'auc_raw: %0.4f auc_neg: %0.4f auc_abs %0.4f' % ( auc_raw, auc_neg, auc_abs ) )
        print( 'Maximum is %.4f from %s' % ( max_scores[ -1 ], max_fn ) )

        fpr = [ fpr_raw, fpr_neg, fpr_abs ][ np.argmax( [ auc_raw, auc_neg, auc_abs ] ) ]
        tpr = [ tpr_raw, tpr_neg, tpr_abs ][ np.argmax( [ auc_raw, auc_neg, auc_abs ] ) ]
        thresh = [ thresh_raw, thresh_neg, thresh_abs ][ np.argmax( [ auc_raw, auc_neg, auc_abs ] ) ]

        youden_j = thresh[ np.argmax( tpr - fpr ) ]

        plt.plot( 100*fpr,
                  100*tpr,
                  color = cmap( i ),
                  label = column + " rocAUC = %0.3f; Youden's J: %0.3f" % ( max_scores[ -1 ], youden_j ),
                  **kwargs )

    plt.plot( np.arange( 0, 100 ),
              np.arange( 0, 100 ),
              color = 'black',
              ls = 'dashed' )

    if add_point:
        plt.plot( add_point[ 0 ],
                  add_point[ 1 ],
                  color = 'black',
                  marker = add_point[ 2 ],
                  markersize = add_point[ 3 ]
                 )

    plt.xlabel( x_ax_label, fontsize = 24 )
    plt.xticks( fontsize = 20 )
    plt.xlim( ( 0, 100 ) )

    plt.ylabel( y_ax_label, fontsize = 24 )
    plt.yticks( fontsize = 20 )
    plt.ylim( ( 0, 100 ) )

    if legend:

        if not bbox:

            plt.legend( loc = legend_loc )

        else:

            plt.legend( loc = legend_loc,
                        bbox_to_anchor = bbox )

    plt.grid( grid )

    if savefile:
        plt.savefig( savefile,
                     dpi = 300,
                     bbox_inches = 'tight' )

    plt.show()

    return max_scores

def roc_curve_subplot( datasets,
                       categories,
                       truth_col,
                       pred_cols,
                       cmap,
                       figsize = ( 40, 30 ),
                       sharex = True,
                       sharey = True,
                       savefile = None,
                       **kwargs ):

    data = { name: datasets[ name ].loc[ datasets[ name ][ truth_col ].notnull() ].copy() for name in datasets.keys() }

    for name in data.keys():

        if len( data[ name ] ) != len( datasets[ name ] ):

            missing = len( datasets[ name ] ) - len( data[ name ] )

            print( '%i missing values in %s in dataset %s' % ( missing, truth_col, name ) )

    fig, ax = plt.subplots( len( data.keys() ),
                            len( categories ),
                            figsize = figsize,
                            sharex = sharex,
                            sharey = sharey )

    youden_j = np.empty( ( len( data.keys() )*len( pred_cols ), len( categories ) ) )
    youden_j[:] = np.nan

    for i, name in enumerate( data.keys() ):

        ax[ i ][ 0 ].set_ylabel( '%s True Positive Rate\n(%%)' % name,
                                 fontsize = 20 )

        df = data[ name ]

        for j, cat in enumerate( categories ):

            if cat != 'PROP':

                cat_df = df.loc[ df.variant_cat == cat ].copy()

            else:

                cat_df = df.copy()

            if len( cat_df ) == 0 or len( cat_df[ truth_col ].unique() ) == 1:

                print( 'Unable to plot category %s for dataset %s' % ( cat, name ) )
                ax[ i ][ j ].set_visible( False )
                continue

            if cat_df[ truth_col ].sum() < 2 or ( ~( cat_df[ truth_col ] ) ).sum() < 2:

                print( 'Unable to plot category %s for dataset %s' % ( cat, name ) )
                ax[ i ][ j ].set_visible( False )
                continue

            for k, col in enumerate( pred_cols ):

                col_df = cat_df.dropna( subset = [ col ] ).copy()

                if len( cat_df ) != len( col_df ):

                    missing = len( cat_df ) - len( col_df )
                    print( '%i missing values in %s column for dataset %s in category %s.' \
                           % ( missing, col, name, cat ) )

                if len( col_df ) == 0:

                    print( 'No non-missing values in %s column for dataset %s in category %s - skipping...' \
                           % ( col, name, cat ) )
                    continue

                if col_df[ truth_col ].sum() < 2 or ( ~( col_df[ truth_col ] ) ).sum() < 2:

                    print( 'Only one event in %s column for dataset %s in category %s - skipping...' \
                           % ( col, name, cat ) )
                    continue

                fpr, tpr, thresh = roc_curve( col_df[ truth_col ], col_df[ col ] )
                auc_val = auc( fpr, tpr )

                youden_j[ len( pred_cols )*i + k ][ j ] = thresh[ np.argmax( tpr - fpr ) ]

                ax[ i ][ j ].plot( 100*fpr,
                                   100*tpr,
                                   color = cmap( k ),
                                   label = col.split( '_' )[ 0 ] + " AUC = %0.3f" % ( auc_val ),
                                   #label = col.split( '_' )[ 0 ] + " AUC = %0.3f\nYouden's J: %0.3f" % ( auc_val, youden_j[ len( pred_cols )*i + k ][ j ] ),
                                   **kwargs )

            ax[ i ][ j ].plot( np.arange( 0, 100 ),
                               np.arange( 0, 100 ),
                               color = 'black',
                                ls = 'dashed' )

            ax[ i ][ j ].set_xlabel( 'False Positive Rate\n(%)', fontsize = 12 )
            ax[ 0 ][ j ].set_title( cat, fontsize = 24 )

            ax[ i ][ j ].set_ylim( ( 0, 100 ) )
            ax[ i ][ j ].set_xlim( ( 0, 100 ) )

            ax[ i ][ j ].legend()

    if savefile:
        plt.savefig( savefile,
                     dpi = 300,
                     #bbox_inches = 'tight'
                   )

    plt.show()

    youden_df = pd.DataFrame( youden_j,
                              columns = categories,
                              index = [ t + ' ' + d for d in data.keys() for t in pred_cols ]
                            )

    youden_df[ 'tool' ] = [ t.split( '_' )[ 0 ] for i in range( len( data.keys() ) ) for t in pred_cols ]
    youden_df[ 'data' ] = [ d for d in data.keys() for i in range( len( pred_cols ) ) ]

    return youden_df

def youden_subplot( youdens_df,
                    tool_col,
                    data_col,
                    cat_cols,
                    stat = 'mean',
                    error_bars = False,
                    figsize = ( 12, 11 ),
                    sharex = False,
                    sharey = False,
                    savefile = None,
                    **kwargs ):

    youden = youdens_df.copy()

    sub_size = int( np.ceil( np.sqrt( len( youden[ tool_col ].unique() ) ) ) )

    fig, ax = plt.subplots( sub_size,
                            sub_size,
                            figsize = figsize,
                            sharex = sharex,
                            sharey = sharey )

    for t, tool in enumerate( youden[ tool_col ].unique() ):

        tool_df = youden.loc[ ( youden[ tool_col ] == tool ) & ( youden[ data_col ] != 'standard' ) ].copy()

        tool_df = tool_df.append( pd.Series( tool_df[ cat_cols ].mean( skipna = True,  ),
                                             name = 'mean' ) )

        tool_df =  tool_df.append( pd.Series( tool_df[ cat_cols ].median( skipna = True,  ),
                                               name = 'median' ) )

        tool_df = tool_df.append( pd.Series( tool_df[ cat_cols ].std( skipna = True,  ),
                                               name = 'std' ) )

        tool_df = tool_df.append( pd.Series( tool_df[ cat_cols ].isnull().sum(),
                                               name = 'n' ) )

        #print( np.sqrt( tool_df.loc[ 'n', cat_cols ] ) )

        tool_df = tool_df.append( pd.Series( tool_df.loc[ 'std', cat_cols ] / np.sqrt( len( tool_df ) ),
                                             name = 'error' ) )

        i,j = ( t // sub_size, t % sub_size )

        if not error_bars:

            ax[ i ][ j ].scatter( cat_cols,
                                  tool_df.loc[ stat, cat_cols ],
                                  label = 'Optimal measured threshold',
                                  **kwargs )

            standard_df = youden.loc[ ( youden[ tool_col ] == tool ) & ( youden[ data_col ] == 'standard' ) ].copy()

            ax[ i ][ j ].scatter( cat_cols,
                                    standard_df[ cat_cols ],
                                    label = 'Recommended threshold' )

        else:

            ax[ i ][ j ].errorbar( cat_cols,
                                  tool_df.loc[ stat, cat_cols ],
                                  yerr = tool_df.loc[ 'error', cat_cols ],
                                  fmt = 'o',
                                  label = 'Optimal measured threshold',
                                  **kwargs )

            standard_df = youden.loc[ ( youden[ tool_col ] == tool ) & ( youden[ data_col ] == 'standard' ) ].copy()

            ax[ i ][ j ].errorbar( cat_cols,
                                    standard_df[ cat_cols ].T,
                                    fmt = 'o',
                                    label = 'Recommended threshold' )

        ax[ i ][ j ].set_ylabel( "%s Youden's J Treshold %s" % ( tool, stat ) )

        ax[ i ][ j ].set_xlabel( 'Variant Category' )
        ax[ i ][ j ].set_xticklabels( [ c if c != 'Transcriptome' else 'All variants' for c in cat_cols ], rotation = 90 )

        if tool_df.loc[ stat, cat_cols ].min() < 0 and tool_df.loc[ stat, cat_cols ].max() > 0:

            continue

        elif tool_df.loc[ stat, cat_cols ].min() < 0:

            ax[ i ][ j ].set_ylim( None, 0 )

        elif tool_df.loc[ stat, cat_cols ].max() > 0:

            ax[ i ][ j ].set_ylim( 0, None )

    ax[ 1 ][ 2 ].legend( loc = 'center left',
                         bbox_to_anchor = ( 1.1, .5 ) )

    if j < sub_size:

        for k in range( j + 1, sub_size ):

            ax[ i ][ k ].set_visible( False )

    plt.tight_layout()

    if savefile:
        plt.savefig( savefile,
                     dpi = 300,
                     #bbox_inches = 'tight'
                   )

    plt.show()

def f1_scores( var_df,
               truth_col,
               pred_cols,
               n_thresh = 100
             ):

    tbv = var_df.dropna(subset = [ truth_col ] ).copy()

    if tbv.shape[ 0 ] != var_df.shape[ 0 ]:
        print( 'Missing values in truth column.', str( var_df.shape[ 0 ] - tbv.shape[ 0 ] ), 'rows removed.' )

    max_scores = []
    prop_scores = []

    tot_events = tbv[ truth_col ].sum()

    for i, column in enumerate( pred_cols ):

        col_df = tbv.dropna( subset = [ column ] ).copy()

        if tbv.shape[ 0 ] != col_df.shape[ 0 ]:
            print( 'Missing values in', column, 'column.', \
                   str( tbv.shape[ 0 ] - col_df.shape[ 0 ] ), 'rows removed.' )

        col_sort = np.sort( np.unique( col_df[ column ] ), kind = 'heapsort' )

        thresh_raw = ( col_sort[ 1: ] + col_sort[ : -1 ] ) / 2

        try:

            prop_thresh_raw = np.min( [ idx for idx,t in enumerate( thresh_raw )
                                        if ( col_df[ column ] >= t ).sum() <= tot_events ] )

        except:

            prop_thresh_raw = len( thresh_raw ) - 1

        f1_raw = [ f1_score( col_df[ truth_col ], col_df[ column ] >= t )
                   for t in thresh_raw ]

        f1_raw_max = np.max( f1_raw )

        max_thresh_raw = thresh_raw[ np.argmax( f1_raw ) ]

        thresh_neg = -thresh_raw

        try:

            prop_thresh_neg = np.min( [ idx for idx,t in enumerate( thresh_neg )
                                        if ( -1*col_df[ column ] >= t ).sum() <= tot_events ] )

        except:

            prop_thresh_neg = len( thresh_neg ) - 1

        f1_neg = [ f1_score( col_df[ truth_col ], -1*col_df[ column ] >= t )
                   for t in thresh_neg ]

        f1_neg_max = np.max( f1_neg )

        max_thresh_neg = thresh_neg[ np.argmax( f1_neg ) ]

        col_sort_abs = np.sort( np.unique( np.abs( col_df[ column ] ) ), kind = 'heapsort' )

        thresh_abs = ( col_sort_abs[ 1: ] + col_sort_abs[ : -1 ] ) / 2

        try:

            prop_thresh_abs = np.min( [ idx for idx,t in enumerate( thresh_abs )
                                        if ( np.abs( col_df[ column ] ) >= t ).sum() <= tot_events ] )

        except:

            prop_thresh_abs = len( thresh_abs ) - 1

        f1_abs = [ f1_score( col_df[ truth_col ], np.abs( col_df[ column ] ) >= t )
                   for t in thresh_abs ]

        f1_abs_max = np.max( f1_abs )

        max_thresh_abs = thresh_abs[ np.argmax( f1_abs ) ]

        max_scores.append( max( [ f1_raw_max, f1_neg_max, f1_abs_max ] ) )
        max_fn = [ 'raw', 'neg', 'abs' ][ np.argmax( [ f1_raw_max, f1_neg_max, f1_abs_max ] ) ]
        max_thresh = [ max_thresh_raw, max_thresh_neg, max_thresh_abs ][ np.argmax( [ f1_raw_max, f1_neg_max, f1_abs_max ] ) ]

        prop_scores.append( max( [ f1_raw[ prop_thresh_raw ], f1_neg[ prop_thresh_neg ], f1_abs[ prop_thresh_abs ] ] ) )
        prop_fn = [ 'raw', 'neg', 'abs' ][ np.argmax( [ f1_raw[ prop_thresh_raw ], f1_neg[ prop_thresh_neg ], f1_abs[ prop_thresh_abs ] ] ) ]
        prop_thresh = [ thresh_raw[ prop_thresh_raw ], thresh_neg[ prop_thresh_neg ], thresh_abs[ prop_thresh_abs ] ][ np.argmax( [ f1_raw[ prop_thresh_raw ], f1_neg[ prop_thresh_neg ], f1_abs[ prop_thresh_abs ] ] ) ]

        print( column, 'f1_raw_max: %1.3f f1_neg_max: %1.3f f1_abs_max: %1.3f' % ( f1_raw_max, f1_neg_max, f1_abs_max ) )
        print( 'Maximum is %1.3f from %s at a threshold of %1.3f' % ( max_scores[ -1 ], max_fn, max_thresh ) )

        print( column, 'f1_raw_prop: %1.3f f1_neg_prop: %1.3f f1_abs_prop: %1.3f' % ( f1_raw[ prop_thresh_raw ], f1_neg[ prop_thresh_neg ], f1_abs[ prop_thresh_abs ] ) )
        print( 'At a threshold of %1.3f, %s maximizes with f1 = %1.3f with the same proportion as the truth set\n' % ( prop_thresh, prop_fn, prop_scores[ -1 ] ) )

    return max_scores, prop_scores

def mcc_scores( var_df,
               truth_col,
               pred_cols,
               n_thresh = 100
             ):

    tbv = var_df.dropna(subset = [ truth_col ] ).copy()

    if tbv.shape[ 0 ] != var_df.shape[ 0 ]:
        print( 'Missing values in truth column.', str( var_df.shape[ 0 ] - tbv.shape[ 0 ] ), 'rows removed.' )

    max_scores = []
    prop_scores = []

    tot_events = tbv[ truth_col ].sum()

    for i, column in enumerate( pred_cols ):

        col_df = tbv.dropna( subset = [ column ] ).copy()

        if tbv.shape[ 0 ] != col_df.shape[ 0 ]:
            print( 'Missing values in', column, 'column.', \
                   str( tbv.shape[ 0 ] - col_df.shape[ 0 ] ), 'rows removed.' )

        col_sort = np.sort( np.unique( col_df[ column ] ), kind = 'heapsort' )

        thresh_raw = ( col_sort[ 1: ] + col_sort[ : -1 ] ) / 2

        try:

            prop_thresh_raw = np.min( [ idx for idx,t in enumerate( thresh_raw )
                                        if ( col_df[ column ] >= t ).sum() <= tot_events ] )

        except:

            prop_thresh_raw = len( thresh_raw ) - 1

        mcc_raw = [ matthews_corrcoef( col_df[ truth_col ], col_df[ column ] >= t )
                   for t in thresh_raw ]

        mcc_raw_max = np.max( mcc_raw )

        max_thresh_raw = thresh_raw[ np.argmax( mcc_raw ) ]

        thresh_neg = -thresh_raw

        try:

            prop_thresh_neg = np.min( [ idx for idx,t in enumerate( thresh_neg )
                                        if ( -1*col_df[ column ] >= t ).sum() <= tot_events ] )

        except:

            prop_thresh_neg = len( thresh_neg ) - 1

        mcc_neg = [ matthews_corrcoef( col_df[ truth_col ], -1*col_df[ column ] >= t )
                   for t in thresh_neg ]

        mcc_neg_max = np.max( mcc_neg )

        max_thresh_neg = thresh_neg[ np.argmax( mcc_neg ) ]

        col_sort_abs = np.sort( np.unique( np.abs( col_df[ column ] ) ), kind = 'heapsort' )

        thresh_abs = ( col_sort_abs[ 1: ] + col_sort_abs[ : -1 ] ) / 2

        try:

            prop_thresh_abs = np.min( [ idx for idx,t in enumerate( thresh_abs )
                                        if ( np.abs( col_df[ column ] ) >= t ).sum() <= tot_events ] )

        except:

            prop_thresh_abs = len( thresh_abs ) - 1

        mcc_abs = [ matthews_corrcoef( col_df[ truth_col ], np.abs( col_df[ column ] ) >= t )
                   for t in thresh_abs ]

        mcc_abs_max = np.max( mcc_abs )

        max_thresh_abs = thresh_abs[ np.argmax( mcc_abs ) ]

        #max_idx = np.argmax( np.abs( [ mcc_raw_max, mcc_neg_max, mcc_abs_max ] ) )
        max_scores.append( max( [ mcc_raw_max, mcc_neg_max, mcc_abs_max ] ) )
        max_fn = [ 'raw', 'neg', 'abs' ][ np.argmax( [ mcc_raw_max, mcc_neg_max, mcc_abs_max ] ) ]
        max_thresh = [ max_thresh_raw, max_thresh_neg, max_thresh_abs ][ np.argmax( [ mcc_raw_max, mcc_neg_max, mcc_abs_max ] ) ]

        prop_scores.append( max( [ mcc_raw[ prop_thresh_raw ], mcc_neg[ prop_thresh_neg ], mcc_abs[ prop_thresh_abs ] ] ) )
        prop_fn = [ 'raw', 'neg', 'abs' ][ np.argmax( [ mcc_raw[ prop_thresh_raw ], mcc_neg[ prop_thresh_neg ], mcc_abs[ prop_thresh_abs ] ] ) ]
        prop_thresh = [ thresh_raw[ prop_thresh_raw ], thresh_neg[ prop_thresh_neg ], thresh_abs[ prop_thresh_abs ] ][ np.argmax( [ mcc_raw[ prop_thresh_raw ], mcc_neg[ prop_thresh_neg ], mcc_abs[ prop_thresh_abs ] ] ) ]

        print( column, 'mcc_raw_max: %1.3f mcc_neg_max: %1.3f mcc_abs_max: %1.3f' % ( mcc_raw_max, mcc_neg_max, mcc_abs_max ) )
        print( 'Maximum is %1.3f from %s at a threshold of %1.3f' % ( max_scores[ -1 ], max_fn, max_thresh ) )

        print( column, 'mcc_raw_prop: %1.3f mcc_neg_prop: %1.3f mcc_abs_prop: %1.3f' % ( mcc_raw[ prop_thresh_raw ], mcc_neg[ prop_thresh_neg ], mcc_abs[ prop_thresh_abs ] ) )
        print( 'At a threshold of %1.3f, %s maximizes with mcc = %1.3f with the same proportion as the truth set\n' % ( prop_thresh, prop_fn, prop_scores[ -1 ] ) )

    return max_scores, prop_scores

def spearman_rank( var_df,
                   measured_col,
                   pred_cols,
                   nan_policy = 'omit'
                 ):

    tbv = var_df.copy()

    max_scores = []

    for column in pred_cols:

        spear_raw = abs( ss.spearmanr( tbv[ [ measured_col, column ] ],
                                       nan_policy = nan_policy )[ 0 ] )

        spear_abs = abs( ss.spearmanr( tbv[ measured_col ],
                                       tbv[ column ].abs(),
                                       nan_policy = nan_policy )[ 0 ] )

        max_scores.append( max( [ spear_raw, spear_abs ] ) )
        max_fn = [ 'raw', 'abs' ][ np.argmax( [ spear_raw, spear_abs ] ) ]

        print( column, 'spearman_raw: %0.3f spearmans_abs: %0.3f' % ( spear_raw, spear_abs ) )
        print( 'Maximum is %0.3f from %s' % ( max_scores[ -1 ], max_fn ) )

    return max_scores

def pearson( var_df,
             measured_col,
             pred_cols, ):

    tbv = var_df.dropna( subset = [ measured_col ] ).copy()

    if tbv.shape[ 0 ] != var_df.shape[ 0 ]:
        print( 'Missing values in measured column.', str( var_df.shape[ 0 ] - tbv.shape[ 0 ] ), 'rows removed.' )

    max_scores = []

    for column in pred_cols:

        col_df = tbv.dropna( subset = [ column ] ).copy()

        if tbv.shape[ 0 ] != col_df.shape[ 0 ]:
            print( 'Missing values in', column, 'column.', \
                   str( tbv.shape[ 0 ] - col_df.shape[ 0 ] ), 'rows removed.' )

        pear_raw = abs( ss.pearsonr( col_df[ measured_col ],
                                      col_df[ column ] )[ 0 ] )

        pear_abs = abs( ss.pearsonr( col_df[ measured_col ],
                                      col_df[ column ].abs(), )[ 0 ] )

        max_scores.append( max( [ pear_raw, pear_abs ] ) )
        max_fn = [ 'raw', 'abs' ][ np.argmax( [ pear_raw, pear_abs ] ) ]

        print( column, 'pearson_raw: %0.3f pearson_abs: %0.3f' % ( pear_raw, pear_abs ) )
        print( 'Maximum is %0.3f from %s' % ( max_scores[ -1 ], max_fn ) )

    return max_scores

def plot_RBP_chg( var_df,
              y_col,
              x_col,
              arrow_col,
              colors,
              color_col = 'alt',
              arrow_filt = None,
              ax = None,
              title = '',
              y_ax_lim = None,
              x_ax_lim = None,
              y_ax_title='',
              x_ax_title = '',
              marker_col = None,
              marker = 'o',
              **kwargs ):

    tbv = var_df.copy()

    if ax:
        ax = ax
    else:
        ax = plt.gca()

    if not marker_col:

        ax.scatter( tbv[ x_col ],
                     tbv[ y_col ],
                     color = tbv[ color_col ],
                     **kwargs
                   )

    else:

        for m, v in marker.items():

            tbv_filt = tbv.loc[ tbv[ marker_col ] == v[ 0 ] ]

            ax.scatter( tbv_filt[ x_col ],
                     tbv_filt[ y_col ],
                     color = tbv_filt[ color_col ],
                    marker = m,
                    s = v[ 1 ],
                     **kwargs
                   )

    arrow_x_start = tbv[ x_col ].values
    arrow_y_start = tbv[ y_col ].values
    arrow_len = tbv[ arrow_col ].values

    #add arrows to plot
    for i, alt in enumerate( tbv.alt ):

        if arrow_filt and arrow_x_start[ i ] < arrow_filt:
            continue

        ax.arrow( arrow_x_start[ i ],
                  arrow_y_start[ i ],
                  0,       #change in x
                  arrow_len[ i ],                      #change in y
                  head_width = 0.3*( np.abs( arrow_len[ i ] ) / 2 ),         #arrow head width
                  head_length = 0.1*( np.abs( arrow_len[ i ] ) / 2 ),        #arrow head length
                  width = 0.03,              #arrow stem width
                  fc = colors[ alt ],             #arrow fill color
                  ec = colors[ alt ]
                )             #arrow edge color

    plt.title( title, fontsize = 24 )

    ax.set_xlabel( x_ax_title, fontsize = 18 )
    ax.tick_params( axis = 'x', which='major', labelsize = 12 )

    ax.set_ylabel( y_ax_title, fontsize = 18 )
    ax.tick_params( axis = 'y', which='major', labelsize = 12 )

    if x_ax_lim:
        ax.set_xlim( x_ax_lim )

    if y_ax_lim:
        ax.set_ylim( y_ax_lim )

    plt.show()

def subplot_psi_wt(   var_df,
                          y_col,
                          pos_col,
                          colors,
                          color_col = 'ref',
                          edge_col = None,
                          ax = None,
                           zoom = None,
                          shade_exons = False,
                          tick_spacing = 10,
                          title = '',
                          y_ax_lim = None,
                          y_ax_title='',
                          x_ax_title = '',
                          legend = True,
                          legend_title = '',
                          legend_loc = 'best',
                          legend_labels = None,
                          tight = True,
                          print_ex_count = False,
                          scale_bar = False,
                          scale_bar_loc = 'left',
                          rev_trans = False,
                          hlines = None,
                          snap = False):

    tbv = var_df.copy()

    if print_ex_count:
        print('This figure shows %i exonic bases.' % sum( c[1]-c[0] for c in shade_exons ) )

    if ax:
        ax = ax
    else:
        ax = plt.gca()

    d2c = dict( zip( [ 'A', 'C', 'G', 'T' ], colors) )

    if zoom:
        assert zoom[1] > zoom[0], 'Your final zoom coordinate must be larger than the first zoom coordinate'
        tbv = tbv.set_index( 'pos' ).loc[ zoom[0]:zoom[1] ].reset_index()

    tbv.plot.bar( x = 'pos',
                  y = y_col,
                        ax = ax,
                        color = map( d2c.get, tbv[ color_col ] ),
                        edgecolor = edge_col,
                        align = 'center',
                        width = 1,
                      )

    plt.title( title, fontsize = 24 )

    ax.set_ylabel( y_ax_title, fontsize = 18 )
    ax.tick_params( axis = 'y', which='major', labelsize = 36 )

    plt.xlabel( x_ax_title, fontsize = 36 )
    plt.xticks( [ idx for idx,p in enumerate( tbv.pos ) if idx%( tick_spacing ) == 0 ],
                [ c for idx,c in enumerate( tbv[ pos_col ] ) if idx%( tick_spacing ) == 0 ],
                fontsize=36,
                rotation='vertical' )

    if hlines:
        for line in hlines:
            x_cds, col, style, lw = line
            ax.axhline( x_cds, c = col, ls = style, linewidth = lw, alpha = .6 )

    if shade_exons:

        for ex in shade_exons:

            plt.axvspan( col_pivot.index.get_loc( ex[0] ).start - .5,
                         col_pivot.index.get_loc( ex[1] ).stop,
                         facecolor = 'gray',
                         alpha = 0.15 )

    if scale_bar:

        assert scale_bar_loc.lower() in [ 'left', 'middle', 'right' ], \
        'Scale bar location can be "left", "middle", or "right".'

        if scale_bar_loc == 'left':
            x_loc = 3*tick_spacing
        elif scale_bar_loc == 'middle':
            x_loc = tbv.shape[0] / 2
        else:
            x_loc = tbv.shape[0] - 3*tick_spacing

        trans = tfrms.blended_transform_factory( ax.transData, ax.transAxes )
        ax.errorbar( x_loc,
                      0.92,
                      xerr = ( tick_spacing / 2 ),
                      color = 'black',
                      capsize=3,
                      transform=trans )

        txt = str( tick_spacing ) + ' bases' if tick_spacing > 1 else str( tick_spacing ) + ' base'
        plt.text( x_loc,
                  0.90,
                  txt,
                  horizontalalignment = 'center',
                  verticalalignment = 'top',
                  transform = trans,
                  fontsize = 14 )

    if legend:

        legend = ax.legend( title = legend_title,
                                 ncol = 2,
                                 bbox_to_anchor = ( 1, 1 ), #if you turn that on you can put legend outside of plot
                                 loc = legend_loc,
                                 fontsize = 14 )
        plt.setp( legend.get_title(), fontsize=14 )

        plt.setp( legend.get_title(), fontsize=14 )
    else:
        ax.legend_ = None
        plt.draw()

    if y_ax_lim:
        ax.set_ylim( y_ax_lim )

    art.set_snap( ax, snap )

    if tight:
        plt.tight_layout()

    return ax

def subplot_psi_by_alt(   var_df,
                          y_col,
                          pos_col,
                          colors,
                          color_col = 'alt',
                          edge_col = None,
                          ax = None,
                          shade_by_base = False,
                          shade_exons = False,
                          tick_spacing = 10,
                          bar_labels = False,
                          bar_label_loc = 'middle',
                          bar_labels_offset = True,
                          darken_bars = None,
                          darken_bars2 = None,
                          darken_edges = None,
                          labels_legend = False,
                          labels_legend_title = '',
                          labels_legend_loc = 'best',
                          wt_labels = False,
                          title = '',
                          y_ax_lim = None,
                          y_ax_title='',
                          x_ax_title = '',
                          legend = True,
                          legend_title = '',
                          legend_loc = 'best',
                          legend_labels = None,
                          tight = True,
                          print_ex_count = False,
                          scale_bar = False,
                          scale_bar_loc = 'left',
                          rev_trans = False,
                          hlines = None,
                          snap = False ):

    tbv = var_df.copy()

    if print_ex_count:
        print('This figure shows %i exonic bases.' % sum( c[1]-c[0] for c in shade_exons ) )

    if ax:
        ax = ax
    else:
        ax = plt.gca()


    col_pivot = tbv.pivot( index = [ 'pos', 'ref', pos_col ], columns = 'alt', values = y_col )

    col_pivot.plot.bar( ax = ax,
                        color = colors,
                        edgecolor = edge_col,
                        align = 'center',
                        width = 1,
                      )

    if bar_labels or wt_labels or shade_by_base or darken_bars or darken_edges:

        rects = ax.patches

        heights = [ rect.get_height() for rect in rects ]
        neg_ht = [ h for h in heights if h < 0 ]

        if bar_label_loc == 'above':
            trans = tfrms.blended_transform_factory( ax.transData,
                                                     ax.transAxes )
        elif bar_label_loc == 'middle':
            trans = tfrms.blended_transform_factory( ax.transData,
                                                     ax.transData )

    if darken_bars2:

        column, colors = darken_bars2

        #true false vector for where to put the marker
        _locs = tbv.pivot( index = [ 'pos' ], columns = 'alt', values = column )

        #bars are ordered with all A's first then all C's... this creates a colors vector to match that pattern
        colors_vec = [ c for c in colors for i in range( _locs.shape[0] ) ]

        locs = []
        for col in _locs.columns:
            locs.extend( _locs[ col ].tolist() )

        for jidx, rloc in enumerate( zip( rects, locs ) ):

                rect, loc = rloc

                if loc and not np.isnan( loc ):

                        color = colors_vec[ jidx ]

                        rect.set_facecolor( color )

    if darken_bars:

        column, colors = darken_bars

        #true false vector for where to put the marker
        _locs = tbv.pivot( index = [ 'pos' ], columns = 'alt', values = column )

        #bars are ordered with all A's first then all C's... this creates a colors vector to match that pattern
        colors_vec = [ c for c in colors for i in range( _locs.shape[0] ) ]

        locs = []
        for col in _locs.columns:
            locs.extend( _locs[ col ].tolist() )

        for jidx, rloc in enumerate( zip( rects, locs ) ):

                rect, loc = rloc

                if loc and not np.isnan( loc ):

                    color = colors_vec[ jidx ]

                    rect.set_facecolor( color )

    if darken_edges:

        column, color = darken_edges

        #true false vector for where to put the marker
        _locs = tbv.pivot( index = [ 'pos' ], columns = 'alt', values = column )

        locs = []
        for col in _locs.columns:
            locs.extend( _locs[ col ].tolist() )

        for jidx, rloc in enumerate( zip( rects, locs ) ):

                rect, loc = rloc

                if loc and not np.isnan( loc ):

                        rect.set_edgecolor( color )

    if wt_labels:

        labels = col_pivot.index.get_level_values( 'ref' )

        min_ht = min( neg_ht ) if len( neg_ht ) > 0 else 0
        neg_ht_mean = np.mean( neg_ht ) if len( neg_ht ) > 0 else -4

        #the way I'm messing with rects should get the label in the middle of the four bars for the pos
        for rect, label in zip( rects[ 2*len( labels ): ], labels ):
            ax.text( rect.get_x(),
                     min_ht + 4*neg_ht_mean,
                     label,
                     fontsize = 18,
                     fontweight = 'bold',
                     ha='center',
                     va='bottom' )

    if bar_labels:

        if bar_labels_offset:

            if bar_label_loc == 'middle':

                max_rect_ht = y_ax_lim[ 1 ] if y_ax_lim else ax.get_ylim()[ 1 ]

                mid_rect_ht = max_rect_ht / 2

                mid_label = int( len( bar_labels ) / 2 )

                #sets up constant y label positions
                #first label is the lowest and they are centered in the middle of the positive y axis
                label_pos = [ mid_rect_ht + ( i - ( mid_label ) )*.2*max_rect_ht for i in range( len( bar_labels ) ) ]

            elif bar_label_loc == 'above':

                label_pos = [ 1 + .1*i for i in range( len( bar_labels ) ) ]

        else:

            if bar_label_loc == 'middle':

                pos_ht_mean = np.mean( pos_ht_sum )

                max_rect_ht = y_ax_lim[ 1 ] if y_ax_lim else ax.get_ylim()[ 1 ]

                mid_rect_ht = max_rect_ht / 2

                #sets up constant y label positions
                #they are centered in the middle of the positive y axis
                label_pos = [ mid_rect_ht for i in range( len( bar_labels ) ) ]

            elif bar_label_loc == 'above':

                label_pos = [ .885 if marker == r"$\bullet$" else 1.01 for _col, marker, _label, _fsize in bar_labels ]

        for iidx, colmark in enumerate( bar_labels ):

            col, marker, _label, fsize = colmark

            #true false vector for where to put the marker
            _locs = tbv.pivot( index = [ 'pos' ], columns = 'alt', values = col )

            locs = []
            for col in _locs.columns:
                locs.extend( _locs[ col ].tolist() )

            for jidx, rloc in enumerate( zip( rects, locs ) ):

                rect, loc = rloc

                xpos = rect.get_x() + rect.get_width() / 2

                if marker == r"$\bullet$":
                    xpos = .99*( xpos )

                if loc and not np.isnan( loc ):

                    ax.text( xpos,
                             label_pos[ iidx ],
                             marker,
                             fontsize = fsize,
                             transform = trans,
                             ha = 'center',
                             va = 'bottom' )

    plt.title( title, fontsize = 24 )

    ax.set_ylabel( y_ax_title, fontsize = 18 )
    ax.tick_params( axis = 'y', which='major', labelsize = 36 )

    plt.xlabel( x_ax_title, fontsize = 36 )
    plt.xticks( [ idx for idx,p in enumerate( col_pivot.index.get_level_values( 'pos' ) ) if idx%( tick_spacing ) == 0 ],
                [ c for idx,c in enumerate( col_pivot.index.get_level_values( pos_col ) ) if idx%( tick_spacing ) == 0 ],
                fontsize=36,
                rotation='vertical' )

    if hlines:
        for line in hlines:
            x_cds, col, style, lw = line
            ax.axhline( x_cds, c = col, ls = style, linewidth = lw, alpha = .6 )

    if shade_by_base:

        for i in range( len( set( tbv.pos ) ) ):

            #shade only every other position
            if i % 2 == 0:

                rect = rects[ i ]

                #this plot has four bases at every position
                ax.axvspan( rect.get_x(),
                             rect.get_x() + 4*rect.get_width(),
                             facecolor = 'gray',
                             alpha = 0.15 )

    if shade_exons:

        for ex in shade_exons:

            plt.axvspan( col_pivot.index.get_loc( ex[0] ).start - .5,
                         col_pivot.index.get_loc( ex[1] ).stop,
                         facecolor = 'gray',
                         alpha = 0.15 )

    if scale_bar:

        assert scale_bar_loc.lower() in [ 'left', 'middle', 'right' ], \
        'Scale bar location can be "left", "middle", or "right".'

        if scale_bar_loc == 'left':
            x_loc = 3*tick_spacing
        elif scale_bar_loc == 'middle':
            x_loc = tbv.shape[0] / 2
        else:
            x_loc = tbv.shape[0] - 3*tick_spacing

        trans = tfrms.blended_transform_factory( ax.transData, ax.transAxes )
        ax.errorbar( x_loc,
                      0.92,
                      xerr = ( tick_spacing / 2 ),
                      color = 'black',
                      capsize=3,
                      transform=trans )

        txt = str( tick_spacing ) + ' bases' if tick_spacing > 1 else str( tick_spacing ) + ' base'
        plt.text( x_loc,
                  0.90,
                  txt,
                  horizontalalignment = 'center',
                  verticalalignment = 'top',
                  transform = trans,
                  fontsize = 14 )

    if legend:

        if legend_labels:
            legend = ax.legend( title = legend_title,
                                 ncol = 2,
                                 bbox_to_anchor = ( 1, 1 ), #if you turn that on you can put legend outside of plot
                                 loc = legend_loc,
                                 labels = legend_labels,
                                 fontsize = 14 )
        else:
            legend = ax.legend( title = legend_title,
                                 ncol = 2,
                                 bbox_to_anchor = ( 1, 1 ), #if you turn that on you can put legend outside of plot
                                 loc = legend_loc,
                                 fontsize = 14 )
        plt.setp( legend.get_title(), fontsize=14 )
    elif labels_legend:

        fake_handles = [ plt.Rectangle( (0, 0), 0, 0, fill=False, edgecolor='none', visible = 'False')
                        for i in range( len( bar_labels ) ) ]

        legend = ax.legend(  handles = fake_handles,
                             title = labels_legend_title,
                             bbox_to_anchor = ( 0, 1 ), #if you turn that on you can put legend outside of plot
                             loc = labels_legend_loc,
                             labels = ( symbol + ' '*6 + label for _col, symbol, label, _fsize in bar_labels ),
                             fontsize = 14 )
        plt.setp( legend.get_title(), fontsize=14 )
    else:
        ax.legend_ = None
        plt.draw()

    if y_ax_lim:
        ax.set_ylim( y_ax_lim )

    art.set_snap( ax, snap )

    if tight:
        plt.tight_layout()

    return ax

def subplots_wrapper( var_df,
                      y_cols,
                      pos_col,
                     colors,
                     fig_size = ( 30, 10 ),
                     share_x = True,
                     share_y = True,
                     color_col = 'alt',
                     edge_col = None,
                     shade_by_base = False,
                     shade_exons = False,
                     zoom=None,
                     tick_spacing = 10,
                     wt_labels = False,
                     bar_labels = False,
                     bar_label_loc = 'middle',
                     bar_labels_offset = True,
                     darken_bars = None,
                     darken_bars2 = None,
                     darken_edges = None,
                     labels_legend = False,
                     labels_legend_title = '',
                     labels_legend_loc = 'best',
                     title = '',
                     y_ax_title='',
                     y_ax_lim = None,
                     x_ax_title = '',
                     legend = True,
                     legend_title = '',
                     legend_loc = 'best',
                     legend_labels = None,
                     tight = True,
                     print_ex_count = False,
                     scale_bar = False,
                     scale_bar_loc = 'left',
                     rev_trans = False,
                     hlines = None,
                     savefile = None,
                     snap = False
                     ):

    tbv = var_df.sort_values( by = [ 'pos', 'alt' ] ).copy()

    if zoom:
        assert zoom[1] > zoom[0], 'Your final zoom coordinate must be larger than the first zoom coordinate'
        tbv_filt = tbv.set_index( 'pos' ).loc[ zoom[0]:zoom[1] ].reset_index()
    else:
        tbv_filt = tbv

    fig, axes = plt.subplots( len( y_cols ),
                              sharex = share_x,
                              sharey = share_y,
                              figsize = fig_size )

    if len( y_cols ) == 1:
        axes = [ axes ]

    if bar_label_loc == 'above':
        all_bar_labels = False
    elif bar_label_loc == 'middle':
        all_bar_labels = bar_labels

    if not darken_bars:
        dbar_dict = { i: None for i in range( len( y_cols) ) }
    elif len( darken_bars ) == 1:
        dbar_dict = { i: darken_bars[ 0 ] for i in range( len( y_cols) ) }
    elif len( darken_bars ) == len( y_cols ):
        dbar_dict = { i: darken_bars[ i ] for i in range( len( y_cols) ) }
    else:
        print( 'Something is wrong with how you entered the darken bars argument!' )

    if not darken_bars2:
        dbar2_dict = { i: None for i in range( len( y_cols) ) }
    elif len( darken_bars2 ) == 1:
        dbar2_dict = { i: darken_bars2[ 0 ] for i in range( len( y_cols) ) }
    elif len( darken_bars2 ) == len( y_cols ):
        dbar2_dict = { i: darken_bars2[ i ] for i in range( len( y_cols) ) }
    else:
        print( 'Something is wrong with how you entered the darken bars2 argument!' )

    if not hlines:
        hline_dict = { i: None for i in range( len( y_cols) ) }
    elif len( hlines ) == 1:
        hline_dict = { i: hlines[ 0 ] for i in range( len( y_cols) ) }
    elif len( hlines ) == len( y_cols ):
        hline_dict = { i: hlines[ i ] for i in range( len( y_cols) ) }
    else:
        print( 'Something is wrong with how you entered the hlines argument!' )

    if not y_ax_lim:
        ylim_dict = { i: None for i in range( len( y_cols) ) }
    elif len( y_ax_lim ) == 1:
        ylim_dict = { i: y_ax_lim[ 0 ] for i in range( len( y_cols) ) }
    elif len( y_ax_lim ) > 1 and share_y:
        print( 'You specified too many y axis limits to have them all sharey' )
    elif len( y_ax_lim ) == len( y_cols ):
        ylim_dict = { i: y_ax_lim[ i ] for i in range( len( y_cols) ) }
    else:
        print( 'Something is wrong with how you entered the y axis limits argument!' )

    if wt_labels:

        ht_max = tbv_filt[ y_cols ].max().max()
        ht_min = tbv_filt[ y_cols ].min().min()

        y_max = y_ax_lim[ -1 ][ 1 ] if y_ax_lim else 1.01*ht_max
        y_min = ht_min + ht_min if ht_min < 0 else -20
        ylim_dict[ len( y_cols ) - 1 ] = ( y_min, y_max )

    #this whole loop is basically to give each its own y axis labels
    #and to only have the legend & scale bar on the top plot
    #and to only have wt labels on the bottom plot
    for idx, col in enumerate( y_cols ):

        if idx == 0:
            subplot_psi_by_alt(   tbv_filt,
                                  col,
                                  pos_col,
                                  colors,
                                  ax = axes[ idx ],
                                  color_col = color_col,
                                 edge_col = edge_col,
                                 shade_by_base = shade_by_base,
                                 shade_exons = shade_exons,
                                 tick_spacing = tick_spacing,
                                 bar_labels = bar_labels,
                                 bar_label_loc = bar_label_loc,
                                 bar_labels_offset = bar_labels_offset,
                                 darken_bars = dbar_dict[ idx ],
                                 darken_bars2 = dbar2_dict[ idx ],
                                 darken_edges = darken_edges,
                                 labels_legend = labels_legend,
                                 labels_legend_title = labels_legend_title,
                                 labels_legend_loc = labels_legend_loc,
                                 title = title,
                                 y_ax_lim = ylim_dict[ idx ],
                                 y_ax_title = y_ax_title[ idx ],
                                 legend = legend,
                                 legend_title = legend_title,
                                 legend_loc = legend_loc,
                                 legend_labels = legend_labels,
                                 tight = tight,
                                 print_ex_count = print_ex_count,
                                 scale_bar = scale_bar,
                                 scale_bar_loc = scale_bar_loc,
                                 rev_trans = rev_trans,
                                 hlines = hline_dict[ idx ],
                                 snap = snap )
        elif idx == ( len( y_cols ) - 1 ):
            subplot_psi_by_alt(   tbv_filt,
                                  col,
                                  pos_col,
                                  colors,
                                  ax = axes[ idx ],
                                  color_col = color_col,
                                 edge_col = edge_col,
                                 shade_by_base = shade_by_base,
                                 shade_exons = shade_exons,
                                 tick_spacing = tick_spacing,
                                 bar_labels = all_bar_labels,
                                 darken_bars = dbar_dict[ idx ],
                                 darken_bars2 = dbar2_dict[ idx ],
                                 darken_edges = darken_edges,
                                 wt_labels = wt_labels,
                                 legend = False,
                                 y_ax_lim = ylim_dict[ idx ],
                                 y_ax_title = y_ax_title[ idx ],
                                 tight = tight,
                                 rev_trans = rev_trans,
                                 hlines = hline_dict[ idx ],
                                 x_ax_title = x_ax_title,
                                 snap = snap )
        else:
            subplot_psi_by_alt(   tbv_filt,
                                  col,
                                  pos_col,
                                  colors,
                                  ax = axes[ idx ],
                                  color_col = color_col,
                                 edge_col = edge_col,
                                 shade_by_base = shade_by_base,
                                 shade_exons = shade_exons,
                                 tick_spacing = tick_spacing,
                                 bar_labels = all_bar_labels,
                                 darken_bars = dbar_dict[ idx ],
                                 darken_bars2 = dbar2_dict[ idx ],
                                 darken_edges = darken_edges,
                                 y_ax_lim = ylim_dict[ idx ],
                                 legend = False,
                                 y_ax_title = y_ax_title[ idx ],
                                 tight = tight,
                                 rev_trans = rev_trans,
                                 hlines = hline_dict[ idx ],
                                 snap = snap )

    if savefile:
        plt.savefig( savefile,
                     #dpi = 300,
                     bbox_inches = 'tight'
                   )

    plt.show()

def split_ax_bcs( var_df,
                  bc_col,
                  pos_col,
                  colors,
                  y_lims,
                  index_cols = [ 'pos', 'alt' ],
                  fig_size = ( 30, 5 ),
                  hratios = [ 1, 1 ],
                  zoom=None,
                  tick_spacing = 10,
                  title = '',
                  y_ax_title='',
                  y_ax_lim = None,
                  x_ax_title = '',
                  legend = True,
                  legend_title = '',
                  legend_loc = 'best',
                  legend_labels = None,
                  tight = True,
                  savefile = None
                  ):

    tbv = var_df.sort_values( by = index_cols ).copy()

    if zoom:
        assert zoom[1] > zoom[0], 'Your final zoom coordinate must be larger than the first zoom coordinate'
        tbv_filt = tbv.set_index( 'pos' ).loc[ zoom[0]:zoom[1] ].reset_index()
    else:
        tbv_filt = tbv

    #check if plot will lose bars due to not enough space for all the pixels
    dpi = 100
    if ( tbv_filt.shape[0]*1.5 ) > ( fig_size[0]*dpi ):
        fig_height = fig_size[1]
        fig_size = ( ( tbv_filt.shape[0]*1.5 / dpi ), fig_height )
        print('Adjusting figure width to accomodate all pixels...')

    fig = plt.figure( figsize = fig_size )
    gs = gridspec.GridSpec( nrows = len( y_lims ),
                            ncols = 1,
                            height_ratios = hratios )

    col_pivot = tbv_filt.pivot( index = 'pos', columns = 'alt', values = bc_col )

    axes = []
    for i in range( len( y_lims ) ):
        axes.append( plt.subplot( gs[ i ],
                                  #sharex = True
                                )
                    )
        col_pivot.plot.bar( ax = axes[ -1 ],
                            color = colors,
                            stacked = True,
                            align = 'center',
                            width = 1, )
        axes[ - 1 ].set_ylim( y_lims[ i ] )
        axes[ -1 ].tick_params( axis = 'y', which='major', labelsize = 36 )

    for i, ax in enumerate( axes ):
        if i != ( len( y_lims ) - 1 ):
            ax.spines[ 'bottom' ].set_visible( False )
            ax.tick_params( axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
            ax.set_xlabel( '' )
        else:
            ax.set_xlabel( x_ax_title, fontsize = 36 )
            ax.tick_params( axis = 'x', which='major', labelsize = 36, labelrotation = 90 )

        if i != 0:
            ax.spines[ 'top' ].set_visible( False )

    pos_pivot = tbv_filt.pivot( index = 'pos', columns = 'alt', values = pos_col )
    pos_pivot = pos_pivot.fillna( '' )
    pos_pivot[ pos_col ] = [ a if a != '' else c for a,c in zip( pos_pivot.A, pos_pivot.C ) ]
    plt.xticks( [ idx for idx,p in enumerate( col_pivot.index ) if idx % tick_spacing == 1 ],
                [ c for idx,c in enumerate( pos_pivot[ pos_col ] ) if idx % tick_spacing == 1 ],)

    if legend:

        if legend_labels:
            legend = plt.legend( title = legend_title,
                                 bbox_to_anchor = ( 1, 1 ), #if you turn that on you can put legend outside of plot
                                 loc = legend_loc,
                                 labels = legend_labels,
                                 fontsize = 14 )
        else:
            legend = plt.legend( title = legend_title,
                                 bbox_to_anchor = ( 1, 1 ), #if you turn that on you can put legend outside of plot
                                 loc = legend_loc,
                                 fontsize = 14 )
        plt.setp( legend.get_title(), fontsize=14 )
    else:
        for ax in axes:
            ax.legend_ = None
            plt.draw()

    if tight:
        plt.tight_layout()

    if savefile:
        plt.savefig( savefile,
                     #dpi = 300,
                     bbox_inches = 'tight'
                   )

    plt.show()

def sat_subplot_psi_by_alt(   var_df,
                          y_col,
                          pos_col,
                          colors,
                          color_col = 'alt',
                          ax = None,
                          tick_spacing = 10,
                          darken_bars = None,
                          darken_bars2 = None,
                          labels_legend = False,
                          labels_legend_title = '',
                          labels_legend_loc = 'best',
                          title = '',
                          y_ax_lim = None,
                          y_ax_title='',
                          x_ax_title = '',
                          legend = True,
                          legend_title = '',
                          legend_loc = 'best',
                          legend_labels = None,
                          ref_labels = None,
                          colors_ref = None,
                          y_ref_cds = -3,
                          ref_font_size = 40,
                          ref_rect_ht = 50,
                          bar_labels = None,
                          hatch_missing = False,
                          hlines = None,
                          tight = True,
                          save_margin = .1 ):

    tbv = var_df.sort_values( by = [ 'pos', 'alt' ] ).reset_index( drop = True ).copy()

    if ax:
        ax = ax
    else:
        ax = plt.gca()

    color_d = { alt: colors[ idx ] for idx,alt in enumerate( sorted( tbv[ color_col ].unique() ) ) }

    for alt in sorted( tbv[ color_col ].unique() ):

        ax.bar( tbv.loc[ tbv[ color_col ] == alt ].index,
                 tbv.loc[ tbv[ color_col ] == alt ][ y_col ],
                 label = alt,
                 color = color_d[ alt ],
                 align = 'center',
                 width = 1, )

    if hlines:

        ax.axhline( y = hlines[ 0 ], color = hlines[ 1 ], linestyle = hlines[ 2 ], zorder = 1 )

    if hatch_missing:

        missing_cds = tbv.loc[ tbv[ hatch_missing[ 0 ] ].isnull() ].index.tolist()

        if y_ax_lim:

            y_min,y_max = y_ax_lim

        else:

            y_min,y_max = ax.get_ylim()

        for cds in missing_cds:

            rect = ax.add_patch( plt.Rectangle( ( cds - .5, y_min ),
                                                1,
                                                y_max - y_min,
                                                fill = True,
                                                facecolor = 'white',
                                                hatch = hatch_missing[ 1 ],
                                                linewidth = 0,
                                                zorder = 2 ), )

    if darken_bars:

        #print( darken_bars )

        #for dbars in darken_bars:

        column, colors2 = darken_bars

        color_d2 = { alt: colors2[ idx ] for idx,alt in enumerate( sorted( tbv[ color_col ].unique() ) ) }

        for alt in sorted( tbv[ color_col ].unique() ):

            ax.bar( tbv.loc[ ( tbv[ color_col ] == alt ) & ( tbv[ column ] ) ].index,
                    tbv.loc[ ( tbv[ color_col ] == alt ) & ( tbv[ column ] ) ][ y_col ],
                    label = alt,
                    color = color_d2[ alt ],
                    align = 'center',
                    width = 1, )

    if darken_bars2:

        #for dbars2 in darken_bars2:

        column, colors2 = darken_bars2

        color_d2 = { alt: colors2[ idx ] for idx,alt in enumerate( sorted( tbv[ color_col ].unique() ) ) }

        for alt in sorted( tbv[ color_col ].unique() ):

            ax.bar( tbv.loc[ ( tbv[ color_col ] == alt ) & ( tbv[ column ] ) ].index,
                    tbv.loc[ ( tbv[ color_col ] == alt ) & ( tbv[ column ] ) ][ y_col ],
                    label = alt,
                    color = color_d2[ alt ],
                    align = 'center',
                    width = 1, )

    plt.title( title, fontsize = 24 )

    ax.set_ylabel( y_ax_title, fontsize = 18 )
    ax.tick_params( axis = 'y', which='major', labelsize = 36 )

    ax.set_xlim( ( -.5, len( tbv ) - .5 ) )

    plt.xlabel( x_ax_title, fontsize = 36 )
    plt.xticks( [ 3*idx + 1 for idx,p in enumerate( tbv.pos.unique() ) if p % ( tick_spacing ) == 0 ],
                [ tbv.iloc[ 3*idx + 1 ][ pos_col ] for idx,p in enumerate( tbv.pos.unique() ) if p % ( tick_spacing ) == 0 ],
                fontsize=36,
                rotation='vertical' )

    if ref_labels:

        assert tick_spacing == 1, 'If your tick_spacing is not set to 1, the ref bases will be wrong!'

        if tight:
            print( 'Are your rectangles not positioned right? Turn off tight!')

        ref_bases = list( tbv.groupby( 'pos' ).ref.apply( lambda x: x.unique()[ 0 ] ) )

        if colors_ref:

            colors_ref_d = { ref: colors_ref[ idx ] for idx,ref in enumerate( sorted( list( set( ref_bases ) ) ) ) }

        else:

            colors_ref_d = { ref: 'white' for ref in ref_bases.unique() }

        x_coords = ax.get_xticks()

        for base,cds in zip( ref_bases, x_coords ):

            rect = ax.add_patch( plt.Rectangle( ( cds - 1.5, y_ref_cds ),
                                                3,
                                                ref_rect_ht,
                                                color = colors_ref_d[ base ],
                                                clip_on = False, ) )

            ax.text( cds,
                     y_ref_cds + ( ref_rect_ht / 2 ),
                     base,
                     ha = 'center',
                     va = 'center',
                     fontsize = ref_font_size, )
                     #bbox = { 'facecolor': colors_ref_d[ base ], 'pad': 40 } )

            #ax.add_patch(  )

            #p = patches.Rectangle( ( cds - 1.5, y_ref_cds ), 3, 1, color = colors_ref_d[ base ] )
            #ax.add_artist(p)

            #plt.text( cds, -.5, base, horizontalalignment = 'center', verticalalignment = 'center', fontsize=40, color='black')

    if bar_labels:

        print( 'Your labels might not show up in the notebook! Check saved pdf before butchering code!')

        if save_margin <= .1:

            print( 'If your labels are not in the pdf, increase save_margin parameter!' )

        x_coords = ax.get_xticks()

        x_coords_per_bp = [ x for x in range( x_coords[ -1 ] + 2 ) ]

        for label_col, marker_d, y_coords, in bar_labels:

            for val in marker_d:

                marker, color, edgecolor, linewidth, size = marker_d[ val ]

                ax.scatter( x_coords_per_bp,
                            [ y_coords if label == val else np.nan for label in tbv[ label_col ] ],
                            marker = marker,
                            color = color,
                            edgecolor = edgecolor,
                            linewidth = linewidth,
                            s = size,
                            zorder = 100,
                            clip_on = False )

    if legend:

        if legend_labels:
            legend = ax.legend( title = legend_title,
                                 ncol = 2,
                                 bbox_to_anchor = ( 1, 1 ), #if you turn that on you can put legend outside of plot
                                 loc = legend_loc,
                                 labels = legend_labels,
                                 fontsize = 14 )
        else:
            legend = ax.legend( title = legend_title,
                                 ncol = 2,
                                 bbox_to_anchor = ( 1, 1 ), #if you turn that on you can put legend outside of plot
                                 loc = legend_loc,
                                 fontsize = 14 )
        plt.setp( legend.get_title(), fontsize=14 )
    elif labels_legend:

        fake_handles = [ plt.Rectangle( (0, 0), 0, 0, fill=False, edgecolor='none', visible = 'False')
                        for i in range( len( bar_labels ) ) ]

        legend = ax.legend(  handles = fake_handles,
                             title = labels_legend_title,
                             bbox_to_anchor = ( 0, 1 ), #if you turn that on you can put legend outside of plot
                             loc = labels_legend_loc,
                             labels = ( symbol + ' '*6 + label for _col, symbol, label, _fsize in bar_labels ),
                             fontsize = 14 )
        plt.setp( legend.get_title(), fontsize=14 )
    else:
        ax.legend_ = None
        plt.draw()

    if y_ax_lim:
        ax.set_ylim( y_ax_lim )

    if tight:
        plt.tight_layout()

    return ax

def sat_subplots_wrapper( var_df,
                      y_cols,
                      pos_col,
                     colors,
                     fig_size = ( 30, 10 ),
                     share_x = True,
                     share_y = True,
                     color_col = 'alt',
                     zoom=None,
                     tick_spacing = 10,
                     darken_bars = None,
                     darken_bars2 = None,
                     labels_legend = False,
                     labels_legend_title = '',
                     labels_legend_loc = 'best',
                     title = '',
                     y_ax_title='',
                     y_ax_lim = None,
                     x_ax_title = '',
                     legend = True,
                     legend_title = '',
                     legend_loc = 'best',
                     legend_labels = None,
                     ref_labels = None,
                     y_ref_cds = -3,
                     colors_ref = None,
                     ref_font_size = 40,
                     ref_rect_ht = 50,
                     bar_labels = None,
                     hatch_missing = False,
                     hlines = None,
                     tight = True,
                     savefile = None,
                     save_margin = .1,
                     ):

    tbv = var_df.sort_values( by = [ 'pos', 'alt' ] ).copy()

    if zoom:
        assert zoom[1] > zoom[0], 'Your final zoom coordinate must be larger than the first zoom coordinate'
        tbv_filt = tbv.set_index( 'pos' ).loc[ zoom[0]:zoom[1] ].reset_index()
    else:
        tbv_filt = tbv

    fig, axes = plt.subplots( len( y_cols ),
                              sharex = share_x,
                              sharey = share_y,
                              figsize = fig_size )

    if len( y_cols ) == 1:
        axes = [ axes ]

    if not darken_bars:
        dbar_dict = { i: None for i in range( len( y_cols) ) }
    elif len( darken_bars ) == 1:
        dbar_dict = { i: darken_bars[ 0 ] for i in range( len( y_cols) ) }
    elif len( darken_bars ) == len( y_cols ):
        dbar_dict = { i: darken_bars[ i ] for i in range( len( y_cols) ) }
    else:
        print( 'Something is wrong with how you entered the darken bars argument!' )

    if not darken_bars2:
        dbar_dict2 = { i: None for i in range( len( y_cols) ) }
    elif len( darken_bars2 ) == 1:
        dbar_dict2 = { i: darken_bars2[ 0 ] for i in range( len( y_cols) ) }
    elif len( darken_bars2 ) == len( y_cols ):
        dbar_dict2 = { i: darken_bars2[ i ] for i in range( len( y_cols) ) }
    else:
        print( 'Something is wrong with how you entered the darken bars2 argument!' )

    if not y_ax_lim:
        ylim_dict = { i: None for i in range( len( y_cols) ) }
    elif len( y_ax_lim ) == 1:
        ylim_dict = { i: y_ax_lim[ 0 ] for i in range( len( y_cols) ) }
    elif len( y_ax_lim ) > 1 and share_y:
        print( 'You specified too many y axis limits to have them all sharey' )
    elif len( y_ax_lim ) == len( y_cols ):
        ylim_dict = { i: y_ax_lim[ i ] for i in range( len( y_cols) ) }
    else:
        print( 'Something is wrong with how you entered the y axis limits argument!' )

    if not hatch_missing:
        hatch_dict = { i: None for i in range( len( y_cols) ) }
    elif len( hatch_missing ) == 1:
        hatch_dict = { i: hatch_missing[ 0 ] for i in range( len( y_cols) ) }
    elif len( hatch_missing ) == len( y_cols ):
        hatch_dict = { i: hatch_missing[ i ] for i in range( len( y_cols) ) }
    else:
        print( 'Something is wrong with how you entered the hatch missing argument!' )

    if not hlines:
        hline_dict = { i: None for i in range( len( y_cols) ) }
    elif len( hlines ) == 1:
        hline_dict = { i: hlines[ 0 ] for i in range( len( y_cols) ) }
    elif len( hlines ) == len( y_cols ):
        hline_dict = { i: hlines[ i ] for i in range( len( y_cols) ) }
    else:
        print( 'Something is wrong with how you entered the hlines argument!' )

    #this whole loop is basically to give each its own y axis labels
    #and to only have the legend & scale bar on the top plot
    #and to only have wt labels on the bottom plot
    for idx, col in enumerate( y_cols ):

        if idx == 0 and len( y_cols ) == 1:
            sat_subplot_psi_by_alt(   tbv_filt,
                                  col,
                                  pos_col,
                                  colors,
                                  ax = axes[ idx ],
                                  color_col = color_col,
                                 tick_spacing = tick_spacing,
                                   darken_bars = dbar_dict[ idx ],
                                 darken_bars2 = dbar_dict2[ idx ],
                                 labels_legend = labels_legend,
                                 labels_legend_title = labels_legend_title,
                                 labels_legend_loc = labels_legend_loc,
                                 title = title,
                                 y_ax_lim = ylim_dict[ idx ],
                                 y_ax_title = y_ax_title[ idx ],
                                 x_ax_title = x_ax_title,
                                 legend = legend,
                                 legend_title = legend_title,
                                 legend_loc = legend_loc,
                                 legend_labels = legend_labels,
                                 ref_labels = ref_labels,
                                 y_ref_cds = y_ref_cds,
                                 colors_ref = colors_ref,
                                 ref_font_size = ref_font_size,
                                 ref_rect_ht = ref_rect_ht,
                                 bar_labels = bar_labels,
                                 hatch_missing = hatch_dict[ idx ],
                                 hlines = hline_dict[ idx ],
                                 tight = tight,
                                 save_margin = save_margin )

        elif idx == 0:
            sat_subplot_psi_by_alt(   tbv_filt,
                                  col,
                                  pos_col,
                                  colors,
                                  ax = axes[ idx ],
                                  color_col = color_col,
                                 tick_spacing = tick_spacing,
                                   darken_bars = dbar_dict[ idx ],
                                 darken_bars2 = dbar_dict2[ idx ],
                                 labels_legend = labels_legend,
                                 labels_legend_title = labels_legend_title,
                                 labels_legend_loc = labels_legend_loc,
                                 title = title,
                                 y_ax_lim = ylim_dict[ idx ],
                                 y_ax_title = y_ax_title[ idx ],
                                 legend = legend,
                                 legend_title = legend_title,
                                 legend_loc = legend_loc,
                                 legend_labels = legend_labels,
                                 bar_labels = bar_labels,
                                 hatch_missing = hatch_dict[ idx ],
                                 hlines = hline_dict[ idx ],
                                 tight = tight,
                                 save_margin = save_margin )

        elif idx == ( len( y_cols ) - 1 ):
            sat_subplot_psi_by_alt(   tbv_filt,
                                  col,
                                  pos_col,
                                  colors,
                                  ax = axes[ idx ],
                                  color_col = color_col,
                                 tick_spacing = tick_spacing,
                                   darken_bars = dbar_dict[ idx ],
                                 darken_bars2 = dbar_dict2[ idx ],
                                 legend = False,
                                 y_ax_lim = ylim_dict[ idx ],
                                 y_ax_title = y_ax_title[ idx ],
                                 ref_labels = ref_labels,
                                 y_ref_cds = y_ref_cds,
                                 colors_ref = colors_ref,
                                 ref_font_size = ref_font_size,
                                 ref_rect_ht = ref_rect_ht,
                                 hatch_missing = hatch_dict[ idx ],
                                 hlines = hline_dict[ idx ],
                                 tight = tight,
                                 x_ax_title = x_ax_title, )
        else:
            sat_subplot_psi_by_alt(   tbv_filt,
                                  col,
                                  pos_col,
                                  colors,
                                  ax = axes[ idx ],
                                  color_col = color_col,
                                 tick_spacing = tick_spacing,
                                 darken_bars = dbar_dict[ idx ],
                                 darken_bars2 = dbar_dict2[ idx ],
                                 y_ax_lim = ylim_dict[ idx ],
                                 legend = False,
                                 y_ax_title = y_ax_title[ idx ],
                                 hatch_missing = hatch_dict[ idx ],
                                 hlines = hline_dict[ idx ],
                                 tight = tight, )

    if savefile:
        plt.savefig( savefile,
                     #dpi = 300,
                     bbox_inches = 'tight',
                     pad_inches = save_margin,
                   )

    plt.show()

def plot_stacked_psi(     var_df,
                          zcols,
                          pos_col,
                          colors,
                          index_cols = [ 'pos', 'alt' ],
                          edge_col = None,
                          fig_size = ( 30, 5 ),
                          shade_exons = False,
                          shade_by_base = False,
                          zoom=None,
                          tick_spacing = 10,
                          alt_labels = False,
                          bar_labels = False,
                          bar_label_loc = 'middle',
                          bar_labels_offset = True,
                          darken_bars = False,
                          darken_edges = False,
                          labels_legend = False,
                          labels_legend_title = '',
                          labels_legend_loc = 'best',
                          title = '',
                          y_ax_title='',
                          y_ax_lim = None,
                          x_ax_title = '',
                          legend = True,
                          legend_title = '',
                          legend_loc = 'best',
                          legend_labels = None,
                          tight = True,
                          print_ex_count = False,
                          scale_bar = False,
                          scale_bar_loc = 'left',
                          rev_trans = False,
                          hlines = None,
                          savefile = None ):

    tbv = var_df.sort_values( by = index_cols ).copy()

    if zoom:
        assert zoom[1] > zoom[0], 'Your final zoom coordinate must be larger than the first zoom coordinate'
        tbv_filt = tbv.set_index( 'pos' ).loc[ zoom[0]:zoom[1] ].reset_index()
    else:
        tbv_filt = tbv

    #check if plot will lose bars due to not enough space for all the pixels
    dpi = 100
    if ( tbv_filt.shape[0]*1.5 ) > ( fig_size[0]*dpi ):
        fig_height = fig_size[1]
        fig_size = ( ( tbv_filt.shape[0]*1.5 / dpi ), fig_height )
        print('Adjusting figure width to accomodate all pixels...')

    if print_ex_count:
        print('This figure shows %i exonic bases.' % sum( c[1]-c[0] for c in shade_exons ) )

    #col_pivot = tbv_filt.set_index( index_cols )[ zcols ]

    col_pivot = tbv_filt.pivot( index = 'pos', columns = 'alt', values = zcols )

    col_pivot.plot.bar( color = colors,
                        edgecolor = edge_col,
                        stacked = True,
                        align = 'center',
                        width = 1,
                        figsize = fig_size )

    ax = plt.gca()

    if alt_labels or bar_labels or shade_by_base or scale_bar or darken_bars or darken_edges:

        rects = ax.patches

        heights = [ rect.get_height() for rect in rects ]

        n_bars = int( len( heights ) / len( zcols ) )

        pos_ht_sum = [ sum( heights[ i ] for i in range( j, len( heights ), n_bars ) if heights[ i ] > 0 )
                       for j in range( n_bars ) ]
        neg_ht_sum = [ sum( heights[ i ] for i in range( j, len( heights ), n_bars ) if heights[ i ] < 0 )
                       for j in range( n_bars ) ]

        if bar_label_loc == 'above':
            trans = tfrms.blended_transform_factory( ax.transData,
                                                     ax.transAxes )
        else:
            trans = tfrms.blended_transform_factory( ax.transData,
                                                     ax.transData )

    if darken_bars:

        column, colors = darken_bars

        locs = tbv_filt[ column ]

        for jidx, rloc in enumerate( zip( rects, locs ) ):

                rect, loc = rloc

                if loc:

                    for iidx in range( len( zcols ) ):

                        color = colors[ iidx ]

                        rects[ jidx + iidx*n_bars ].set_facecolor( color )

    if darken_edges:

        column, color = darken_edges

        locs = tbv_filt[ column ]

        for jidx, rloc in enumerate( zip( rects, locs ) ):

                rect, loc = rloc

                if loc:

                    for iidx in range( len( zcols ) ):

                        rects[ jidx + iidx*n_bars ].set_edgecolor( color )

    if alt_labels:

        labels = col_pivot.index.get_level_values( 'alt' )

        min_ht = min( neg_ht_sum )
        neg_ht_mean = np.mean( neg_ht_sum )

        for rect, label in zip( rects, labels ):
            ax.text( rect.get_x() + rect.get_width() / 2,
                     min_ht + .75*neg_ht_mean,
                     label,
                     fontsize = 16,
                     fontweight = 'bold',
                     ha='center',
                     va='bottom' )

    if bar_labels:

        if bar_labels_offset:

            if bar_label_loc == 'middle':

                pos_ht_mean = np.mean( pos_ht_sum )

                max_rect_ht = y_ax_lim[ 1 ] if y_ax_lim else ax.get_ylim()[ 1 ]

                mid_rect_ht = max_rect_ht / 2

                mid_label = int( len( bar_labels ) / 2 )

                #sets up constant y label positions
                #first label is the lowest and they are centered in the middle of the positive y axis
                label_pos = [ mid_rect_ht + ( i - ( mid_label ) )*.2*max_rect_ht for i in range( len( bar_labels ) ) ]

            elif bar_label_loc == 'above':

                label_pos = [ 1 + .1*i for i in range( len( bar_labels ) ) ]

        else:

            if bar_label_loc == 'middle':

                pos_ht_mean = np.mean( pos_ht_sum )

                max_rect_ht = y_ax_lim[ 1 ] if y_ax_lim else ax.get_ylim()[ 1 ]

                mid_rect_ht = max_rect_ht / 2

                #sets up constant y label positions
                #they are centered in the middle of the positive y axis
                label_pos = [ mid_rect_ht for i in range( len( bar_labels ) ) ]

            elif bar_label_loc == 'above':

                label_pos = [ 1.01 for i in range( len( bar_labels ) ) ]

        for iidx, colmark in enumerate( bar_labels ):

            col, marker, _label, fsize = colmark

            #true false vector for where to put the marker
            locs = tbv_filt[ col ]

            for jidx, rloc in enumerate( zip( rects, locs ) ):

                rect, loc = rloc

                if loc:

                    ax.text( rect.get_x() + rect.get_width() /2,
                             label_pos[ iidx ],
                             marker,
                             fontsize = fsize,
                             transform = trans,
                             ha = 'center',
                             va = 'bottom' )

    plt.title( title, fontsize = 24 )

    if alt_labels:

        col_pivot[ 'y_min' ] = col_pivot[ col_pivot < 0 ].sum( axis = 1 )
        col_pivot[ 'y_max' ] = col_pivot[ col_pivot > 0 ].sum( axis = 1 )

        y_max = y_ax_lim[ 1 ] if y_ax_lim else 1.01*col_pivot.y_max.max()
        y_ax_lim = ( col_pivot.y_min.min() + neg_ht_mean,
                     y_max )

    if y_ax_lim:
        plt.ylim( y_ax_lim )

    plt.ylabel( y_ax_title, fontsize = 36 )
    plt.yticks( fontsize = 36 )

    plt.xlabel( x_ax_title, fontsize = 36 )

    pos_pivot = tbv_filt.pivot( index = 'pos', columns = 'alt', values = pos_col )

    pos_pivot = pos_pivot.fillna( '' )

    pos_pivot[ pos_col ] = [ a if a != '' else c for a,c in zip( pos_pivot.A, pos_pivot.C ) ]

    plt.xticks( [ idx for idx,p in enumerate( col_pivot.index ) if idx % tick_spacing == 1 ],
                [ c for idx,c in enumerate( pos_pivot[ pos_col ] ) if idx % tick_spacing == 1 ],
                fontsize=36,
                rotation='vertical' )

    if hlines:
        for line in hlines:
            plt.axhline( line, c = 'black', ls = '--', alpha = .6 )

    if shade_by_base:

        for i in range( len( set( tbv_filt.pos ) ) ):

            #shade only every other position
            if i % 2 == 0:

                #three bases are non-missing at each position
                rect = rects[ 3*i ]

                plt.axvspan( rect.get_x(),
                             rect.get_x() + 3*rect.get_width(),
                             facecolor = 'gray',
                             alpha = 0.2 )

    if shade_exons:
        for ex in shade_exons:
            plt.axvspan( col_pivot.index.get_loc( ex[0] ).start - .5,
                         col_pivot.index.get_loc( ex[1] ).stop,
                         facecolor = 'gray',
                         alpha = 0.15 )

    if scale_bar:

        assert scale_bar_loc.lower() in [ 'left', 'middle', 'right' ], \
        'Scale bar location can be "left", "middle", or "right".'

        if scale_bar_loc == 'left':
            x_loc = 3*tick_spacing
        elif scale_bar_loc == 'middle':
            x_loc = tbv_filt.shape[0] / 2
        else:
            x_loc = tbv_filt.shape[0] - 3*tick_spacing

        trans = tfrms.blended_transform_factory( ax.transData,
                                                 ax.transAxes )
        plt.errorbar( x_loc,
                      0.96,
                      #xerr = ( 3*tick_spacing / 2 ),
                      xerr = ( 3*tick_spacing*rects[0].get_width() / 2 ),
                      color = 'black',
                      capsize=3,
                      transform=trans )

        txt = str( tick_spacing ) + ' bases' if tick_spacing > 1 else str( tick_spacing ) + ' base'
        plt.text( x_loc,
                  0.94,
                  txt,
                  horizontalalignment = 'center',
                  verticalalignment = 'top',
                  transform = trans,
                  fontsize = 14 )

    if legend:

        if legend_labels:
            legend = plt.legend( title = legend_title,
                                 bbox_to_anchor = ( 1, 1 ), #if you turn that on you can put legend outside of plot
                                 loc = legend_loc,
                                 labels = legend_labels,
                                 fontsize = 14 )
        else:
            legend = plt.legend( title = legend_title,
                                 bbox_to_anchor = ( 1, 1 ), #if you turn that on you can put legend outside of plot
                                 loc = legend_loc,
                                 fontsize = 14 )
        plt.setp( legend.get_title(), fontsize=14 )
    elif labels_legend:

        fake_handles = [ plt.Rectangle( (0, 0), 0, 0, fill=False, edgecolor='none', visible = 'False')
                        for i in range( len( bar_labels ) ) ]

        legend = plt.legend( handles = fake_handles,
                             title = labels_legend_title,
                             bbox_to_anchor = ( 0, 1 ), #if you turn that on you can put legend outside of plot
                             loc = labels_legend_loc,
                             labels = ( symbol + ' '*6 + label for _col, symbol, label, _fsize in bar_labels ),
                             fontsize = 14 )
        plt.setp( legend.get_title(), fontsize=14 )
    else:
        ax.legend_ = None
        plt.draw()

    if tight:
        plt.tight_layout()

    if savefile:
        plt.savefig( savefile,
                     #dpi = 300,
                     #bbox_inches = 'tight'
                   )

    plt.show()

def swarm_plot( tblbyvar,
                x_col,
                y_col,
                x_ax_label = '',
                x_tick_labels = None,
                y_ax_label = '',
                hlines = None,
                savefile = None,
                **kwargs ):

    sns.set_style( 'ticks' )

    tbv = tblbyvar.copy()

    ax = sns.swarmplot( x = x_col,
                        y = y_col,
                        data = tbv,
                        **kwargs )

    ax.set_xlabel( x_ax_label,
                   fontsize = 18 )

    ax.set_ylabel( y_ax_label,
                   fontsize = 18 )

    if x_tick_labels:
        ax.set_xticklabels( x_tick_labels,
                            fontsize = 14 )

    plt.yticks( fontsize = 14 )

    if hlines:
        for hl in hlines:
            ax.axhline( hl, color = 'black', linestyle = '--' )

    if savefile:
        plt.savefig( savefile )

    plt.show()

def violin( tbl_byvar,
            cat_col,
            val_col,
            ylim = None,
            x_ax_label = '',
            y_ax_label = '',
            x_tick_labels = None,
            savefig = None,
            **kwargs ):

    tbv = tbl_byvar.copy()

    ax = sns.violinplot( x = cat_col,
                        y = val_col,
                        data = tbv,
                        **kwargs
                         )

    if ylim:
        ax.set_ylim( ylim )

    ax.set_xlabel( x_ax_label,
                   fontsize = 18 )

    ax.set_ylabel( y_ax_label,
                   fontsize = 18 )

    if x_tick_labels:
        ax.set_xticklabels( x_tick_labels,
                            fontsize = 14 )

    plt.yticks( fontsize = 14 )

    if savefig:
        plt.savefig( savefig,
                     bbox_to_inches = 'tight' )

    plt.show()

def upset_plot( byvartbl,
                cat_cols,
                fig_size = ( 10, 20 ),
                dist_cols = None,
                dist_cols_kind = None,
                dist_cols_col = None,
                violin_inner = None,
                savefig = None,
                **kwargs ):

    tbv = byvartbl.set_index( cat_cols ).copy()

    up_df = up.UpSet( tbv, **kwargs )

    inner = {  }


    if dist_cols:

        for i, col in enumerate( dist_cols ):

            if dist_cols_kind[ i ] == 'violin':

                up_df.add_catplot( value = col,
                                   kind = dist_cols_kind[ i ],
                                   color = dist_cols_col[ i ],
                                   inner = 'points',
                                   #jitter = True,
                                   #s = 1.5,
                                   elements = 3,
                                   cut = 0,
                                   scale = 'width',
                                   #linewidth = 1,
                                   width = .5,
                                   #bw = 2
                                 )
            else:

                up_df.add_catplot( value = col,
                                   kind = dist_cols_kind[ i ],
                                   color = dist_cols_col[ i ],
                                   #s = 3,
                                   elements = 2,
                                   #linewidth = 0,
                                 )

    #fig1 = plt.figure( figsize = fig_size )

    fig1 = plt.gcf()

    fig1.set_size_inches(20, 40)

    up_df.plot( fig = fig1 )

    #g.fig.set_figwidth(20)
    #g.fig.set_figheight(20)

    if savefig:

        plt.savefig( savefig,
                     bbox_inches = 'tight' )

    plt.show()

def sdv_waterfall_plot(
                    vartbl,
                    waterfall_col,
                    percentile = None,
                    title = '',
                    x_ax_title = '',
                    y_ax_title = '',
                    ):

    ranks = vartbl.copy()

    ranks = ranks.sort_values( by = [ waterfall_col ],
                               ascending = False )

    ranks[ waterfall_col + '_rank' ] = np.arange( ranks.shape[0] )

    fig, ax1 = plt.subplots()

    color = 'blue'
    ax1.set_xlabel( x_ax_title )
    ax1.set_ylabel( y_ax_title, color=color )
    ax1.plot( ranks[ waterfall_col + '_rank' ],
              ranks[ waterfall_col ], color=color)

    if percentile:
        percentile_cutoff = ranks.iloc[ int( ranks.shape[0]*percentile*0.01 ) ][ waterfall_col ]
        plt.axvline( int( ranks.shape[0]*percentile*0.01 ),
                     color = 'black' )
        print( ' To call %d%% of variants as SDV, use the threshold %.2f ' % ( percentile, percentile_cutoff ) )

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def waterfall_by_sdv( measured_tbl,
                      score_col,
                      sdv_col, ):

    mdf = measured_tbl.loc[ ~measured_tbl[ score_col ].isnull() ].copy()

    if len( mdf ) != len( measured_tbl ):

        print( '%i values missing in %s' % ( len( measured_tbl ) - len( mdf ),
                                             score_col ) )

    sort_sdv = { val: mdf.loc[ mdf[ sdv_col ] == val ].sort_values( by = score_col,
                                                                    ascending = False ).copy()
                 for val in mdf[ sdv_col ].unique() }

    for val,tbl in sort_sdv.items():

        plt.plot( np.arange( len( tbl ) ) / len( tbl ),
                  tbl[ score_col ],
                  label = '%s ( n = %i )' % ( val, len( tbl ) ) )

    plt.xlabel( 'Rank' )
    plt.ylabel( 'Quantile from random VCF' )

    plt.ylim( ( 0, 100 ) )
    plt.xlim( ( 0, 1 ) )

    plt.legend()

    plt.show()

def waterfall_by_sdv_subplot( measured_tbl,
                              tool_names,
                              categories,
                              sdv_col,
                              figsize = ( 20, 20 ),
                              sharex = True,
                              sharey = True,
                              savefile = None ):

    mdf = measured_tbl.sort_values( sdv_col ).copy()

    fig, ax = plt.subplots( len( tool_names ),
                            len( categories ),
                            figsize = figsize,
                            sharex = sharex,
                            sharey = sharey )

    max_dist = np.empty( ( len( tool_names ), len( categories ) ) )
    max_dist[:] = np.nan

    max_area = np.empty( ( len( tool_names ), len( categories ) ) )
    max_area[:] = np.nan

    for i, tool in enumerate( tool_names ):

        ax[ i ][ 0 ].set_ylabel( '%s random VCF quantile' % tool )

        for j, cat in enumerate( categories ):

            score_col = '_'.join( [ tool.lower(), cat.lower().replace( ' ', '_' ), 'per' ] )

            if score_col not in mdf.columns:

                ax[ i ][ j ].xaxis.set_visible( False )
                plt.setp( ax[ i ][ j ].spines.values(), visible = False )

                if j != 0:
                    ax[ i ][ j ].tick_params( left = False, labelleft = False )
                #ax[ i ][ j ].patch.set_visible( False )
                continue

            mij = mdf.loc[ mdf[ score_col ].notnull() ].copy()

            if len( mij ) != len( mdf ):

                print( '%i values missing in %s' % ( len( mdf ) - len( mij ), score_col ) )

            if len( mij ) == 0 or len( mij[ sdv_col ].unique() ) < 2:

                print( 'Unable to plot %s for tool %s' % ( cat, tool ) )
                ax[ i ][ j ].xaxis.set_visible( False )
                plt.setp( ax[ i ][ j ].spines.values(), visible = False )

                if j != 0:
                    ax[ i ][ j ].tick_params( left = False, labelleft = False )
                continue

            if any( [ ( mij[ sdv_col ] == sdv_cat ).sum() < 2 for sdv_cat in mij[ sdv_col ].unique() ] ):

                print( 'Less than two values per event for category %s and tool %s' % ( cat, tool ) )
                ax[ i ][ j ].xaxis.set_visible( False )
                plt.setp( ax[ i ][ j ].spines.values(), visible = False )

                if j != 0:
                    ax[ i ][ j ].tick_params( left = False, labelleft = False )
                continue

            sort_sdv = { val: mij.loc[ mij[ sdv_col ] == val ].sort_values( by = score_col,
                                                                            ascending = False ).copy()
                         for val in mij[ sdv_col ].unique() }

            xvals = { val: np.arange( ( mij[ sdv_col ] == val ).sum() ) / ( ( mij[ sdv_col ] == val ).sum() - 1 )
                      for val in mij[ sdv_col ].unique() }

            max_dist[ i ][ j ] = waterfall_dist( [ x for x in xvals.values() ],
                                                 [ np.array( y[ score_col ].tolist() )
                                                   for y in sort_sdv.values() ] )

            area = { val: simps( tbl[ score_col ].to_numpy(),
                                 xvals[ val ] )
                     for val,tbl in sort_sdv.items() }

            for val,tbl in sort_sdv.items():

                ax[ i ][ j ].plot( xvals[ val ],
                                  tbl[ score_col ],
                                  label = '%s ( n = %i )' % ( val, len( tbl ) ) )

                if val == 'SDV' or val == 'LOF':
                    sdv_area = area[ val ]

                elif val == 'Neutral' or val == 'FUNC':
                    neut_area = area[ val ]

            max_area[ i ][ j ] = sdv_area - neut_area

            ax[ i ][ j ].set_xlabel( 'Rank' )
            ax[ 0 ][ j ].set_title( cat )

            ax[ i ][ j ].set_ylim( ( 0, 100 ) )
            ax[ i ][ j ].set_xlim( ( 0, 1 ) )

            ax[ i ][ j ].legend()

    if savefile:
        plt.savefig( savefile,
                     dpi = 300,
                     #bbox_inches = 'tight'
                   )

    plt.show()

    dist_df = pd.DataFrame( max_dist,
                             columns = categories,
                             index = tool_names )

    area_df = pd.DataFrame( max_area,
                             columns = categories,
                             index = tool_names )

    return ( dist_df, area_df )

def waterfall_auc_scatter( wf_dist,
                            categories,
                            #cmap,
                            figsize = ( 15, 10 ),
                            sharex = True,
                            sharey = True,
                            **kwargs ):

    fig, ax = plt.subplots( 2,
                            int( np.ceil( len( categories ) / 2 ) ),
                            figsize = figsize,
                            sharex = sharex,
                            sharey = sharey )

    for idx,cat in enumerate( categories ):

        i,j = int( idx // ( np.ceil( len( categories ) / 2 ) ) ), int( idx % ( np.ceil( len( categories ) / 2 ) ) )

        for k,name_dist in enumerate( wf_dist.items() ):

            name,dist = name_dist

            ax[ i ][ j ].scatter( dist.index,
                                  dist[ cat ],
                                  #cmap( k ),
                                  label = name,
                                  **kwargs )

        ax[ i ][ j ].set_title( cat )

        ax[ i ][ j ].set_xticklabels( dist.index, rotation= 45 )
        ax[ i ][ 0 ].set_ylabel( 'Area between SDV and neutral' )

    ax[ 0 ][ 2 ].legend( loc = 'center right',
                         bbox_to_anchor = ( 1.35, .5 ) )

    plt.tight_layout()

    plt.show()

def barplot_per_repeat( table_by_var,
                        sdv_col,
                        nsamp,
                        ylim = None,
                        ylabel = '',
                        xlabel = '',
                        savefig = False,
                        **kwargs ):

    tbv = table_by_var.copy()

    vals = tbv[ sdv_col ].value_counts()

    total = vals.sum()

    xvals = [ i for i in range( 1, nsamp + 1 ) ]

    yvals = []
    for x in xvals:

        if x in vals:
            yvals.append( 100*( int( vals[ x ] ) / total ) )
        else:
            yvals.append( 0 )

    plt.bar( xvals, yvals, **kwargs )

    ax = plt.gca()

    rects = ax.patches

    trans = tfrms.blended_transform_factory( ax.transData, ax.transData )

    for rect, y in zip( rects, yvals ):

        if y > 0:

            ax.text( rect.get_x() + rect.get_width() /2,
                     rect.get_height(),
                     str( round( ( y / 100 )*total ) ),
                     fontsize = 16,
                     transform = trans,
                     ha = 'center',
                     va = 'bottom' )

    if ylim:
        plt.ylim( ylim )

    plt.ylabel( ylabel, fontsize = 18 )
    plt.yticks( fontsize = 18 )

    plt.xlabel( xlabel, fontsize = 18 )
    plt.xticks( fontsize = 18 )

    plt.tight_layout()

    if savefig:
        plt.savefig( savefig,
                     bbox_inches = 'tight' )

    plt.show()

def pr_curves_bytruth( var_df,
                     truth_cols,
                     pred_col,
                     colors,
                   fig_size = ( 5, 5 ),
                   grid = False,
                   x_ax_label = 'Recall\n(%)',
                   y_ax_label = 'Precision\n(%)',
                   add_point = False,
                   savefile = None,
                   **kwargs
             ):

    #pr curve function hates missing values
    tbv = var_df.dropna(subset = [ pred_col ] ).copy()

    if tbv.shape[ 0 ] != var_df.shape[ 0 ]:
        print( 'Missing values in predictor column.', str( var_df.shape[ 0 ] - tbv.shape[ 0 ] ), 'rows removed.' )

    plt.figure( figsize = fig_size )

    for truth, color in zip( truth_cols, colors ):

        #pr curve function hates missing values
        vdf = tbv.dropna(subset = [ truth ] ).copy()

        if vdf.shape[ 0 ] != tbv.shape[ 0 ]:
            print( 'Missing values in truth column.', str( tbv.shape[ 0 ] - vdf.shape[ 0 ] ), 'rows removed.' )

        precision, recall, _ = precision_recall_curve( vdf[ truth ],
                                                       vdf[ pred_col ] )

        plt.plot( 100*recall,
                  100*precision,
                  color = color,
                  **kwargs )

        print( truth, auc( recall, precision ) )

        if add_point:
            plt.plot( add_point[ truth ][ 0 ],
                      add_point[ truth ][ 1 ],
                      color = 'black',
                      marker = add_point[ truth ][ 2 ],
                      markersize = add_point[ truth ][ 3 ]
                    )

    plt.xlabel( x_ax_label, fontsize = 24 )
    plt.xticks( fontsize = 20 )

    plt.ylabel( y_ax_label, fontsize = 24 )
    plt.yticks( fontsize = 20 )

    plt.grid( grid )

    if savefile:
        plt.savefig( savefile,
                     dpi = 300,
                     bbox_inches = 'tight' )

    plt.show()

def plot_stacked_bar(     var_df,
                          ycols,
                          xcol,
                          sort_cols = [ 'pos', 'alt' ],
                          title = '',
                          y_ax_title='',
                          y_ax_lim = None,
                          x_ax_title = '',
                          legend = True,
                          legend_title = '',
                          legend_loc = 'best',
                          legend_labels = None,
                          tight = True,
                          hlines = None,
                          savefile = None,
                          **kwargs ):

    tbv = var_df.sort_values( by = sort_cols ).set_index( xcol ).copy()

    tbv[ ycols ].plot.bar( stacked = True,
                            **kwargs )

    plt.title( title, fontsize = 24 )

    if y_ax_lim:
        plt.ylim( y_ax_lim )

    plt.ylabel( y_ax_title, fontsize = 20 )
    plt.yticks( fontsize = 18 )

    plt.xlabel( x_ax_title, fontsize = 18 )
    plt.xticks( fontsize = 15 )

    if hlines:
        for line in hlines:
            plt.axhline( line, c = 'black', ls = '--', alpha = .6 )

    ax = plt.gca()

    if legend:

        legend = plt.legend( title = legend_title,
                                 bbox_to_anchor = ( 1, 1 ), #if you turn that on you can put legend outside of plot
                                 loc = legend_loc,
                                 fontsize = 14 )
        plt.setp( legend.get_title(), fontsize=14 )
    else:
        ax.legend_ = None
        plt.draw()

    if tight:
        plt.tight_layout()

    if savefile:
        plt.savefig( savefile, )

    plt.show()

def scatter_truth_pred( tbl_by_var,
                        measured_col,
                        pred_cols,
                        color_bcols,
                        cmap,
                        figsize = ( 10, 20 ),
                        sharex = True,
                        sharey = 'row',
                        xlim = None,
                        **kwargs ):

    tbv = tbl_by_var.copy()

    fig, ax = plt.subplots( len( pred_cols ),
                            len( color_bcols ),
                            figsize = figsize,
                            sharex = sharex,
                            sharey = sharey )

    for cidx, ccol in enumerate( color_bcols ):

        ax[ 0 ][ cidx ].set_title( ccol )

        for pidx, pcol in enumerate( pred_cols ):

            colors = [ 'gray' if not c else cmap( pidx ) for c in tbv[ ccol ] ]

            ax[ pidx ][ cidx ].scatter( tbv[ measured_col ],
                                        tbv[ pcol ],
                                        c = colors,
                                        **kwargs )

            ax[ pidx ][ 0 ].set_ylabel( pcol )

            if xlim:
                ax[ pidx ][ cidx ].set_xlim( xlim )

    mid = int( cidx / 2 )

    ax[ len( pred_cols ) - 1 ][ mid ].set_xlabel( measured_col )

    plt.show()

def scatter_by_sdv_alt_interp( tbl_by_var,
                               interp_col,
                               x_col,
                               y_col,
                               cmap,
                               marker_by_interp,
                               alt_col = 'alt',
                               sdv_col = 'sdv',
                               intmed_col = None,
                               intmed_cmap = None,
                               null_color = '.85',
                               marker_size = 5,
                               xlabel = '',
                               ylabel = '',
                               xlim = None,
                               ylim = None,
                               savefig = None ):

    tbv = tbl_by_var.copy()

    #we want the SDV variants on top so loop through twice for maximum control
    for interp in tbv[ interp_col ].unique():

        lit_null = tbv.loc[ ~( tbv[ sdv_col ] ) & ( tbv[ interp_col ] == interp ) ].copy()

        if len( lit_null ) > 0:

            plt.scatter( lit_null[ x_col ],
                         lit_null[ y_col ],
                         marker = marker_by_interp[ interp ],
                         c = null_color,
                         s = marker_size )

    if intmed_col:

        for interp in tbv[ interp_col ].unique():

            lit_intmed = tbv.loc[ ( tbv[ intmed_col ] ) & ( tbv[ interp_col ] == interp ) ].copy()

            if len( lit_intmed ) > 0:

                plt.scatter( lit_intmed[ x_col ],
                             lit_intmed[ y_col ],
                             marker = marker_by_interp[ interp ],
                             c = [ 0 if a.upper() == 'A' \
                                   else 1 if a.upper() == 'C' \
                                   else 2 if a.upper() == 'G' \
                                   else 3 for a in lit_intmed[ alt_col ] ],
                             cmap = intmed_cmap,
                             s = 2*marker_size )

    for interp in tbv[ interp_col ].unique():

        lit_sdv = tbv.loc[ ( tbv[ sdv_col ] ) & ( tbv[ interp_col ] == interp ) ].copy()

        if len( lit_sdv ) > 0:

            plt.scatter( lit_sdv[ x_col ],
                         lit_sdv[ y_col ],
                         marker = marker_by_interp[ interp ],
                         c = [ 0 if a.upper() == 'A' \
                               else 1 if a.upper() == 'C' \
                               else 2 if a.upper() == 'G' \
                               else 3 for a in lit_sdv[ alt_col ] ],
                         cmap = cmap,
                         s = 2*marker_size )

    plt.xlabel( xlabel )
    plt.ylabel( ylabel )

    if xlim:
        plt.xlim( xlim )
    if ylim:
        plt.ylim( ylim )

    if savefig:
        plt.savefig( savefig,
                     dpi = 300,
                     bbox_inches = 'tight' )

    plt.show()

def scatter_by_interp( tbl_by_var,
                               interp_col,
                               x_col,
                               y_col,
                               marker_by_interp,
                               cmap,
                               null_color = '.85',
                               marker_size = 5,
                               xlabel = '',
                               ylabel = '',
                               xlim = None,
                               ylim = None,
                               savefig = None ):

    tbv = tbl_by_var.copy()

    lit_null = tbv.loc[ ( tbv[ interp_col ].isnull() ) | ( tbv[ interp_col ] == '' ) ].copy()

    if len( lit_null ) > 0:

        plt.scatter( lit_null[ x_col ],
                     lit_null[ y_col ],
                     marker = marker_by_interp[ '' ],
                     c = null_color,
                     s = marker_size )

    #we want the lit variants on top so plot these second for maximum control
    for i,interp in enumerate( tbv[ interp_col ].unique() ):

        #I want to allow the interp to be null here instead of just '' but its so difficult w strings
        if interp == '':
            continue

        lit_interp = tbv.loc[ ( tbv[ interp_col ] == interp ) ].copy()

        if len( lit_interp ) > 0:

            plt.scatter( lit_interp[ x_col ],
                         lit_interp[ y_col ],
                         marker = marker_by_interp[ interp ],
                         color = cmap( i - 1 ),
                         s = 2*marker_size )

    plt.xlabel( xlabel )
    plt.ylabel( ylabel )

    if xlim:
        plt.xlim( xlim )
    if ylim:
        plt.ylim( ylim )

    if savefig:
        plt.savefig( savefig,
                     dpi = 300,
                     bbox_inches = 'tight' )

    plt.show()

def plot_iso_stats( iso_df_stats,
                    sa_col_stem = '_sum_sa_reads',
                    read_col_stem = '_read_count',
                    iso_rows = [ 'secondary', 'unmapped', 'unpaired', 'bad_starts', 'bad_ends', 'soft_clipped' ] ):

    iso_df = iso_df_stats.copy()

    plot_dfs = [ iso_df[ [ col for col in iso_df if read_col_stem in col and 'total' not in col ] ].sum() ]
    plot_dfs[ -1 ].index = [ idx.replace( read_col_stem, '' ) for idx in plot_dfs[ -1 ].index ]

    plot_dfs.append( iso_df[ [ col for col in iso_df if sa_col_stem in col and 'total' not in col ] ].sum() )
    plot_dfs[ -1 ].index = [ idx.replace( sa_col_stem, '' ) for idx in plot_dfs[ -1 ].index ]

    used_isos = []
    for iso in iso_rows:

        if iso in iso_df.isoform.tolist():

            used_isos.append( iso )
            plot_dfs.append( iso_df.loc[ iso_df.isoform == iso ][ [ col for col in iso_df if read_col_stem in col and 'total' not in col ] ].T )
            plot_dfs[ -1 ].index = [ idx.replace( read_col_stem, '' ) for idx in plot_dfs[ -1 ].index ]

    all_cnts = pd.concat( plot_dfs,
                          axis = 1 )

    all_cnts.columns = [ 'total_reads', 'sa_reads' ] + [ 'total_' + str( iso ) for iso in used_isos ]

    print( 'Raw counts\n' )

    for col in all_cnts:

        print( col )

        all_cnts[ col ].plot.bar()

        plt.ylabel( col )

        plt.show()

    print( 'Percentages\n' )
    for col in all_cnts:

        if col == 'total_reads':
            continue

        print( col )

        all_cnts[ 'per_' + col ] = all_cnts[ col ] / all_cnts.total_reads

        all_cnts[ 'per_' + col ].plot.bar()

        plt.ylim( ( 0, 1 ) )

        plt.ylabel( col + ' percent of total' )

        plt.show()

    all_cnts.index.name = 'sample'

    all_cnts = all_cnts.reset_index()

    return all_cnts

def plot_waterfall_bysamp( cutoff_df,
                           x_samp_col = 'sample',
                           y_cols = [ '_x_log10', '_y' ],
                           cutoffs = ( 75, 95 ),
                           ylabels = [ 'Log10 BC Rank', '# of reads/BC' ] ):

    cut = cutoff_df.copy()

    assert len( y_cols ) == len( ylabels ), 'You did not provide enough y axis labels to match your y columns'

    fig,ax = plt.subplots( len( y_cols ),
                           len( cutoffs ),
                           sharey = 'row',
                           sharex = True,
                           figsize = ( 12, 10 ) )

    for i,y in enumerate( y_cols ):

        for j,val in enumerate( cutoffs ):

            ax[ i ][ j ].scatter( cut[ x_samp_col ],
                                  cut[ str( val ) + y ] )

            if j == 0:

                ax[ i ][ j ].set_ylabel( ylabels[ i ] )

            if i == 0:

                ax[ i ][ j ].set_title( '%ith Percentile' % val )

            if i == len( y_cols ) - 1:

                ax[ i ][ j ].tick_params( axis='x', labelrotation = 90 )

    plt.show()

def plot_clinvar_by_interp( tbl_by_var,
                            yaxis_cols,
                            markers,
                            colors,
                            interp_col = 'lit_interp',
                            sort_col = 'pos',
                            plot_pos_col = 'hgvs_var',
                            figsize = ( 8, 3 ),
                            sharex = 'col',
                            sharey = 'row',
                            marker_size = 100,
                            row_label = False,
                            col_label = False, ):

    tbv = tbl_by_var.loc[ tbl_by_var[ interp_col ] != '' ].copy()

    assert len( markers ) == len( tbv[ interp_col ].unique() ), \
    'Number of markers does not match the number of unique values for interpretation column!'

    fig,ax = plt.subplots( len( yaxis_cols ), len( tbv[ interp_col ].unique() ),
                           gridspec_kw = { 'width_ratios' : [ len( tbv.loc[ tbv[ interp_col ] == l ] ) for l in tbv[ interp_col ].unique() ] },
                           sharex = sharex,
                           sharey = sharey,
                           figsize = figsize )

    for i,interp in enumerate( tbv[ interp_col ].unique() ):

        interp_df = tbv.loc[ tbv[ interp_col ] == interp ].sort_values( by = sort_col ).reset_index().copy()

        if col_label:

            ax[ 0 ][ i ].set_title( interp )

        for j,ycol in enumerate( yaxis_cols ):

            ax[ j ][ i ].scatter( interp_df.index,
                                  interp_df[ ycol ],
                                  marker = markers[ i ],
                                  color = colors( i ),
                                  s = marker_size )

            if row_label:

                ax[ j ][ 0 ].set_ylabel( ycol )

            #my stupid markers keep getting cut off
            x_l,x_r = ax[ j ][ i ].get_xlim()

            ax[ j ][ i ].set_xlim( ( x_l - .5, x_r + .5 ) )

        ax[ j ][ i ].set_xticks(  interp_df.index )

        ax[ j ][ i ].set_xticklabels( interp_df[ plot_pos_col ],
                                      fontsize=12,
                                      rotation='vertical' )

    plt.show()
