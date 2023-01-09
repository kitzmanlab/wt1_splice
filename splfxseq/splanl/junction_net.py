import pysam
import os
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt


# In[2]:


def ImportAnnot(bedfile):
    """Opens a bed file with columns: chromosome, start position, end position, gene name and returns a pandas
    dataframe.

    Args:
        bedfile (str): /path/and/name/to.bed 
            (file must be unzipped and positions are assumed to be 0-coordinate based)
            (NOTE: plasmid vector coordinates from serial cloner are 1-coordinate based)

    Returns:
        annotation dataframe (pandas dataframe): pandas dataframe with columns: chrom, start, end, gene
        
    Usage:
        annotdf = ImportAnnot('/path/and/name/to.bed')
    """
    annotdf=pd.read_csv(bedfile, sep='\t',header=None)
    annotdf.columns=['chrom','start','end','gene']
    return (annotdf)


# In[84]:


def DefineJunctions(annotdf,start,readLength):
    """Uses a fully-specified (all exons and cryptic exons listed) gene annotation pandas dataframe file with 
    columns: chromosome, start position, and end position along with information about the read start site and the
    read length to generate a list of possible and visible isoforms.

    Args:
        annotdf (pandas dataframe): columns - chromosome, start position (1-coordinate based), 
                                    end position (1-coordinate based)
        start (int): read start position (1-coordinate based) for fixed RNA-seq dataset
        readLength (int): length of reads within RNA-seq data
        
    Prints: All possible isoforms from the pandas dataframe. Isoforms are numbered which will indicate which read
    matches which isoform in the final dataset. Also prints a list of isoforms which will be indistinguishable 
    from each other at the given read length.

    Returns:
        visible isoforms (dictionary): isoformN: [visible, junctions, in, isoform]
                Each isoform contains the positions of junctions that will exist in the given isoform. Only 
                junctions which are reachable given the read length are shown. This will be used to match reads
                to isoforms in downstream analyses.
        
    Usage:
        visibleIsoforms = DefineJunctions(annotdf,649,150)
    """
    starts=annotdf.start.tolist()
    ends=annotdf.end.tolist()
    possiblePaths=DefineGraph(start,list(set(starts+ends)))
    reasonablePaths=TruncatePaths(possiblePaths,start,starts,ends)
    visibleIsoforms=CreateIsoformDict(reasonablePaths,readLength,starts,ends)
    PrintCrypticIsoforms(visibleIsoforms)
    return(visibleIsoforms)


# In[4]:


def DefineGraph(start,coords):
    """Given a start site for the reads and lists of all possible start and end positions for the exons within 
    the plasmid vector, this function creates all possible directed paths through the nodes (start and end 
    positions)

    Args:
        start (int): read start position (1-coordinate based) for fixed RNA-seq dataset 
        coords (list of ints): all start and end sites of exons and cryptic exons (1-coordinate based) (nodes of
        graph)

    Returns:
        all possible paths (list of lists of ints): list containing lists which indicate possible directed paths 
        through the start and end sites of the exons (nodes of the graph) beginning at the read start site.
        NOTE: This list will not include any start or end junction which begins before the read start site.
        NOTE: This list is the permutations of paths and is not necessarily biologically sensible.
        
    Usage:
        possiblePaths = DefineGraph(649,[347,697,1304,1454,3742,3856])
        print(possiblePaths)
        [[649, 697, 1304, 1454, 3742, 3856], [649, 697, 1304, 1454, 3856], [649, 697, 1304, 3742, 3856], 
        [649, 697, 1304, 3856], [649, 697, 1454, 3742, 3856], [649, 697, 1454, 3856], [649, 697, 3742, 3856], 
        [649, 697, 3856], [649, 1304, 1454, 3742, 3856], [649, 1304, 1454, 3856], [649, 1304, 3742, 3856], 
        [649, 1304, 3856], [649, 1454, 3742, 3856], [649, 1454, 3856], [649, 3742, 3856], [649, 3856]]
    """
    for c in coords:
        if c<start:
            coords.remove(c)
    coords.append(start)
    coords.sort()
    G=nx.complete_graph(coords)
    GDi=nx.DiGraph()
    GDi.add_nodes_from(G)
    GDi.add_edges_from(G.edges)
    return([path for path in nx.all_simple_paths(GDi, source=start, target=coords[-1])])


# In[5]:


def TruncatePaths(possiblePaths,start,starts,ends):
    """Given all possible directed paths from the start and end junctions of the exons, the start site for the 
    reads and lists of all possible start and end positions for the exons within the plasmid vector, this function
    truncates all possible paths to return only biologically relevant paths.

    Args:
        possiblePaths (list of lists of ints): list containing lists which indicate possible directed paths 
        through the start and end sites of the exons (nodes of the graph) beginning at the read start site.
        start (int): read start position (1-coordinate based) for fixed RNA-seq dataset 
        starts (list of ints): all start sites of exons and cryptic exons (1-coordinate based) 
        ends (list of ints): all end sites of exons and cryptic exons (1-coordinate based)
        
    Returns:
        reasonable paths (list of lists of ints): list containing lists which indicate biologically possible 
        directed paths through the start and end sites of the exons (nodes of the graph) beginning at the read 
        start site.
        NOTE: This list will not include any start or end junction which begins before the read start site.
        
    Usage:
        possiblePaths=[[649, 697, 1304, 1454, 3742, 3856], [649, 697, 1304, 1454, 3856], [649, 697, 1304, 3742, 3856], 
        [649, 697, 1304, 3856], [649, 697, 1454, 3742, 3856], [649, 697, 1454, 3856], [649, 697, 3742, 3856], 
        [649, 697, 3856], [649, 1304, 1454, 3742, 3856], [649, 1304, 1454, 3856], [649, 1304, 3742, 3856], 
        [649, 1304, 3856], [649, 1454, 3742, 3856], [649, 1454, 3856], [649, 3742, 3856], [649, 3856]]
        reasonablePaths = TruncatePaths(possiblePaths,649,[347,1304,3742],[697,1454,3856])
        print(reasonablePaths)
        [[649, 697, 1304, 1454, 3742, 3856], [649, 697, 3742, 3856]]
    """
    removeList=[]
    for path in possiblePaths:
        if len(path)<3:
            removeList.append(path)
            continue
        prevNode=None
        startIndex=None
        for node in path:
            if node in ends:
                endIndex={i for i,e in enumerate(ends) if e==node}
            if node==start:
                prevNode='s'
                startIndex={max({i for i,s in enumerate(starts) if s<=start})}
                continue
            #can't have two start sites ajacent to each other
            elif node in starts and prevNode=='s':
                removeList.append(path)
                break
            #can't have two end sites ajacent to each other
            elif node in ends and prevNode=='e':
                removeList.append(path)
                break
            #end must be from the same exon as start - can't encompass two full exons with only two junctions
            elif startIndex and node in ends and len(startIndex.intersection(endIndex))==0:
                removeList.append(path)
                break
            elif node in starts:
                prevNode='s'
                startIndex={i for i,s in enumerate(starts) if s==node}
            elif node in ends:
                prevNode='e'
    for path in removeList:
        possiblePaths.remove(path)
    return(possiblePaths)


# In[104]:


def IdentifyTruncations(starts,ends):
    """Identifies truncated exon forms so the final visible paths can included exonic nodes that will exist but 
    do not define that isoform.

    Args:
        starts (list of ints): all start sites of exons and cryptic exons (1-coordinate based) 
        ends (list of ints): all end sites of exons and cryptic exons (1-coordinate based)

    Returns:
        truncEnds,truncStarts (tuple of dictionaries): 
            truncEnds (dictionary - startPos:endPos) - dictionary with keys as the start sites of 5' truncated exon
            and the values the end site of the 5' truncated exon
            truncStarts (dictionary - endPos:startPos) - dictionary with keys as the end sites of 3' truncated exon
            and the values the start site of the 3' truncated exon
        
    Usage:
        truncEnds,truncStarts=IdentifyTruncations([347,1304,1304,3742],[697,1372,1454,3856])
        print(truncEnds)
        {1304:1372}
        print(truncStarts)
        {}
        truncEnds,truncStarts=IdentifyTruncations([347,1304,1354,3742],[697,1454,1454,3856])
        print(truncEnds)
        {}
        print(truncStarts)
        {1454:1354}
    """
    truncEnds={}
    if len(set(starts))!=len(starts):
        dupStarts=[start for start,count in Counter(starts).items() if count>1]
        for dup in dupStarts:
            truncEnds[dup]=min([ends[idx] for idx,start in enumerate(starts) if start==dup])
    truncStarts={}
    if len(set(ends))!=len(ends):
        dupEnds=[end for end,count in Counter(ends).items() if count>1]
        for dup in dupEnds:
            truncStarts[dup]=max([starts[idx] for idx,end in enumerate(ends) if end==dup])
    return (truncEnds,truncStarts)


# In[126]:


def CreateIsoformDict(reasonablePaths,readLength,starts,ends):
    """Given all reasonable directed paths from the start and end junctions of the exons and the read length, this 
    function assigns a number to each possible isoform (path) and prints out this result for the user. The function
    also computes which junctions can be contained within each read given the read length.

    Args:
        reasonablePaths (list of lists of ints): list containing lists which indicate junctions contained in
        each biologically relevant directed path
        readLength (int): length of reads within RNA-seq data
        
    Prints:
        numbered possible isoforms in dataset for inspection by the user and to match with final dataset
        isoform1: [each, junction, in, isoform1]
        isoform2: [each, junction, in, isoform2]
        ...
        isoformN: [each, junction, in, isoformN]
        
    Returns:
        visible isoforms (dictionary: string:list of ints): isoformN:[each, junction, in, isoformN, within, read]
        
    Usage:
        possiblePaths=[[649, 697, 1304, 1454, 3742, 3856], [649, 697, 3742, 3856]]
        visibleIsoforms = CreateIsoformDict(reasonablePaths,150)
        isoform0: [649, 697, 1304, 1454, 3742, 3856]
        isoform1: [649, 697, 3742, 3856]
        print(visibleIsoforms)
        {'isoform0': [649, 697, 1304], 'isoform1': [649, 697, 3742]}
    """
    pathDict={}
    for i,path in enumerate(reasonablePaths):
        isoform='isoform'+str(i)
        pathDict[isoform]=path
        print(isoform+': '+str(path))
    truncEnds,truncStarts=IdentifyTruncations(starts,ends)
    trueDict={}
    for iso,path in pathDict.items():
        truePath=[]
        basesUsed=0
        #loop by two's to only add exons and not intronic regions
        for i in range(0,len(path),2):
            if basesUsed<readLength:
                truePath.append(path[i])
            basesUsed+=path[i+1]-path[i]
            if basesUsed>readLength and path[i] in truncEnds:
                truncBases=basesUsed-(path[i+1]-truncEnds[path[i]])
                if truncBases<=readLength:
                    truePath.append(truncEnds[path[i]])
                break
            elif basesUsed>readLength and path[i+1] in truncStarts:
                truncBases=basesUsed-(path[i+1]-truncStarts[path[i+1]])
                if truncBases<readLength:
                    truePath.append(truncStarts[path[i+1]])
            elif basesUsed<=readLength:
                if path[i] in truncEnds and path[i+1]!=truncEnds[path[i]]:
                    truePath.append(truncEnds[path[i]])
                elif path[i+1] in truncStarts and path[i]!=truncStarts[path[i+1]]:
                    truePath.append(truncStartss[path[i+1]])
                truePath.append(path[i+1])
            else:
                break
        trueDict[iso]=truePath
    return(trueDict)


# In[7]:


def PrintCrypticIsoforms(visibleIsoforms):
    """Given a dictionary with all the junctions for each isoform that can be contained within a given read, this
    function prints out a warning if two isoforms are indiscernible from each other.

    Args:
        visible isoforms (dictionary: string:list of ints): isoformN:[each, junction, in, isoformN, within, read]
        
    Prints:
        WARNING: isoformi, isoformj are identical at given read length
        
    Returns:
        Nothing
        
    Usage:
        visibleIsoforms = {'isoform0': [649, 697, 1304], 'isoform1': [649, 697, 3742]}
        PrintCrypticIsoforms(visibleIsoforms)
        
        visibleIsoforms = {'isoform0': [649, 697, 1304], 'isoform1': [649, 697, 1304], 
        'isoform2': [649, 697, 3742],}
        PrintCrypticIsoforms(visibleIsoforms)
        WARNING: isoform1, isoform0 are identical at given read length
    """
    overlap={}
    for i,isoi in enumerate(list(visibleIsoforms)):
        for isoj in list(visibleIsoforms)[i+1:]:
            if visibleIsoforms[isoi]==visibleIsoforms[isoj]:
                if isoi in overlap:
                    overlap[isoi].append(isoj)
                else:
                    overlap[isoi]=[isoj]
    for key,value in overlap.items():
        value.append(key)
        print('WARNING: '+', '.join(value)+' are identical at given read length')


# In[8]:


def CreatePreProcessPlots(bamfile):
    """Saves a bar plot of read start positions in the same directory as the bam file and computes the most 
    frequent read start position to be used if the user does not input a read start position.
    NOTE: This function ignores unmapped reads.

    Args:
        bamfile (str): /path/and/name/to.bam 
            (file must be unzipped)
            
    Prints:
        If one start position takes up at least 80% of the reads:
            Percentage of reads starting at position x along with position x.
        Else:
            WARNING
            Percentage of reads starting at top three positions along with each position.

    Returns:
        most common read start position (int): The most common read start position in the dataset
        most common read length (int): The most common read length in the dataset
        
    Saves:
        pdf of the log10 frequencies of read starts by position
        
    Usage:
        readStart,readLength = CreatePreProcessPlots('/path/and/name/to.bam')
    """
    samFile=ImportBam(bamfile)
    startCounts,readLengthCounts=ComputeCounts(samFile)
    startSite,readLength=ComputeCountStats(startCounts,readLengthCounts)
    PlotStarts(startCounts,bamfile)
    return(startSite,readLength)


# In[9]:


def ImportBam(bamfile):
    """Opens a bam file as a pysam alignment file for reading

    Args:
        bamfile (str): /path/and/name/to.bam 
            (file must be unzipped)

    Returns:
        samFile (pysam generator object): pysam alignment file
        
    Usage:
        samFile = ImportBam('/path/and/name/to.bam')
    """
    samFile = pysam.AlignmentFile(bamfile, "rb")
    return(samFile)


# In[10]:


def ComputeCounts(samFile):
    """Counts mapped start sites and read lengths.

    Args:
        samFile (pysam generator object): pysam alignment file

    Returns:
        startCounts (Counter): counts of mapped reads starting at each position (0-based coordinates)
        readLengthCounts (Counter): counts of read lengths
        
    Usage:
        startCounts,readLengthCounts = ComputeCounts(samFile)
    """
    startCounts=Counter()
    readLengthCounts=Counter()
    for read in samFile:
        if not read.is_unmapped:
            startCounts[read.reference_start]+=1
            readLengthCounts[read.query_length]+=1
    samFile.close()
    return(startCounts,readLengthCounts)            


# In[11]:


def ComputeCountStats(startCounts,readLengthCounts):
    """Computes the three most common start sites, informs the user of the percentage of reads starting at each
    position, and returns the most common position.

    Args:
        startCounts (Counter): counts of mapped reads starting at each position (0-based coordinates)
        
    Prints:
        If the most common start site occurs in at least 80% of the reads:
            percent% of mapped reads start at position [x] (x is 1-based coordinates)
        Else:
            WARNING: Less than 80% of mapped reads start at any one position!
            Top three read start sites:
            percent% of mapped reads start at position [x] (x is 1-based coordinates)
            percent% of mapped reads start at position [y] (y is 1-based coordinates)
            percent% of mapped reads start at position [z] (z is 1-based coordinates)
            The top read start site will be assumed as correct if no read start site was specified as input.
        The function follows the same procedure for read lengths.

    Returns:
        mostCommonStart (int): the position where the most reads start (1-based coordinates)
        mostCommonReadLength (int): the most common read length in the dataset
        
    Usage:
        mostCommonStart,mostCommonReadLength = ComputeCountStats(startCounts)
    """
    startAndCount=startCounts.most_common(3)
    lengthAndCount=readLengthCounts.most_common(3)
    totalMappedReads=sum(startCounts.values())
    topStartPercent=100*(startAndCount[0][1]/totalMappedReads)
    topLengthPercent=100*(lengthAndCount[0][1]/totalMappedReads)
    if topStartPercent>=80:
        print('%.2f' % topStartPercent + '% of mapped reads start at position '+str(startAndCount[0][0]+1))
    else:
        print('WARNING: Less than 80% of mapped reads start at any one position!')
        print('Top read start sites:')
        secondStartPercent=100*(startAndCount[1][1]/totalMappedReads)
        print('%.2f' % topStartPercent + '% of mapped reads start at position '+str(startAndCount[0][0]+1))
        print('%.2f' % secondStartPercent + '% of mapped reads start at position '+str(startAndCount[1][0]+1))
        if len(startAndCount)==3:
            thirdStartPercent=100*(startAndCount[2][1]/totalMappedReads)
            print('%.2f' % thirdStartPercent + '% of mapped reads start at position '+str(startAndCount[2][0]+1))
        print('The top read start site will be assumed as correct if no read start site was specified as input.')
    if topLengthPercent>=80:
        print('%.2f' % topLengthPercent + '% of mapped reads have length '+str(lengthAndCount[0][0]))
    else:
        print('WARNING: Less than 80% of mapped reads have the same length!')
        print('Top read lengths:')
        secondLengthPercent=100*(lengthAndCount[1][1]/totalMappedReads)
        print('%.2f' % topLengthPercent + '% of mapped reads have length '+str(lengthAndCount[0][0]))
        print('%.2f' % secondLengthPercent + '% of mapped reads have length '+str(lengthAndCount[1][0]))
        if len(lengthAndCount)==3:
            thirdLengthStartPercent=100*(lengthAndCount[2][1]/totalMappedReads)
            print('%.2f' % thirdLengthPercent + '% of mapped reads start at position '+str(lengthAndCount[2][0]))
        print('The top read length will be assumed as correct if no read length was specified as input.')
    return(startAndCount[0][0]+1,lengthAndCount[0][0])


# In[12]:


def PlotStarts(startCounts,bamfile):
    """Saves a pdf of the barplot of read start frequency by position (1-based coordinates)

    Args:
        startCounts (Counter): counts of mapped reads starting at each position (0-based coordinates)
        bamfile (str): /path/and/name/to.bam 

    Returns:
        Nothing
        
    Saves:
        pdf of barplot of read start frequency by position (1-based coordinates) within the same directory as 
        the original bam file
        
    Usage:
        PlotStarts(startCounts,'/path/and/name/to.bam')
    """
    plt.figure(figsize=(12,6))
    #changes to 1-based coordinates to match vector positional system
    truePos=[start+1 for start in startCounts.keys()]
    plt.bar(truePos,np.log10(list(startCounts.values())),width=5)
    plt.title('Log10 Frequency of Read Starts by Position',fontsize=22)
    plt.xlabel('Start Position (1-based)',fontsize=18)
    plt.ylabel('Frequency of Reads Starts (log10)',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(top=np.log10(max(startCounts.values()))+.5)
    plt.tight_layout()
    directory,file=os.path.split(bamfile)
    plt.savefig(directory+'/ReadStarts_'+file.rsplit('.',1)[0]+'.pdf',format='pdf')


# In[48]:


def ColumnAppend(rowList,colList):
    """Appends items to an existing list. Not a difficult task but cleans up main section of code.

    Args:
        rowList (list): List to append to
        colList (list): Items to append - if the items are a list, they are incorporated into existing list. 

    Returns:
        rowList (list): original list with additional items added
        
    Usage:
        rowList=ColumnAppend(rowList,colList)
        ColumnAppend([2,4],[1,4,6])
        [2,4,1,4,6]
        ColumnAppend([2,4],[[1,1,0],4,2])
        [2,4,1,1,0,4,2]
    """
    for col in colList:
        if isinstance(col,list):
            rowList+=col
        else:
            rowList.append(col)
    return(rowList)


# In[141]:


def ComputeIsoformCounts(bamfile,visibleIsoforms,readStarts,tol=4):
    """Computes dataset by barcode for each bam file. Resulting dataset contains a count of total reads, unmapped
    reads, reads with a bad start (greater than 5 (default) basepairs away from most common start), how many reads
    match each isoform, and the psis for each isoform (matching reads/usable reads).

    Args:
        bamfile (str): /path/and/name/to.bam 
        visibleIsoforms (dictionary {str:[int,int,int]}): Dictionary with each key corresponding to a particular
        isoform and the values are a list of all junctions we could see within each read given the read length
        readStarts (int): either user specified start or most common start site within the file
        
    Returns:
        Nothing
        
    Saves:
        A text file of a pandas data frame with the following columns: barcode, total reads, unmapped reads, bad
        start reads, reads matching each isoform, usable reads (total - (unmapped+badstart)), psi for each isoform
        (matching/mapped) which is saved in the same directory as the bamfile
        
    Usage:
        ComputeIsoformCounts('/path/and/name/to.bam',visibleIsoforms,readStarts)
    """
    samFile=ImportBam(bamfile)
    checkJunctions=list(set([v for val in visibleIsoforms.values() for v in val]))
    checkJunctions.sort()
    lastBC=None
    dfList=[]
    print(visibleIsoforms)
    for read in samFile:
        rowList=[]
        currBC=read.get_tag('RX')
        readPos=read.get_reference_positions()
        #if currBC=='ATCTATTGGAATCCGGTCTGTGG':
            #print(readPos)
            #print(read.is_unmapped)
            #if not read.is_unmapped:
                #print(abs(readPos[0]-readStarts))
        if read.is_unmapped:
            iUnmapped=1
            ibadStart=0
            iMatches=[0]*len(visibleIsoforms)
        elif abs(readPos[0]-readStarts)>tol:
            iUnmapped=0
            ibadStart=1
            iMatches=[0]*len(visibleIsoforms)
        else:
            iUnmapped=0
            ibadStart=0
            iMatches=CheckJunctions(readPos,checkJunctions,visibleIsoforms,tol)
            #if currBC=='ATCTATTGGAATCCGGTCTGTGG':
                #print(iMatches)
        if lastBC and lastBC==currBC:
            total+=1
            unmapped+=iUnmapped
            badStarts+=ibadStart
            matches=[match+iMatches[idx] for idx,match in enumerate(matches)]
        elif lastBC and lastBC!=currBC:
            rowList=ColumnAppend(rowList,[lastBC,total,unmapped,badStarts,matches])
            #if lastBC=='ATCTATTGGAATCCGGTCTGTGG':
                #print(rowList)
            dfList.append(rowList)
            rowList=[]
            lastBC=currBC
            total=1
            unmapped=iUnmapped
            badStarts=ibadStart
            matches=iMatches
        else:
            lastBC=currBC
            total=1
            unmapped=iUnmapped
            badStarts=ibadStart
            matches=iMatches
    rowList=ColumnAppend(rowList,[lastBC,total,unmapped,badStarts,matches])
    dfList.append(rowList)
    samFile.close()
    psidf=pd.DataFrame(dfList, columns=['barcode','num_reads','unmapped_reads','bad_starts']+list(visibleIsoforms.keys()))
    psidf['usable_reads']=psidf.num_reads-(psidf.unmapped_reads+psidf.bad_starts)
    psis=[[psidf.iloc[:,-(idx+1)]/psidf.usable_reads] for idx in range(len(visibleIsoforms),0,-1)]
    for idx,iso in enumerate(visibleIsoforms.keys()):
        colName=iso+'_psi'
        psidf[colName]=psis[idx][0]
    directory,file=os.path.split(bamfile)
    junk,exon=os.path.split(directory)
    outfile='/PSIdata_byBC_'+file.rsplit('.',2)[0]+'_'+exon+'.txt'
    if not os.path.exists(directory+outfile):
        psidf.to_csv(directory+outfile, index=None, mode='a', sep='\t')
    else:
        psidf.to_csv(directory+outfile, index=None, mode='w', sep='\t')


# In[134]:


def CheckJunctions(readPos,possibleJunctions,visibleIsoforms,tol=4):
    """Checks if each junction appears in a read with 3 (default) bases of wiggle room on either side of the 
    junction and then matches the junction to each possible isoform.

    Args:
        readPos (list): List of reference positions that the read maps to from pysam get_reference_positions
        possibleJunctions (list): Sorted list of all junctions that appear in any isoform
        visibleIsoforms (dictionary {str:[int,int,int]}): Dictionary with each key corresponding to a particular
        isoform and the values are a list of all junctions we could see within each read given the read length

    Returns:
        isoforms (list): Truth value (0,1) for each isoform - shows if read matches each isoform
        
    Usage:
        isoforms=CheckJunctions(readPos,possibleJunctions,visibleIsoforms)
    """
    present=[]
    for junction in possibleJunctions:
        #allows some wiggle and misspecification in the bed file (3 bases on either side of the specified junction)
        for j in range(junction-tol,junction+tol+1):
            if j in readPos:
                present.append(junction)
                break
    isoforms=[]
    junctionsPresent=sum(present)
    for iso in visibleIsoforms.values():
        if iso==present:
            isoforms.append(1)
        else:
            isoforms.append(0)
    return(isoforms)


# In[64]:


def JunctionCentricSplicing(bamfile,bedfile=None,readStarts=None,readLength=None):
    startSite,length=CreatePreProcessPlots(bamfile)
    if not readStarts:
        readStarts=startSite
    if not readLength:
        readLength=length
    annotdf=ImportAnnot(bedfile)
    visibleIsoforms=DefineJunctions(annotdf,readStarts,readLength)
    ComputeIsoformCounts(bamfile,visibleIsoforms,readStarts)


# In[ ]:


get_ipython().run_cell_magic('time', '', "files=['/nfs/kitzman2/smithcat/proj/campersplice/rna-seq/exon10/JKP555_cDNA_fixed_rep1.tagsort.bam',\n      '/nfs/kitzman2/smithcat/proj/campersplice/rna-seq/exon10/JKP555_cDNA_fixed_rep2.tagsort.bam']\nfor file in files:\n    JunctionCentricSplicing(file,\n                           '/nfs/kitzman2/smithcat/proj/campersplice/refs/jkp0555_exon10.annots.bed',\n                          649,150)")


# In[142]:


get_ipython().run_cell_magic('time', '', "files=['/nfs/kitzman2/smithcat/proj/campersplice/rna-seq/exon11/JKP556_cDNA_fixed_rep1.tagsort.bam',\n      '/nfs/kitzman2/smithcat/proj/campersplice/rna-seq/exon11/JKP556_cDNA_fixed_rep2.tagsort.bam']\nfor file in files:\n    JunctionCentricSplicing(file,\n                           '/nfs/kitzman2/smithcat/proj/campersplice/refs/jkp0556_exon11.annots.bed',\n                          649,150)")

