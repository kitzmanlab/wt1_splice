mkdir /nfs/kitzman2/lab_common2/annots/gtex
mkdir /nfs/kitzman2/lab_common2/annots/gtex/v8

cd /nfs/kitzman2/lab_common2/annots/gtex/v8/

wget https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz
wget https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt

header=$(gunzip -c /nfs/kitzman2/lab_common2/annots/gtex/v8/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz | awk 'NR==3')
#echo $header > /nfs/kitzman2/smithcat/proj/wt1_2022/tmp/chr$i.gtex_junctions.hg38.gct

for i in {1..22} X Y
do
echo chr$i
gunzip -c /nfs/kitzman2/lab_common2/annots/gtex/v8/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz |  awk -v chr="chr${i}_" 'NR==3 || $1 ~ chr' > /nfs/kitzman2/smithcat/proj/wt1_2022/tmp/chr$i.gtex_junctions.hg38.gct
bgzip /nfs/kitzman2/smithcat/proj/wt1_2022/tmp/chr$i.gtex_junctions.hg38.gct
done
