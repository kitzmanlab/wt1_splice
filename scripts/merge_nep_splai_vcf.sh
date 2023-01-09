module load bcftools

/home/smithcat/bs list datasets

/home/smithcat/bs download dataset -i ds.20a701bc58ab45b59de2576db79ac8d0 --extension=masked.snv.hg38.vcf.gz -o /nfs/kitzman2/lab_common2/annots/spliceai/1.3/
/home/smithcat/bs download dataset -i ds.20a701bc58ab45b59de2576db79ac8d0 --extension=masked.snv.hg38.vcf.gz.tbi -o /nfs/kitzman2/lab_common2/annots/spliceai/1.3/

bcftools view -r Y /nfs/kitzman2/lab_common2/annots/spliceai/1.3/spliceai_scores.masked.snv.hg38.vcf.gz | bgzip > /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/spliceai_scores.masked.snv.hg38.chrY.vcf.gz

tabix -p vcf /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/spliceai_scores.masked.snv.hg38.chrY.vcf.gz

bcftools annotate --rename-chrs /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/chr_name_conv.txt /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/spliceai_scores.masked.snv.hg38.chrY.vcf.gz | bgzip > /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/spliceai_scores.masked.snv.hg38.chrY_rename.vcf.gz

tabix -p vcf /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/spliceai_scores.masked.snv.hg38.chrY_rename.vcf.gz

#I think this is the regular expression I want - requires at least one SpliceAI prediction to be >= 10%
bcftools view -e'SpliceAI~".|.*|0.0.|0.0.|0.0.|0.0.|"' /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/spliceai_scores.masked.snv.hg38.chrY_rename.vcf.gz | less

bcftools isec -c none -n=2 -O z -p /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/tmp/ -e- -e'SpliceAI~".|.*|0.0.|0.0.|0.0.|0.0.|"' /nfs/kitzman2/neptune/202207_wgs_vcfs/chrY.hg38.annotated.vcf.gz /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/spliceai_scores.masked.snv.hg38.chrY_rename.vcf.gz 

#that produced two separate vcfs...can I merge them together?

bcftools merge /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/0000.vcf.gz /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/0001.vcf.gz -o /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/nep_splai_merge_chrY.vcf

for i in {1..22} X Y
do
echo "$i chr$i" >> /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/chr_name_conv.txt
done

bcftools annotate --rename-chrs /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/chr_name_conv.txt /nfs/kitzman2/lab_common2/annots/spliceai/1.3/spliceai_scores.masked.snv.hg38.vcf.gz | bgzip > /nfs/kitzman2/lab_common2/annots/spliceai/1.3/spliceai_scores.masked.snv.hg38_chr.vcf.gz

tabix -p vcf /nfs/kitzman2/lab_common2/annots/spliceai/1.3/spliceai_scores.masked.snv.hg38_chr.vcf.gz

bcftools view -e'SpliceAI~".|.*|0.0.|0.0.|0.0.|0.0.|"' -O z -o /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/tmp/spliceai_scores.masked.snv.hg38_chr.filt.vcf.gz --threads 8 /nfs/kitzman2/lab_common2/annots/spliceai/1.3/spliceai_scores.masked.snv.hg38_chr.vcf.gz

tabix -p vcf /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/tmp/spliceai_scores.masked.snv.hg38_chr.filt.vcf.gz

for i in {1..22} X Y
do

echo chr$i

bcftools isec -c none -n=2 -O z -p /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/tmp/ /nfs/kitzman2/neptune/202207_wgs_vcfs/chr$i.hg38.annotated.vcf.gz /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/tmp/spliceai_scores.masked.snv.hg38_chr.filt.vcf.gz 

bcftools merge -O z -o /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/nep_splai_merge_chr$i.hg38.vcf.gz --threads 8 /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/tmp/0000.vcf.gz /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/tmp/0001.vcf.gz

tabix -p vcf /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/nep_splai_merge_chr$i.hg38.vcf.gz

done

bcftools concat -O z -o /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/nep_splai_merge_allchrom.hg38.vcf.gz /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/*.hg38.vcf.gz

bcftools sort -O z -o /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/nep_splai_merge_allchrom_sort.hg38.vcf.gz /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/nep_splai_merge_allchrom.hg38.vcf.gz

tabix -p vcf /nfs/kitzman2/smithcat/proj/wt1_2022/neptune_splai/nep_splai_merge_allchrom_sort.hg38.vcf.gz



