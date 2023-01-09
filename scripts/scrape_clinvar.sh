#################################################### 
#
#    2022-1018 - clinvar (VCF) - WT1 
#
# /nfs/kitzman2/smithcat/proj/wt1_2022/clinvar/

wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/clinvar_20221015.vcf.gz -P /nfs/kitzman2/smithcat/proj/wt1_2022/clinvar/
​
bedtools intersect \
    -wa \
    -header \
    -a /nfs/kitzman2/smithcat/proj/wt1_2022/clinvar/clinvar_20221015.vcf.gz \
    -b /nfs/kitzman2/smithcat/proj/wt1_2022/clinvar/wt1_coding_ex9_pad50.bed \
| bgzip -c  > /nfs/kitzman2/smithcat/proj/wt1_2022/clinvar/clinvar_20221015_codex9pad50.vcf.gz

tabix -p vcf /nfs/kitzman2/smithcat/proj/wt1_2022/clinvar/clinvar_20221015_codex9pad50.vcf.gz
​
