function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}



eval $(parse_yaml config.yaml)

o_prefix=$output_dir/$output_prefix
mkdir -p $o_prefix

#check if basset_samplefile exists

# if file present
if [[ -f ${samplefile_basset} ]]; then
echo Peak centering
# set avoid regions = unmap regions + non-peaks

# get merged peaks from bed files
echo Generating bed region combined file for all TFs
./bed_generation.py -y -m 200 -s $input_size \
                  -o $o_prefix -c $genomefile_size \
                  $samplefile_basset
# sort bedfile and genomesize file
bedfile="$o_prefix.bed"
actfile="${o_prefix}_act.bed"

python act_bed_construction.py "$o_prefix.bed" "$o_prefix"_act.txt

clean_bedfile="$o_prefix"_cleanpeak.bed
clean_actfile="$o_prefix"_cleanact.bed


bedtools intersect -a $bedfile -b $genomefile_unmap -v > $clean_bedfile
bedtools intersect -a $actfile -b $genomefile_unmap -v > $clean_actfile

rm $bedfile
rm $actfile
rm ${o_prefix}_act.txt

grep -w $chroms_valid $clean_bedfile > "valid_bedfile.bed"
grep -w $chroms_test $clean_bedfile > "test_bedfile.bed"
grep -w -v -e $chroms_valid -e $chroms_test $clean_bedfile > "train_bedfile.bed"

grep -w $chroms_valid $clean_actfile > "valid_actfile.bed"
grep -w $chroms_test $clean_actfile > "test_actfile.bed"
grep  -w -v -e $chroms_valid -e $chroms_test $clean_actfile > "train_actfile.bed"


bedtools getfasta -fi $genomefile_fa -s -bed "train_bedfile.bed" -fo 'train.fa'
bedtools getfasta -fi $genomefile_fa -s -bed "valid_bedfile.bed" -fo 'valid.fa'
bedtools getfasta -fi $genomefile_fa -s -bed "test_bedfile.bed" -fo 'test.fa'

python basset_write_tfr.py ./ "$o_prefix.h5"

mv ${o_prefix}*.* $o_prefix
rm train_bedfile.bed valid_bedfile.bed test_bedfile.bed
rm valid_actfile.bed test_actfile.bed train_actfile.bed
rm test.fa train.fa valid.fa
mv config.yaml $o_prefix
fi

