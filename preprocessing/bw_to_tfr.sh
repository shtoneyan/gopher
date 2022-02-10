#!/bin/bash

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
mkdir -p $output_dir

#check if basset_samplefile exists
echo ${samplefile_basset}
if [ "$samplefile_basset" = 'random' ] || [ "$samplefile_basset" = 'Random' ]; then
  echo Chopping randomly
    # set avoid regions as unmap regions only
  avoid_regions=$genomefile_unmap
else
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
    bedtools intersect -a $bedfile -b $genomefile_unmap -v > clean_peaks.bed
    clean_bedfile=clean_peaks.bed
    sorted_bedfile="sorted_bedfile.bed"
    sorted_genome="sorted_genome.bed"
    echo Sorting bedfile and genome file
    sort -k1,1 -k2,2n $clean_bedfile > $sorted_bedfile # sort best bed
    sort -k1,1 -k2,2n $genomefile_size > $sorted_genome # sort genome

    # get the complement of the sorted bed and the genome to get which parts to avoid
    echo Generating bed file complementary to peak regions
    bedtools complement -i $sorted_bedfile -g $sorted_genome > nonpeaks.bed
    # complete the avoid regions by adding unmappable
    ###REVERSE
    # cat nonpeaks.bed $genomefile_unmap > avoid_regions.bed
    cat nonpeaks.bed > avoid_regions.bed
    sort -k1,1 -k2,2n avoid_regions.bed > sorted_avoid_regions.bed
    echo Merging nonpeak and blacklisted regions
    bedtools merge -i sorted_avoid_regions.bed > merged_avoid_regions.bed

    rm nonpeaks.bed
    rm avoid_regions.bed
    rm sorted_avoid_regions.bed
    rm $sorted_genome
    rm $sorted_bedfile
    avoid_regions=merged_avoid_regions.bed
  else
    # if file not found exit
    echo ERROR: basset samplefile does not exist!
    exit
  fi

fi


./basenji_data.py $genomefile_fa \
                    $samplefile_basenji \
                    -g $avoid_regions \
                    -l $input_size -o $output_dir/$output_prefix \
                    -t $chroms_test -v $chroms_valid \
                    -w $input_pool --local -d $input_downsample --norm $input_norm \
                    --step $input_step --padding $input_padding -p 19 --threshold $threshold \
                    --test_threshold $test_threshold \
                    --only_chroms $chroms_only \
                    --stride_test $stride_test
if [[ -f ${samplefile_basset} ]]; then
  mv merged_avoid_regions.bed "$output_dir/$output_prefix/"
  mv $bedfile "$output_dir/$output_prefix/"
  mv $clean_bedfile "$output_dir/$output_prefix/"
  mv "${o_prefix}_act.txt" "$output_dir/$output_prefix/"
fi
mv config.yaml "$output_dir/$output_prefix/"
mkdir -p $output_dir/zip_datasets
zip -q -r $output_dir/zip_datasets/${output_prefix}.zip $output_dir/$output_prefix -x *seqs_cov*
