import pandas as pd
import re
import sys
#read bed file
#constructure acitivity table
#output tfr file

def main():
    
    bed_file = sys.argv[1]
    act_table = sys.argv[2]  
    
    data = pd.read_csv(act_table,sep = '\t')
    data.rename(columns={'Unnamed: 0':'loci'}, inplace=True)
    
    chrom = [i.split(':')[0] for i in list(data.loci)]
    start = [re.split(':|-',i)[1] for i in list(data.loci)]
    end = [re.split(":|-",i)[2] for i in list(data.loci)]
    clean_end = [i[:-3] for i in end]
    strand = [i[-2] for i in end]
    data = data.drop(columns=['loci'])
    
    data['chrom'] = chrom
    data['start'] = start
    data['end'] = clean_end
    data['strand'] = strand
    cols = data.columns.tolist()
    cols = cols[-4:]+cols[:-4]
    data = data[cols]
    
    output_act = act_table.split('.txt')[0]+'.bed'
    data.to_csv(output_act,sep='\t',index = False)

##############################################################    
    
def align_seqs_scores_1hot(seq_vecs, seq_scores, sort=True):
    if sort:
        seq_headers = sorted(seq_vecs.keys())
    else:
        seq_headers = seq_vecs.keys()

    # construct lists of vectors
    train_scores = []
    train_seqs = []
    for header in seq_headers:
        train_seqs.append(seq_vecs[header])
        train_scores.append(seq_scores[header])

    # stack into matrices
    train_seqs = np.vstack(train_seqs)
    train_scores = np.vstack(train_scores)

    return train_seqs, train_scores    
    
if __name__ == '__main__':
    main()
