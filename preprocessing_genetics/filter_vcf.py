import io
import os
import numpy as np
import pandas as pd
import gzip


def get_vcf_names(vcf_path):
    with open(vcf_path, "rt") as ifile:
          for line in ifile:
            if line.startswith("#CHROM"):
                vcf_names = [x for x in line.split('\t')]
                break
    ifile.close()
    return vcf_names


def read_vcf(path):
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})

#checks the mutation of a nucleotide from a list of 172 genes
def in_between(position, relevent):
    for i in range(len(relevent)):
        row = relevent.iloc[i]
        if (position >= relevent.iloc[i].start) and (position <= relevent.iloc[i].end):
            return True
    return False

def main():
    
    
    genes = pd.read_csv("gene_list.csv")
    files = os.listdir("vcf_folder")  
    
    
    for vcf_file in files:
        file_name = "vcf_folder/" + vcf_file
        
        output_file = open('log.txt','a')
        output_file.write(file_name)
        output_file.close()
        names = get_vcf_names(file_name)
        
        start = vcf_file.find("ADNI_ID.") + len("ADNI_ID.")
        end = -4
        substring = vcf_file[start:end]
        relevent = genes[genes["chrom"] == substring]
        relevent = relevent.reset_index()
        
        #Reading the file chunk by chunk by setting 1000 rows
        chunksize=1000
        cnt=0
        pos=0
        df = pd.DataFrame()
        for vcf in  pd.read_csv(file_name,comment='#', chunksize=chunksize, delim_whitespace=True, header=None, names=names):
            cnt+=1
            print(cnt)
            positions = vcf["POS"]
            for i in range(len(positions)):
                indexes = []
                boo = in_between(positions[pos], relevent)
                if boo:
                    temp_df = vcf.iloc[pos % chunksize]
                    df=df.append(temp_df,ignore_index = False)
                pos+=1
        print("creating the pickle file")
        df.to_pickle("pickle_files/" + vcf_file[:-4] + ".pkl")

        
    

    
if __name__ == '__main__':
    main()