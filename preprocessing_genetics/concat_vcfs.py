import io
import os
import numpy as np
import pandas as pd
cnt=1


def main():
    files = os.listdir("pickle_files")
    diag = pd.read_csv("ground_truth.csv")[["Subject", "Group"]]    
    diag = diag.rename(columns={"Subject": "index"})
    
    vcfs = []
    
    for vcf_file in files:
        file_name = "pickle_files/" + vcf_file
        
        vcf = pd.read_pickle(file_name)
	
        vcf = vcf.drop(['#CHROM', 'POS', 'ID','REF','ALT','QUAL','FILTER','INFO', 'FORMAT'], axis=1)
        vcf = vcf.T
        vcf.reset_index(level=0, inplace=True)
        vcf["index"] = vcf["index"].str.replace("s", "S").str.replace("\n", "")
        merged = diag.merge(vcf, on = "index")
        print(merged.columns)
        merged = merged.rename(columns={"index": "subject"})
        d = {'0/0': 0, '0/1': 1, '1/0': 1,  '1/1': 2, "./.": 3}
        cols = list(set(merged.columns) - set(["subject", "Group"]))   

        global cnt
        cnt=0     
        for col in cols:

            merged[col] = merged[col].str[:3].replace(d)
            cnt+=1
            print(cnt)
            idx = cols.index(col)
            if idx % 500 == 0:
                output_file = open('log_clean.txt','a')
                output_file.write("Percent done: " + str((idx/len(cols))*100) + "\n" + vcf_file + "\n")
                output_file.close()
        
        
        vcf = merged.groupby('subject', group_keys=False).apply(lambda x: x.loc[x.Group.idxmax()])
        print("adding to vcf ")
        vcfs.append(vcf)
    
    vcf = pd.concat(vcfs, ignore_index=True)
    vcf = vcf.drop_duplicates()
    print("creating the all vcfs pickle file")
    vcf.to_pickle("all_vcfs.pkl")



    
if __name__ == '__main__':
    main()