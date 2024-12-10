# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:54:21 2024

@author: Z
"""

from Bio import Seq, SeqIO
import random

def TxttoFasta():
    fo1 = open("./data/pos.fa").readlines()
    newfo1 = []
    for i in range(0, len(fo1)):
        line = fo1[i].strip()
        if line[-3:] in ["TAA","TAG","TGA"]:
            newline = '>' + 'pri-miRNA_miPEPs_' + str(i) +'\n' + line[:-3] + '\n'
            newfo1.append(newline)
        else:
            newline = '>' + 'pri-miRNA_miPEPs_' + str(i) +'\n' + line + '\n'
            newfo1.append(newline)
    # print(len(newfo1))
    
    fo2 = open("./data/neg.fa").readlines()
    newfo2 = []
    flag = 0
    for i in range(0, len(fo2)):
        line = fo2[i].strip()
        if line[-3:] in ["TAA","TAG","TGA"]:
            if line[:3] == 'ATG':
                newline = '>' + 'ncRNA_URS_' + str(flag) +'\n' + line[:-3] + '\n'
                newfo2.append(newline)
                flag += 1
            else:
                continue
        else:
            if line[:3] != 'ATG':
                newline = '>' + 'ncRNA_URS_' + str(flag) +'\n' + line + '\n'
                newfo2.append(newline)
                flag += 1
            else:
                continue 
    # print(len(newfo2))
    
    # sequences with start condons ATG
    newfo = newfo1 + newfo2
    with open("./data/all_dataset.fa", 'w') as file:    
        file.writelines(newfo)
        
    random.seed(60)
    random.shuffle(newfo1)
    random.seed(60)
    random.shuffle(newfo2)
    
    
    # # training set
    # newfo = newfo1[:50] + newfo2[:500]
    # with open("./data/train_dataset.fa", 'w') as file:    
    #     file.writelines(newfo)
    
    # # testing set1
    # newfo = newfo1[50:100] + newfo2[500:1000]
    # with open("./data/test_dataset_one.fa", 'w') as file:    
    #     file.writelines(newfo)
    
    # # testing set2
    # newfo = newfo1[100:150] + newfo2[1000:1500]
    # with open("./data/test_dataset_two.fa", 'w') as file:    
    #     file.writelines(newfo)
    
    # # testing set3
    # newfo = newfo1[150:200] + newfo2[1500:2000]
    # with open("./data/test_dataset_three.fa", 'w') as file:    
    #     file.writelines(newfo) 
    
    
    # training set
    newfo = newfo1[:535] + newfo2[:5350]
    # newfo = newfo1[:10] + newfo2[:100]
    with open("./data/train_dataset.fa", 'w') as file:    
        file.writelines(newfo)
    
    # testing set1
    newfo = newfo1[535:635] + newfo2[5350:6850]
    with open("./data/test_dataset_one.fa", 'w') as file:    
        file.writelines(newfo)
    
    # testing set2
    newfo = newfo1[635:735] + newfo2[6850:7850]
    with open("./data/test_dataset_two.fa", 'w') as file:    
        file.writelines(newfo)
    
    # testing set3
    newfo = newfo1[735:] + newfo2[7850:8350]
    with open("./data/test_dataset_three.fa", 'w') as file:    
        file.writelines(newfo) 
    
# def ReadFileFromFasta(filepath):
#     seq = []
#     for seq_record in SeqIO.parse(filepath, "fasta"):
#         seq.append(['>' + seq_record.id.strip(), str(seq_record.seq).strip()])
#     return seq

def getAAs(sORFs):
    aas = []
    for sorf in sORFs:
        name, fasta = sorf
        fasta = Seq.Seq(fasta)
        aas.append((name, str(fasta.translate())))
        
    # with open("./data/train_aas.fa", 'w') as file:   
    #     for i in aas:
    #         name, fasta = i
    #         line = name + '\n' + fasta + '\n'  
    #         file.write(line)
         
    # print(aas)
    return aas

# sORFs = [("1","ATGGATGCC"),("2","ATGGATGCCATGGGG")]
# sORFs = ReadFileFromFasta(filepath="./data/all_dataset.fa")
# getAAs(sORFs)
# TxttoFasta()