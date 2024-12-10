import math
from collections import Counter
from Bio import SeqIO
import numpy as np
from ImbDpro import *
from sklearn.preprocessing import MinMaxScaler
from DataPreprocessing import *

def minSequenceLength(fastas):
	minLen = 10000000000
	for i in fastas:
		if minLen > len(i[1]):
			minLen = len(i[1])
	return minLen

def countnum(seq, nuacid):
    return len([1 for i in range(len(seq)) if seq.startswith(nuacid, i)])

# k-mer
def construct_kmer():
    ntarr = ("A", "C", "G", "T")

    kmerArray = []

    for n in range(4):
        kmerArray.append(ntarr[n])

    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            kmerArray.append(str2)
    #############################################
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                kmerArray.append(str3)
    #############################################
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                for y in range(4):
                    str4 = str3 + ntarr[y]
                    kmerArray.append(str4)
    ############################################
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                for y in range(4):
                    str4 = str3 + ntarr[y]
                    for z in range(4):
                        str5 = str4 + ntarr[z]
                        kmerArray.append(str5)
    return kmerArray

def get_kmer(seq,kmerArray):
    rst = []
    total = 0.0
    for n in range(len(kmerArray)):
        item = countnum(seq, kmerArray[n])
        total = total + item
        rst.append(item)
    for n in range(len(rst)):
        if total!=0:
            rst[n] = rst[n]/total

    return rst

def KMER(fastas):
    seq_data = []
    kmerArray = construct_kmer()
    header = ['#']
    
    for ka in kmerArray:
        header.append('Kmer_' + ka)
    seq_data.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        seq_feature = get_kmer(sequence, kmerArray)
        seq_data.append([name] + seq_feature)
        if i == 5000 or i == 10000 or i == 20000:
            print(i)
    return seq_data

# g-gap
def big_gap_single(seq, ggaparray, g):
    # seq length is fix =23

    rst = np.zeros((256))
    for i in range(len(seq) - 1 - g):
        str1 = seq[i] + seq[i + 1]
        str2 = seq[i + g] + seq[i + 1 + g]
        idx = ggaparray.index(str1 + str2)
        rst[idx] += 1

    for j in range(len(ggaparray)):
        rst[j] = rst[j] / (len(seq) - 1 - g)  # l-1-g

    return rst

def GGAP(fastas, g=2):
    ggaparray = construct_kmer()[84:340]
    encodings = []
    header = ['#']
    
    for bg in ggaparray:
        header.append('BiGGAP_' + bg)
    encodings.append(header)
    
    for x in fastas:
        name, sequence = x[0], x[1]
        temp = big_gap_single(sequence, ggaparray, g)
        encodings.append([name] + temp.tolist())
    return encodings

# SSM
def ssm_single(seq, ggaparray, g):
    # seq length is fix =23

    rst = np.zeros((16))
    for i in range(len(seq) - 1 - g):
        str1 = seq[i]
        str2 = seq[i + 1 + g]
        idx = ggaparray.index(str1 + str2)
        rst[idx] += 1

    for j in range(len(ggaparray)):
        rst[j] = rst[j] / (len(seq) - 1 - g)  # l-1-g

    return rst

def SSM(fastas):
    g = [1,2,3]
    ssmarray = construct_kmer()[4:20]
    encodings = []
    header = ['#']
    
    for ssm in ssmarray:
        header.append('SSM_1_' + ssm)
    for ssm in ssmarray:
        header.append('SSM_2_' + ssm)
    for ssm in ssmarray:
        header.append('SSM_3_' + ssm)
    encodings.append(header)
    
    for x in fastas:
        name, sequence = x[0], x[1]
        temp0 = ssm_single(sequence, ssmarray, g[0])
        temp1 = ssm_single(sequence, ssmarray, g[1])
        temp2 = ssm_single(sequence, ssmarray, g[2])
        temp = [name] + temp0.tolist() + temp1.tolist() + temp2.tolist()
        encodings.append(temp)
        
    return encodings

# AAC 20D
def AAC(fastas):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = ['#']

    for aa in AA:
        header.append('AAC_' + aa)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        count = Counter(sequence)
        for key in count:
            count[key] = count[key] / len(sequence)
        code = [name]
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return encodings

# DPC 400D
def DPC(fastas):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#']
    for aa in diPeptides:
        header.append('DPC_' + aa)
    encodings.append(header)

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        tmpCode = [0] * 400
        for j in range(len(sequence) - 1):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] + 1
        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings

# CKSAAP 400D
def CKSAAP(fastas, gap=1):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if minSequenceLength(fastas) < gap + 2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#']

    for aa in diPeptides:
        header.append('CKSAAP_' + aa + '.gap' + str(gap))
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        myDict = {}
        for pair in diPeptides:
            myDict[pair] = 0
        sum = 0
        for index1 in range(len(sequence)):
            index2 = index1 + gap + 1
            if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
                myDict[sequence[index1] + sequence[index2]] += 1
                sum += 1
        for pair in diPeptides:
            code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings

def ASDC(fastas):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#']

    for aa in diPeptides:
        header.append('ASDC_' + aa)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        myDict = {}
        for pair in diPeptides:
            myDict[pair] = 0
        sum = 0
        for index1 in range(len(sequence)):
            for index2 in range(index1 + 1, len(sequence)):
                if sequence[index1] in AA and sequence[index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] += 1
                    sum += 1
        for pair in diPeptides:
            code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings

# GAAC 5D
def GAAC(fastas):
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }
    groupKey = group.keys()

    encodings = []
    header = ['#']
    for key in groupKey:
        header.append('GAAC_' + key)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        count = Counter(sequence)
        myDict = {}
        for key in groupKey:
            for aa in group[key]:
                myDict[key] = myDict.get(key, 0) + count[aa]

        for key in groupKey:
            code.append(myDict[key] / len(sequence))
        encodings.append(code)

    return encodings

# GDPC 25D
def GDPC(fastas):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group.keys()
    dipeptide = [g1 + '.' + g2 for g1 in groupKey for g2 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    encodings = []
    header = ['#']
    for key in dipeptide:
        header.append('GDPC_' + key)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]

        code = [name]
        myDict = {}
        for t in dipeptide:
            myDict[t] = 0

        sum = 0
        for j in range(len(sequence) - 1):
            myDict[index[sequence[j]] + '.' + index[sequence[j + 1]]] = myDict[index[sequence[j]] + '.' + index[sequence[j + 1]]] + 1
            sum = sum + 1

        if sum == 0:
            for t in dipeptide:
                code.append(0)
        else:
            for t in dipeptide:
                code.append(myDict[t] / sum)
        encodings.append(code)

    return encodings

# GTPC 125D
def GTPC(fastas):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group.keys()

    triple = [g1 + '.' + g2 + '.' + g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    encodings = []
    header = ['#']
    for key in triple:
        header.append('GTPC_' + key)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]

        code = [name]
        myDict = {}
        for t in triple:
            myDict[t] = 0

        sum = 0
        for j in range(len(sequence) - 2):
            myDict[index[sequence[j]] + '.' + index[sequence[j + 1]] + '.' + index[sequence[j + 2]]] = myDict[index[
                                                                                                                  sequence[
                                                                                                                      j]] + '.' +
                                                                                                              index[
                                                                                                                  sequence[
                                                                                                                      j + 1]] + '.' +
                                                                                                              index[
                                                                                                                  sequence[
                                                                                                                      j + 2]]] + 1
            sum = sum + 1

        if sum == 0:
            for t in triple:
                code.append(0)
        else:
            for t in triple:
                code.append(myDict[t] / sum)
        encodings.append(code)

    return encodings

# CKSAAGP 25D
def CKSAAGP(fastas, gap=1):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0
    if minSequenceLength(fastas) < gap + 2:
        print('Error: all the sequence length should be greater than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }
    AA = 'ARNDCQEGHILKMFPSTWYV'
    groupKey = group.keys()
    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key
    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)
    encodings = []
    header = ['#']

    for p in gPairIndex:
        header.append('CKSAAGP_' + p + '.gap' + str(gap))
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        gPair = {}
        for key1 in groupKey:
            for key2 in groupKey:
                gPair[key1 + '.' + key2] = 0
        sum = 0
        for p1 in range(len(sequence)):
            p2 = p1 + gap + 1
            if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[index[sequence[p1]] + '.' + index[
                    sequence[p2]]] + 1
                sum = sum + 1
        if sum == 0:
            for gp in gPairIndex:
                code.append(0)
        else:
            for gp in gPairIndex:
                code.append(gPair[gp] / sum)
        encodings.append(code)
    return encodings

# CTDC 39D
def CTDC(fastas):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = ['#']
    for p in property:
        for g in range(1, len(groups) + 1):
            header.append('CTDC_' + p + '.G' + str(g))
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for p in property:
            sum1 = 0
            for aa in group1[p]:
                sum1 = sum1 + sequence.count(aa)
            c1 = sum1 / len(sequence)

            sum2 = 0
            for aa in group2[p]:
                sum2 = sum2 + sequence.count(aa)
            c2 = sum2 / len(sequence)

            sum3 = 0
            for aa in group3[p]:
                sum3 = sum3 + sequence.count(aa)
            c3 = sum3 / len(sequence)

            code = code + [c1, c2, c3]
        encodings.append(code)
    return encodings

# CTDT 39D
def CTDT(fastas):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = ['#']
    for p in property:
        for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
            header.append('CTDT_' + p + '.' + tr)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 = c1221 + 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 = c1331 + 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 = c2332 + 1
            code = code + [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
        encodings.append(code)
    return encodings

# CTDD 195D
def Count(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence))
                    break
        if myCount == 0:
            code.append(0)
    return code

def CTDD(fastas):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = ['#']
    for p in property:
        for g in ('1', '2', '3'):
            for d in ['0', '25', '50', '75', '100']:
                header.append('CTDD_' + p + '.' + g + '.residue' + d)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for p in property:
            code = code + Count(group1[p], sequence) + Count(group2[p], sequence) + Count(group3[p], sequence)
        encodings.append(code)
    return encodings

# Fasta
def ReadFileFromFasta(filepath):
    seq = []
    for seq_record in SeqIO.parse(filepath, "fasta"):
        seq.append(['>' + seq_record.id.strip(), str(seq_record.seq).strip()])
    return seq

def FeatureGenerator(fastas, flag=1, types='aas'):
    labels = []

    for i in fastas:
        name = i[0]
        if str(name).startswith('>') and str(name).find('_URS') != -1:
            labels.append(0)
        else:
            labels.append(1)
            
    #Divide subset directly
    if types == 'aas':
        if flag == 1:
            flf_dict = {}
            data_dict = getSubsetDir(fastas, labels)
            for key,value in data_dict.items():
                fastas = getAAs(value[:,:-1])
                labels = value[:,-1]
                FeatureDict, labels, FeatureNameDict = FeatureExtraction(fastas, labels, types='aas')
                flf_dict[key] = (FeatureDict, labels, FeatureNameDict)
            return flf_dict, data_dict
        else:
            FeatureDict, labels, FeatureNameDict = FeatureExtraction(getAAs(fastas), labels, types='aas')
            return FeatureDict, labels, FeatureNameDict
    else:
        FeatureDict, labels, FeatureNameDict = FeatureExtraction(fastas, labels, types='sorfs')
        return FeatureDict, labels, FeatureNameDict
    

def FeatureExtraction(fastas, labels, types='aas'):
    scaler = MinMaxScaler()
    if types == 'aas':
        FeatureDict = {}
        FeatureNameDict = {}
        aac = np.array(AAC(fastas))
        dpc = np.array(DPC(fastas))
        asdc = np.array(ASDC(fastas))
        # c1saap = np.array(CKSAAP(fastas, 1))
        c2saap = np.array(CKSAAP(fastas, 2))
        c3saap = np.array(CKSAAP(fastas, 3))
        # gaac = np.array(GAAC(fastas))
        # gdpc = np.array(GDPC(fastas))
        # gtpc = np.array(GTPC(fastas))
        # c1saagp = np.array(CKSAAGP(fastas, 1))
        # c2saagp = np.array(CKSAAGP(fastas, 2))
        # c3saagp = np.array(CKSAAGP(fastas, 3))
        # ctdc = np.array(CTDC(fastas))
        # ctdt = np.array(CTDT(fastas))
        # ctdd = np.array(CTDD(fastas))
        
        #aa
        FeatureDict['aac'] = scaler.fit_transform(np.array(aac[1:, 1:], dtype=float))
        FeatureNameDict['aac'] = aac[:1, :][0]

        FeatureDict['dpc'] = scaler.fit_transform(np.array(dpc[1:, 1:], dtype=float))
        FeatureNameDict['dpc'] = dpc[:1, :][0]

        FeatureDict['asdc'] = scaler.fit_transform(np.array(asdc[1:, 1:], dtype=float))
        FeatureNameDict['asdc'] = asdc[:1, :][0]

        # FeatureDict['c1saap'] = scaler.fit_transform(np.array(c1saap[1:, 1:], dtype=float))
        # FeatureNameDict['c1saap'] = c1saap[:1, :][0]

        FeatureDict['c2saap'] = scaler.fit_transform(np.array(c2saap[1:, 1:], dtype=float))
        FeatureNameDict['c2saap'] = c2saap[:1, :][0]

        FeatureDict['c3saap'] = scaler.fit_transform(np.array(c3saap[1:, 1:], dtype=float))
        FeatureNameDict['c3saap'] = c3saap[:1, :][0]

        # FeatureDict['gaac'] = scaler.fit_transform(np.array(gaac[1:, 1:], dtype=float))
        # FeatureNameDict['gaac'] = gaac[:1, :][0]

        # FeatureDict['gdpc'] = scaler.fit_transform(np.array(gdpc[1:, 1:], dtype=float))
        # FeatureNameDict['gdpc'] = gdpc[:1, :][0]

        # FeatureDict['gtpc'] = scaler.fit_transform(np.array(gtpc[1:, 1:], dtype=float))
        # FeatureNameDict['gtpc'] = gtpc[:1, :][0]

        # FeatureDict['c1saagp'] = scaler.fit_transform(np.array(c1saagp[1:, 1:], dtype=float))
        # FeatureNameDict['c1saagp'] = c1saagp[:1, :][0]

        # FeatureDict['c2saagp'] = scaler.fit_transform(np.array(c2saagp[1:, 1:], dtype=float))
        # FeatureNameDict['c2saagp'] = c2saagp[:1, :][0]

        # FeatureDict['c3saagp'] = scaler.fit_transform(np.array(c3saagp[1:, 1:], dtype=float))
        # FeatureNameDict['c3saagp'] = c3saagp[:1, :][0]

        # FeatureDict['ctdc'] = scaler.fit_transform(np.array(ctdc[1:, 1:], dtype=float))
        # FeatureNameDict['ctdc'] = ctdc[:1, :][0]

        # FeatureDict['ctdt'] = scaler.fit_transform(np.array(ctdt[1:, 1:], dtype=float))
        # FeatureNameDict['ctdt'] = ctdt[:1, :][0]

        # FeatureDict['ctdd'] = scaler.fit_transform(np.array(ctdd[1:, 1:], dtype=float))
        # FeatureNameDict['ctdd'] = ctdd[:1, :][0]
        
        org_feature_info = {}
        for k,v in FeatureNameDict.items():
            org_feature_info[k]=len(v)
        print("The dimension of original aas feature, ", org_feature_info)
        return FeatureDict, np.array(labels).astype(int), FeatureNameDict
    else:
        FeatureDictsorfs = {}
        FeatureNameDictsorfs = {}
        
        kmer = np.array(KMER(fastas))
        ssm = np.array(SSM(fastas))
        ggap1 = np.array(GGAP(fastas, g=1))
        ggap2 = np.array(GGAP(fastas, g=2))
        
        #sORFs
        FeatureDictsorfs['kmer'] = scaler.fit_transform(np.array(kmer[1:, 1:], dtype=float))
        FeatureNameDictsorfs['kmer'] = kmer[:1, :][0]
        
        FeatureDictsorfs['ssm'] = scaler.fit_transform(np.array(ssm[1:, 1:], dtype=float))
        FeatureNameDictsorfs['ssm'] = ssm[:1, :][0]
        
        FeatureDictsorfs['ggap1'] = scaler.fit_transform(np.array(ggap1[1:, 1:], dtype=float))
        FeatureNameDictsorfs['ggap1'] = ggap1[:1, :][0]
        
        FeatureDictsorfs['ggap2'] = scaler.fit_transform(np.array(ggap2[1:, 1:], dtype=float))
        FeatureNameDictsorfs['ggap2'] = ggap2[:1, :][0]
        
        org_feature_info = {}
        for k,v in FeatureNameDictsorfs.items():
            org_feature_info[k]=len(v)
        print("The dimension of original sorfs feature, ", org_feature_info)
        return FeatureDictsorfs, np.array(labels).astype(int), FeatureNameDictsorfs