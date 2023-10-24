def DPC(sequences):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []

    DiPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    AADict = dict()
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for sequence in sequences:
        code = [0] * 400
        sequence = sequence.replace("X", "")
        for i in range(len(sequence) - 1):

            code[AADict[sequence[i]] * 20 + AADict[sequence[i+1]]] = code[AADict[sequence[i]] * 20 + AADict[sequence[i+1]]] + 1
        if sum(code) != 0:
            code = [i / len(sequence) for i in code]
        encodings.append(code)
    return encodings
