from collections import Counter


def AAC(sequences):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []

    for sequence in sequences:
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = []
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return encodings