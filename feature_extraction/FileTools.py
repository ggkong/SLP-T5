def readfasta(path):
    # 按路径读取文件所有行数，内容储存在lines变量中
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    # 保存fasta文件的索引和序列
    indexList = []
    seqList = []

    # 保存临时序列，这是因为fasta文件的整个序列往往是多行的
    tempSeq = ""

    for line in lines:
        if '>' in line:
            indexList.append(line[1:].strip())  # 保存序列索引
            seqList.append(tempSeq.strip())  # 将读完的总序列保存到seqList
            tempSeq = ""  # 对tempSeq进行初始化
        else:
            tempSeq += line.strip()  # 将读到的本行序列追加到tempSeq
    seqList.append(tempSeq.strip())  # 循环执行完毕后，将最后一条序列保存到seqList

    del seqList[0]  # 第一次读入的seq是空字符串，需要将其去除

    return indexList, seqList  # 返回indexList与seqList

