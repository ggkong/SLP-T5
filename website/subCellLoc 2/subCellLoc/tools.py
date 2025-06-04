def handle_uploaded_file(f):
    chunk_s = ""
    for chunk in f.chunks():
        chunk_s = chunk
    str_seq = chunk_s.decode('UTF-8')
    # 通过分割换行符，将多行字符串分割成列表
    lines = str_seq.split('\n')
    remaining_lines = '\n'.join(lines[1:])
    # 通过分割换行符，将多行字符串分割成列表
    lines = remaining_lines.split('\n')

    # 去掉每行前后的空格，并使用空格连接各行
    single_line_string = ''.join(line.strip() for line in lines if line.strip())
    return single_line_string
