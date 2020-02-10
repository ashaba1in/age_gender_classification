def main():
    filename = 'coursework.tex'
    bib_begin = '\\begin{thebibliography}{0}'
    bib_end = '\\end{thebibliography}'
    document = None
    order = []
    order_dict = {}
    with open(filename, 'r') as file_input:
        document = file_input.read()
    for item in document.split('\\cite{')[1:]:
        name = item[:item.find('}')]
        if name not in order:
            order.append(name)
            order_dict[name] = ''
    
    print(order)
    line_triplets = []
    pos = 0
    tmp_string = ''
    for line in document[document.find(bib_begin) + len(bib_begin):document.find(bib_end)].split('\n')[1:]:
        if pos == 3:
            line_triplets.append(tmp_string)
            pos = 0
            tmp_string = ''
        if pos < 3:
            tmp_string += line + '\n'
            pos += 1
    line_triplets.append(tmp_string)
    
    for i in range(len(line_triplets)):
        order_dict[line_triplets[i][line_triplets[i].find('{') + 1:line_triplets[i].find('}')]] = line_triplets[i]
    for elem in order:
        print(order_dict[elem])


if __name__ == '__main__':
    main()
