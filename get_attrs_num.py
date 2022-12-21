def get_attrs_predicate_num(input_file_path_1, input_file_path_2):
    attrs_predicate_set_1 = set()
    attrs_predicate_set_2 = set()
    with open(input_file_path_1, 'r', encoding='utf-8') as input_f1, open(input_file_path_2, 'r', encoding='utf-8') as input_f2:
        for input_line1 in input_f1.readlines():
            line_list = input_line1.strip().split('\t')
            attrs_predicate_set_1.add(line_list[1])
        for input_line2 in input_f2.readlines():
            line_list = input_line2.strip().split('\t')
            attrs_predicate_set_2.add(line_list[1])
    print('kg1 have {}'.format(len(attrs_predicate_set_1)), 'attrs predicate')
    print('kg2 have {}'.format(len(attrs_predicate_set_2)), 'attrs predicate')


def get_attrs_triples_num(input_file_path_1, input_file_path_2):
    attrs_triples_1_num = 0
    attrs_triples_2_num = 0
    with open(input_file_path_1, 'r', encoding='utf-8') as input_f1, open(input_file_path_2, 'r', encoding='utf-8') as input_f2:
        for input_line1 in input_f1.readlines():
            # line_list = input_line1.strip().split('\t')
            attrs_triples_1_num += 1
        for input_line2 in input_f2.readlines():
            # line_list = input_line2.strip().split('\t')
            attrs_triples_2_num += 1
    print('kg1 have {}'.format(attrs_triples_1_num), 'attrs triples')
    print('kg2 have {}'.format(attrs_triples_2_num), 'attrs triples')

if __name__ == '__main__':
    input_file_path_1 = './data/srprs15k-mo/dbp_yg/training_attrs_1'
    input_file_path_2 = './data/srprs15k-mo/dbp_yg/training_attrs_2'
    get_attrs_predicate_num(input_file_path_1, input_file_path_2)
    get_attrs_triples_num(input_file_path_1, input_file_path_2)
