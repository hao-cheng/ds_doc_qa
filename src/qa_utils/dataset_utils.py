"""This scripts is derived from official TriviaQA evaluation script."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import codecs
import json


def write_json_to_file(json_object, json_file, mode='wt'):
    with open(json_file, mode) as outfile:
        json.dump(
            json_object, outfile, indent=4, sort_keys=True, ensure_ascii=False)


def get_file_contents(filename, encoding='utf-8'):
    with open(filename) as f:
        content = f.read()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)


def get_file_contents_as_list(file_path, encoding='utf-8', ignore_blanks=True):
    contents = get_file_contents(file_path, encoding=encoding)
    lines = contents.split('\n')
    lines = [line for line in lines if line != ''] if ignore_blanks else lines
    return lines


# Key for wikipedia eval is question-id. Key for web eval is the (question_id, filename) tuple
def get_key_to_ground_truth(data):
    if data['Domain'] == 'Wikipedia':
        return {datum['QuestionId']: datum['Answer'] for datum in data['Data']}
    else:
        return get_qd_to_answer(data)


def get_question_doc_string(qid, doc_name):
    return '{}--{}'.format(qid, doc_name)


def get_qd_to_answer(data):
    key_to_answer = {}
    for datum in data['Data']:
        for page in datum.get('EntityPages', []) + datum.get('SearchResults', []):
            qd_tuple = get_question_doc_string(datum['QuestionId'], page['Filename'])
            key_to_answer[qd_tuple] = datum['Answer']
    return key_to_answer


def read_clean_part(datum):
    for key in ['EntityPages', 'SearchResults']:
        new_page_list = []
        for page in datum.get(key, []):
            if page['DocPartOfVerifiedEval']:
                new_page_list.append(page)
        datum[key] = new_page_list
    assert len(datum['EntityPages']) + len(datum['SearchResults']) > 0
    return datum


def read_triviaqa_data(qajson):
    """Reads triviaqa examples from the json file."""
    data = read_json(qajson)
    # Reads only documents and questions that are a part of clean data set
    if data['VerifiedEval']:
        clean_data = []
        for datum in data['Data']:
            if datum['QuestionPartOfVerifiedEval']:
                if data['Domain'] == 'Web':
                    datum = read_clean_part(datum)
                clean_data.append(datum)
        data['Data'] = clean_data
    return data


def answer_index_in_document(answer, document):
    answer_list = answer['NormalizedAliases']
    for answer_string_in_doc in answer_list:
        index = document.lower().find(answer_string_in_doc)
        if index != -1:
            return answer_string_in_doc, index
    return answer['NormalizedValue'], -1
