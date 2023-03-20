import json
import datasets
import pdb
from promptsource.templates import DatasetTemplates


def load_contrast_data(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    
    data = data['data'][1:] # the first one is just the header

    my_dataset = []
    answer_map = {'FALSE': False, 'TRUE': True}
    for i, dt in enumerate(data):
        for dic in dt['perturbed_questions']:
            try:
                example = {"set_id": i, "passage": dt['paragraph'],
                    "question": dic['perturbed_q'], "answer": answer_map[dic['answer']]}
                my_dataset.append(example)
            except:
                #pdb.set_trace()
                pass

    print('# contrast set:', len(data), '# examples:', len(my_dataset))
    return my_dataset

def apply_prompts(my_dataset, template='exercise'):
    prompts = DatasetTemplates('super_glue/boolq')
    prompt = prompts[template]

    task_name = 'contrast_boolq'
    def _apply(dp):
        label_map = {False: 'no', True: 'yes'}
        new_dp = {'task': task_name, "set_id": dp["set_id"]}
        text = prompt.apply(dp)[0]
        new_dp["input"] = text.replace("answer the question by True or False", "answer the question by yes or no")
        new_dp["output"] = label_map[dp['answer']]
        new_dp["options"] = ["no", "yes"]
        return new_dp

    with open(f'{task_name}_500_0_test.jsonl', "w") as fout:
        for dp in my_dataset:
            line = _apply(dp)
            fout.write(json.dumps(line)+"\n")


if __name__ == "__main__":
    dataset = load_contrast_data('boolq_perturbed.json')
    apply_prompts(dataset)

