from vllm import LLM, SamplingParams
from tqdm import tqdm
import pickle

from src.utils import LMToolkit
from src.datasets import GSM8K

class Experiment1:
    
    def __init__(self, path_to_gsm8k, model = "google/gemma-2b", gpu_memory_utilization = 0.98, max_new_tokens = 250, max_model_len = 8000):
        self.path_to_gsm8k = path_to_gsm8k
        self.gsm8k = GSM8K(path_to_gsm8k)
        
        # Setting up the models
        self.model = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
        self.toolkit = LMToolkit(self.model, max_new_tokens = max_new_tokens)
        
    
    def run(self, checkpoint_savepath, checkpoint_step = 20, printstep = 20, truncate_examples_to = None):
        if truncate_examples_to is None:
            truncate_examples_to = len(self.gsm8k)
                
        number_of_correct_answers = 0
        observations = []
        for i in tqdm(range(truncate_examples_to)):
            try:
                observation = {
                    'number': 0,
                    'question': self.gsm8k[i]['question'],
                    'completion': '',
                    'answer': '',
                    'correct_answer': '',
                    'correct': False,
                    'token_scores': None
                }
                
                observation['number'] = i
                question = self.gsm8k[i]['question']

                
                completion = self.toolkit.get_completion(question)
                
                answer = float(GSM8K.maybe_remove_comma(GSM8K.find_number(completion)))
                correct_answer = float(GSM8K.maybe_remove_comma(GSM8K.find_number(self.gsm8k[i]['answer'])))
                
                observation['completion'] = completion
                observation['answer'] = answer
                observation['correct_answer'] = correct_answer
                observation['correct'] = (correct_answer == answer)
                observation['token_scores'] = self.toolkit.get_next_topk_tokens(question)
                
                number_of_correct_answers += observation['correct']
                observations.append(observation)
                
                
                if (i + 1) % checkpoint_step == 0:
                    with open(checkpoint_savepath + f"expr1_checkpoint_{i + 1}.pkl", "wb") as f:
                        pickle.dump(observations, f)
                        
                if (i + 1) % printstep == 0:
                    print("Accuracy so far: ", number_of_correct_answers / (1 + i))
                    # print("\n\n\n\nQ. ", question)
                    # print("A. ", self.gsm8k[i]['answer'])
                    # print("C. ", completion)
                    # print("D.", observation['correct'])
            except Exception as e:
                print(f"Example {i} failed:\n {e}")                
        with open(checkpoint_savepath + f"expr1_checkpoint_final.pkl", "wb") as f:
                    pickle.dump(observations, f)
                    
        return number_of_correct_answers / truncate_examples_to, observations
            
class Experiment1_1:
    
    def __init__(self, path_to_gsm8k, model = "google/gemma-2b", gpu_memory_utilization = 0.98, max_new_tokens = 250, max_model_len = 8000):
        self.path_to_gsm8k = path_to_gsm8k
        self.gsm8k = GSM8K(path_to_gsm8k)
        
        # Setting up the models
        self.model = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
        self.toolkit = LMToolkit(self.model, max_new_tokens = max_new_tokens)
        
    
    def run(self, checkpoint_savepath, checkpoint_step = 20, printstep = 20, truncate_examples_to = None):
        if truncate_examples_to is None:
            truncate_examples_to = len(self.gsm8k)
                
        number_of_correct_answers = 0
        observations = []
        for i in tqdm(range(truncate_examples_to)):
            try:
                observation = {
                    'number': 0,
                    'question': self.gsm8k[i]['question'],
                    'completion': '',
                    'answer': '',
                    'correct_answer': '',
                    'correct': False,
                    'token_scores': None
                }
                
                observation['number'] = i
                question = self.gsm8k[i]['question']

                
                completion, logprobs = self.toolkit.get_completion(question, get_scores = True)
                
                answer = float(GSM8K.maybe_remove_comma(GSM8K.find_number(completion)))
                correct_answer = float(GSM8K.maybe_remove_comma(GSM8K.find_number(self.gsm8k[i]['answer'])))
                
                observation['completion'] = completion
                observation['answer'] = answer
                observation['correct_answer'] = correct_answer
                observation['correct'] = (correct_answer == answer)
                observation['token_scores'] = logprobs
                
                number_of_correct_answers += observation['correct']
                observations.append(observation)
                
                if (i + 1) % checkpoint_step == 0:
                    with open(checkpoint_savepath + f"expr1_1_checkpoint.pkl", "wb") as f:
                        pickle.dump(observations, f)
                        
                if (i + 1) % printstep == 0:
                    print("Accuracy so far: ", number_of_correct_answers / (1 + i))
                    # print("\n\n\n\nQ. ", question)
                    # print("A. ", self.gsm8k[i]['answer'])
                    # print("C. ", completion)
                    # print("D.", observation['correct'])
            except Exception as e:
                print(f"Example {i} failed:\n {e}")                
        with open(checkpoint_savepath + f"expr1_1_checkpoint_final.pkl", "wb") as f:
                    pickle.dump(observations, f)
                    
        return number_of_correct_answers / truncate_examples_to, observations
            
# class Experiment_2:
    # CoT decoding.
    
    