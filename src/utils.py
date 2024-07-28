from vllm import LLM, SamplingParams
import torch

class LMToolkit:
    
    def __init__(self, model : LLM, max_new_tokens : int = 1024, topk : int = 10):
        """ 
            Args:
                model: The model is expected to be an instance of a vLLM LLM.
                
        """
        
        self.model = model
        self.tokenizer = self.model.llm_engine.tokenizer.tokenizer
        self.max_new_tokens = max_new_tokens
        self.topk = topk
        
        self.model.llm_engine.model_config.max_logprobs = self.topk + 1
        
    @torch.inference_mode
    def get_completion(self, input_string : str, get_scores = False):
        """
            Args:
                input_string: The input string to the model.
                
            Returns:
                The completion of the input string.
        """
        
        if get_scores:
            sampling_params = SamplingParams(
                n = 1, 
                temperature = 0, 
                top_p=1,
                max_tokens=self.max_new_tokens,
                logprobs=self.topk
            )
        else:
            sampling_params = SamplingParams(
                n = 1, 
                temperature = 0, 
                top_p=1,
                max_tokens=self.max_new_tokens
            )
        
        outputs = self.model.generate(
            input_string, 
            sampling_params, 
            use_tqdm = False,
        )
        
        if get_scores:
            scores = []
            for logprob in outputs[0].outputs[0].logprobs:
                scores.append(LMToolkit.vllm_logprob_to_score(logprob))
        
        
        return (outputs[0].outputs[0].text, scores) if get_scores else outputs[0].outputs[0].text
    
    @torch.inference_mode()
    def get_next_topk_tokens(self, input_string : str, drop_objects = True):
        """
            Args:
                input_string: The input string to the model.
                
            Returns:
                A list of tuples. Each tuple contains a token and its log probability.
        """
        
        sampling_params = SamplingParams(
            n = 1, 
            temperature = 0, 
            top_p=1,
            max_tokens=1,
            logprobs=self.topk    
        )
        
        outputs = self.model.generate(
            input_string, 
            sampling_params, 
            use_tqdm = False
        )[0].outputs[0].logprobs[0]

        return LMToolkit.vllm_logprob_to_score(outputs, drop_objects)
        
    @staticmethod
    def vllm_logprob_to_score(logprobs, drop_objects = True):
        topk_tokens = {'decoded': [], 'probs': [], 'token_id': [], 'logprobs': None if drop_objects else []}
        
        for token_id, logprob_obj in logprobs.items():
            if not drop_objects:
                topk_tokens['logprobs'].append({token_id: logprob_obj})
            topk_tokens['decoded'].append(logprob_obj.decoded_token)
            topk_tokens['probs'].append(logprob_obj.logprob)
            topk_tokens['token_id'].append(token_id)

        topk_tokens['probs'] = torch.exp(torch.tensor(topk_tokens['probs'])).tolist()
        
        return topk_tokens
        
        
        
        
        
    
