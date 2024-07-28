from dotenv import load_dotenv
from src.experiments import Experiment1_1

load_dotenv()

results_save_path = "./results/serialized/"
run_experiment_1 = True

if run_experiment_1:
    print("Running experiment 1")
    experiment1 = Experiment1_1(path_to_gsm8k="./data/gsm8k", model="google/gemma-2b")
    accuracy, observations = experiment1.run(checkpoint_savepath="./results/serialized/")
    print("Experiment 1 has concludede with an accuracy of", accuracy)
    
    
    