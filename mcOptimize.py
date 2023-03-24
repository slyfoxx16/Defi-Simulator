import itertools
import mcTrainer
from tqdm import tqdm
import torch.multiprocessing as mp

def find_best_model(data, hidden_layer_options=[1, 2, 3], batch_size_options=[1, 7, 14], n_epochs=100):
    mp.set_start_method("spawn", force=True)
    total_models = len(hidden_layer_options) * len(batch_size_options)
    best_mape = float("inf")
    best_model = None
    with mp.Pool(processes=mp.cpu_count()) as pool, tqdm(total=total_models) as pbar:
        results = []
        for hidden_layers, batch_size in itertools.product(hidden_layer_options, batch_size_options):
            result = pool.apply_async(mcTrainer.train_model, args=(data, hidden_layers, batch_size, n_epochs))
            results.append((hidden_layers, batch_size, result))
        for hidden_layers, batch_size, result in itertools.product(hidden_layer_options, batch_size_options, results):
            if result[2].ready():
                markov_chain = result[2].get()
                mape = markov_chain.score()
                if mape < best_mape:
                    best_mape = mape
                    best_model = markov_chain
                    
                pbar.update(1)
                pbar.set_postfix({"Hidden Layers": hidden_layers, "Batch Size": batch_size, "MAPE": f"{mape:.2f}%"})

    return best_model