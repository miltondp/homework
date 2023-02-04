import torch
from ray import tune, air
from ray.air import session
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from model import MLP
from main_tune import get_mnist_data, train


# 1. Wrap a PyTorch model in an objective function.
def objective(config):
    train_loader, test_loader = get_mnist_data()
    # train_loader.num_workers = 2
    # test_loader.num_workers = 2

    layers = [v for k, v in config.items() if k.startswith("layer_") and v > 1]
    print(layers)

    model = MLP(
        784,
        layers,
        10,
        config["activation"],
        torch.nn.init.xavier_uniform_,
    )

    while True:
        acc = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=10,
            learning_rate=0.001,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        # return {"mean_accuracy": acc}
        session.report({"mean_accuracy": acc})  # Report to Tune


if __name__ == "__main__":
    # 2. Define a search space and initialize the search algorithm.
    search_space = {
        "layer_0": tune.choice([2**i for i in range(0, 12)]),
        "layer_1": tune.choice([2**i for i in range(0, 12)]),
        "layer_2": tune.choice([2**i for i in range(0, 12)]),
        "layer_3": tune.choice([2**i for i in range(0, 12)]),
        "layer_4": tune.choice([2**i for i in range(0, 12)]),
        "activation": tune.choice(["ReLU", "Tanh", "Sigmoid"]),
    }

    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=20)

    # 3. Start a Tune run that maximizes mean accuracy and stops after 5 iterations.
    trainable_with_resources = tune.with_resources(objective, {"cpu": 1, "gpu": 0.15})
    tuner = tune.Tuner(
        trainable_with_resources,
        # objective,
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            search_alg=algo,
            num_samples=100,
        ),
        run_config=air.RunConfig(
            name="cis522_hw3",
            local_dir="data",
            stop={"training_iteration": 1},
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
