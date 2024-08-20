import torch
import datetime
import yaml
import time
import mimic_dataset


from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from net.dbkd import DBKBFramework
from evaluation import all_metrics, print_metrics
from log import logger

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} ran in: {end_time - start_time:.6f} seconds")
        return result
    return wrapper

def logitsToPred(logits, threshold):
    probabilities = torch.sigmoid(logits)
    predicted_labels = (probabilities >= threshold).float()
    return predicted_labels


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        metric,
        decision,
        device,
        save_path,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.decision = decision
        self.metric = metric
        self.device = device
        self.model.to(self.device)
        self.adj_param = 0.5
        self.save_path = save_path

    def _calc_logits_loss(self, x, y):
        logits = self.model(x)
        loss = self.criterion(logits, y)
        return logits, loss

    def train_step(self, x, y):
        self.model.train()  # Set model to training mode
        x, y = x.to(self.device), y.to(self.device)

        logits, loss = self._calc_logits_loss(x, y)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = all_metrics(logitsToPred(logits, self.decision), y)

        return loss.item(), metrics

    def validation(self, val_loader):
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        total_iter = len(val_loader)
        total_metrics = 0

        with torch.no_grad():  # Disable gradient computation for validation
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                logits, loss = self._calc_logits_loss(x, y)
                total_loss += loss.item()
                total_metrics += all_metrics(logitsToPred(logits, self.decision), y)[
                    self.metric
                ]

        avg_loss = total_loss / total_iter
        avg_metric = total_metrics / total_iter
        return avg_loss, avg_metric

    def fit(self, train_loader, val_loader, epochs, log_interval, val_interval):
        train_curve = list()
        iter_count = 0
        best_val_metric = 0

        writer = SummaryWriter(
            comment=datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"),
            filename_suffix="_train_curve",
        )

        continue_low_metric = 0

        for epoch in range(epochs):
            total_loss = 0
            loss_mean = 0.0
            metric = 0.0
            total = 0.0
            self.adj_param = 1 - (epoch / (epochs * 1.0)) ** 2

            # Training loop
            for x, y in train_loader:
                iter_count += 1
                loss, metrics = self.train_step(x, y)
                metric = metrics[self.metric]
                total += y.size(0)
                total_loss += loss
                train_curve.append(loss)
                loss_mean += loss

                if (iter_count) % log_interval == 0:
                    loss_mean = loss_mean / log_interval
                    logger.info(
                        "Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Metric:{:.2%}".format(
                            epoch,
                            epochs,
                            iter_count,
                            len(train_loader),
                            loss_mean,
                            metric,
                        )
                    )
                    loss_mean = 0.0

                writer.add_scalars("Loss", {"Train": loss}, iter_count)
                writer.add_scalars("Metric", {"Train": metric}, iter_count)

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(name + "_grad", param.grad, epoch)
                    writer.add_histogram(name + "_data", param, epoch)

            if (epoch % val_interval) == 0:
                val_loss, metric = self.validation(val_loader)
                logger.info(
                    "Val:Epoch[{:0>3}/{:0>3}] Loss: {:.4f} Metric:{:.2%}".format(
                        epoch,
                        epochs,
                        val_loss,
                        metric,
                    )
                )
                if metric > best_val_metric:
                    best_val_metric = metric
                    logger.info("Val:Best-Metric: {:.2%}".format(best_val_metric))
                    torch.save(self.model.state_dict(), self.save_path)
                else:
                    continue_low_metric += 1
                    if continue_low_metric > 20:
                        logger.info("early stopping")
                        return


class Evaluator:
    def __init__(self, model, params_path, decision, device):
        self.model = model
        self.model.load_state_dict(torch.load(params_path))
        self.model.eval()
        self.decision = decision
        self.device = device

    def _calc_logits(self, x, y):
        return self.model(x)
    
    @timer
    def eval(self, test_loader):
        total = len(test_loader)
        metrics = {}
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self._calc_logits(x, y)
            _metrics = all_metrics(logitsToPred(logits, self.decision), y)
            metrics = {key: (_metrics[key] + metrics.get(key, 0)) for key in _metrics}

        metrics = {key: metrics[key] / total for key in metrics}
        print_metrics(metrics)


class DBKDTrainer(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        metric,
        decision,
        device,
        save_path,
    ):
        super().__init__(
            model, optimizer, criterion, metric, decision, device, save_path
        )

    def _calc_logits_loss(self, x, y):
        logits, loss = self.model(x, y, self.adj_param)
        return logits, loss


class FBTrainer(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        metric,
        decision,
        device,
        save_path,
    ):
        super().__init__(
            model, optimizer, criterion, metric, decision, device, save_path
        )

    def _calc_logits_loss(self, x, y):
        _, logits = self.model(x)
        loss = self.criterion(logits, y)
        return logits, loss


class DBKDEvaluator(Evaluator):
    def __init__(self, model, params_path, decision, device):
        super().__init__(model, params_path, decision, device)

    def _calc_logits(self, x, y):
        logits, _ = self.model(x, y, 0.0)
        return logits


class FBEvaluator(Evaluator):
    def __init__(self, model, params_path, decision, device):
        super().__init__(model, params_path, decision, device)

    def _calc_logits(self, x, y):
        _, logits = self.model(x)
        return logits


def train_dbkd():
    config = {}
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    # Model
    model = DBKBFramework(config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = 0  # unused
    metric = "f1_micro"
    decision = config["cls_threshold"]
    save_path = config["persistent"]["dbkd"]

    # Data
    train_data = mimic_dataset.MimicData(config=config, stage="train")
    train_loader = DataLoader(
        dataset=train_data, batch_size=config["batch_size"], shuffle=True
    )
    logger.info("train data size: {}, iter: {}".format(len(train_data),  len(train_loader)))
    val_data = mimic_dataset.MimicData(config=config, stage="dev")
    val_loader = DataLoader(
        dataset=val_data, batch_size=config["batch_size"], shuffle=True
    )

    logger.info("val data size: {}, iter: {}".format(len(val_data),  len(val_loader)))

    trainer = DBKDTrainer(
        model, optimizer, criterion, metric, decision, device, save_path
    )
    trainer.fit(
        train_loader,
        val_loader,
        config["epochs"],
        config["log_interval"],
        config["val_interval"],
    )


def eval_dbkd():
    config = {}
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    # Data
    test_data = mimic_dataset.MimicData(config=config, stage="test")
    test_loader = DataLoader(
        dataset=test_data, batch_size=config["batch_size"], shuffle=False
    )
    logger.info("test data size: {}, iter: {}".format(len(test_data),  len(test_loader)))
    # Model
    model = DBKBFramework(config)
    model = model.to(device)
    decision = config["cls_threshold"]

    save_path = config["persistent"]["dbkd"]
    evt = DBKDEvaluator(model, save_path, decision, device)
    evt.eval(test_loader)


def train_feat():
    from net.flb import FeatureBranch

    config = {}
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    # Model
    model = FeatureBranch(config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = torch.nn.BCEWithLogitsLoss()  # unused
    metric = "f1_micro"
    decision = config["cls_threshold"]
    save_path = config["persistent"]["feat"]

    # Data
    train_data = mimic_dataset.MimicData(config=config, stage="train")
    train_loader = DataLoader(
        dataset=train_data, batch_size=config["batch_size"], shuffle=True
    )
    
    val_data = mimic_dataset.MimicData(config=config, stage="dev")
    val_loader = DataLoader(
        dataset=val_data, batch_size=config["batch_size"], shuffle=True
    )

    trainer = FBTrainer(
        model, optimizer, criterion, metric, decision, device, save_path
    )
    trainer.fit(
        train_loader,
        val_loader,
        config["epochs"],
        config["log_interval"],
        config["val_interval"],
    )


def eval_feat():
    from net.flb import FeatureBranch

    config = {}
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    # Data
    test_data = mimic_dataset.MimicData(config=config, stage="test")
    test_loader = DataLoader(
        dataset=test_data, batch_size=config["batch_size"], shuffle=False
    )
    # Model
    model = FeatureBranch(config)
    model = model.to(device)
    decision = config["cls_threshold"]

    save_path = config["persistent"]["feat"]
    evt = FBEvaluator(model, save_path, decision, device)
    evt.eval(test_loader)


def train_clas():
    from net.clb import ClassifierBranch
    from net.bhl import BHLLoss

    config = {}
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    # Model
    model = ClassifierBranch(config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = BHLLoss(config)
    metric = "f1_micro"
    decision = config["cls_threshold"]
    save_path = config["persistent"]["clas"]

    # Data
    train_data = mimic_dataset.MimicData(config=config, stage="train")
    train_loader = DataLoader(
        dataset=train_data, batch_size=config["batch_size"], shuffle=True
    )
    val_data = mimic_dataset.MimicData(config=config, stage="dev")
    val_loader = DataLoader(
        dataset=val_data, batch_size=config["batch_size"], shuffle=True
    )

    trainer = Trainer(model, optimizer, criterion, metric, decision, device, save_path)
    trainer.fit(
        train_loader,
        val_loader,
        config["epochs"],
        config["log_interval"],
        config["val_interval"],
    )


def eval_clas():
    from net.clb import ClassifierBranch

    config = {}
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    # Data
    test_data = mimic_dataset.MimicData(config=config, stage="test")
    
    test_loader = DataLoader(
        dataset=test_data, batch_size=config["batch_size"], shuffle=False
    )
    # Model
    model = ClassifierBranch(config)
    model = model.to(device)
    decision = config["cls_threshold"]

    save_path = config["persistent"]["clas"]
    evt = Evaluator(model, save_path, decision, device)
    evt.eval(test_loader)
