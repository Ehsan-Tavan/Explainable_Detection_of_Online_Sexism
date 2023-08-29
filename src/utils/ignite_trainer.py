from functools import partial
import torch
import torch.utils.data as Data
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar

from .helper import F1Score


class IgniteTrainer:
    def __init__(self, model, graph, device, train_iterator, valid_iterator, checkpoint_path,
                 class_weights=None):
        self.model = model
        self.device = device
        self.graph = graph
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.checkpoint_path = checkpoint_path
        self.best_f1_score = float("-inf")
        self.model_path = ""
        self.graph_path = ""

        self.trainer = None
        self.train_evaluator = None
        self.validation_evaluator = None
        self.progress_bar = None

        self.criterion = self.configure_losses(class_weights=class_weights)
        self.optimizer = self.configure_optimizers()

    def setup(self):
        self.configure_trainer_engine()
        self.configure_train_evaluator()
        self.configure_validation_evaluator()
        self.configure_early_stopping()

    def configure_trainer_engine(self):
        self.trainer = Engine(self.train_step)
        RunningAverage(output_transform=lambda x: x).attach(self.trainer, "loss")
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._update_feature)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_training_results)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_validation_results)
        self.progress_bar = ProgressBar(persist=True, bar_format="")
        self.progress_bar.attach(self.trainer, ["loss"])
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.save_best_epoch_only)

    def configure_train_evaluator(self):
        eval_step = partial(self.eval_step, mask="train_mask")
        self.train_evaluator = Engine(eval_step)
        Accuracy(output_transform=self.threshold_output_transform).attach(self.train_evaluator,
                                                                          "accuracy")

        F1Score(output_transform=self.threshold_output_transform).attach(self.train_evaluator,
                                                                         "f1_score")
        Loss(self.criterion).attach(self.train_evaluator, "ce")

    def configure_validation_evaluator(self):
        eval_step = partial(self.eval_step, mask="val_mask")
        self.validation_evaluator = Engine(eval_step)

        Accuracy(output_transform=self.threshold_output_transform).attach(self.validation_evaluator,
                                                                          "accuracy")
        F1Score(output_transform=self.threshold_output_transform).attach(self.validation_evaluator,
                                                                         "f1_score")
        Loss(self.criterion).attach(self.validation_evaluator, "ce")

    def configure_early_stopping(self, patience=5):
        handler = EarlyStopping(patience=patience, score_function=self.score_function,
                                trainer=self.trainer)
        self.validation_evaluator.add_event_handler(Events.COMPLETED, handler)

    @staticmethod
    def threshold_output_transform(output):
        y_pred, y = output
        y_pred = torch.argmax(y_pred, -1)
        return y_pred, y

    @staticmethod
    def score_function(engine):
        val_loss = engine.state.metrics["ce"]
        return -val_loss

    def train_step(self, engine, batch):
        self.model.train()
        (batch,) = [x.to(self.device) for x in batch]
        self.optimizer.zero_grad()
        train_mask = self.graph.train_mask[batch].type(torch.BoolTensor)
        y_pred = self.model(batch)[train_mask]
        y_true = self.graph.y[batch][train_mask]
        loss = self.criterion(y_pred, y_true)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()

    def eval_step(self, engine, batch, mask):
        self.model.eval()
        with torch.no_grad():
            (batch,) = [x.to(self.device) for x in batch]
            val_mask = self.graph[mask][batch].type(torch.BoolTensor)
            y_pred = self.model(batch)[val_mask]
            y_true = self.graph.y[batch][val_mask]
        return y_pred, y_true

    def log_training_results(self, engine):
        self.train_evaluator.run(self.train_iterator)
        metrics = self.train_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_f1_score = metrics["f1_score"]
        avg_ce = metrics["ce"]
        self.progress_bar.log_message(
            f"Training Results - Epoch: {engine.state.epoch}  "
            f"Avg accuracy: {avg_accuracy:.2f} Avg F1 Score : {avg_f1_score:.2f} "
            f"Avg loss: {avg_ce:.2f}")

    def log_validation_results(self, engine):
        self.validation_evaluator.run(self.valid_iterator)
        metrics = self.validation_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_f1_score = metrics["f1_score"]
        avg_bce = metrics["ce"]
        self.progress_bar.log_message(
            f"Validation Results - Epoch: {engine.state.epoch}  "
            f"Avg accuracy: {avg_accuracy:.2f} Avg F1 Score : {avg_f1_score:.2f} "
            f"Avg loss: {avg_bce:.2f}")
        self.progress_bar.n = self.progress_bar.last_print_n = 0

    def configure_optimizers(self):
        return torch.optim.Adam([
            {"params": self.model.lm_model.parameters(), "lr": 1e-5},
            {"params": self.model.gnn_model.parameters(), "lr": 1e-3},
        ], lr=1e-3
        )

    def configure_losses(self, class_weights):
        if class_weights.any():
            return torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights)).to(self.device)
        return torch.nn.CrossEntropyLoss().to(self.device)

    def _update_feature(self):
        doc_mask = self.graph.train_mask + self.graph.val_mask + self.graph.test_mask
        # no gradient needed, uses a large batchsize to speed up the process
        dataloader = Data.DataLoader(
            Data.TensorDataset(self.graph.input_ids, self.graph.attention_mask),
            batch_size=1024
        )
        with torch.no_grad():
            self.model.eval()
            nodes_features = []
            for index, batch in enumerate(dataloader):
                input_ids, attention_mask = [x.to(self.device) for x in batch]
                output = self.model.lm_model(input_ids=input_ids,
                                             attention_mask=attention_mask).pooler_output
                nodes_features.append(output)
            nodes_features = torch.cat(nodes_features, axis=0)
        self.graph.x = nodes_features
        print("Node features are updated")

    def save_best_epoch_only(self, engine):
        epoch = engine.state.epoch

        metrics = self.validation_evaluator.state.metrics
        if metrics["f1_score"] > self.best_f1_score:
            self.best_f1_score = metrics["f1_score"]
            self.model_path = f"../assets/saved_models/gnn/model_epoch_{epoch}_val_f1_score_" \
                              f"{self.best_f1_score:.4f}.pt"
            self.graph_path = f"../assets/saved_models/gnn/model_epoch_{epoch}_graph.pt"
            torch.save(self.model, self.model_path)
            torch.save(self.graph, self.graph_path)
