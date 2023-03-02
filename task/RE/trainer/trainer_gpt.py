from trainer.trainer_bert import RE_FedAvg_bert, trainer_bert
from models.GPT import GPTModel

class trainer_gpt(trainer_bert):
    pass

class RE_FedAvg_gpt(RE_FedAvg_bert):
    def generate_models(self):
        return GPTModel(num_labels = self.num_labels, model_name=self.model_name)