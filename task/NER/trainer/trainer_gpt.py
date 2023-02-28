from trainer.trainer_bert import NER_FedAvg_bert, trainer_bert
from models.GPT import GPTModel

class trainer_gpt(trainer_bert):
    pass

class NER_FedAvg_gpt(NER_FedAvg_bert):
    def generate_models(self):
        return GPTModel(num_labels = self.num_labels, model_name=self.model_name)