import torch.nn as nn

from transformers import HubertModel

from module.content_encoder import ContentEncoder
from module.decoder import Decoder
from module.discriminator import Discriminator
from module.flow import Flow
from module.posterior_encoder import PosteriorEncoder
from module.speaker_encoder import SpeakerEncoder


class VoiceConvertor(nn.Module):
    def __init__(self):
        self.content_encoder = ContentEncoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()
        self.flow = Flow()
        self.posterior_encoder = import PosteriorEncoder()
        self.speaker_encoder = SpeakerEncoder()
        self.hubert = HubertModel.from_pretrained("rinna/japanese-hubert-base")
        self.hubert.eval()
        for p in self.hubert.parameters():
            p.requires_grad = False

