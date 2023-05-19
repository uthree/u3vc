import torch.nn as nn

from module.content_encoder import ContentEncoder
from module.decoder import Decoder
from module.discriminator import Discriminator
from module.flow import Flow
from module.posterior_encoder import PosteriorEncoder
from module.speaker_encoder import SpeakerEncoder


class ConvertorModules(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = ContentEncoder()
        self.decoder = Decoder()
        self.flow = Flow()
        self.posterior_encoder = PosteriorEncoder()
        self.speaker_encoder = SpeakerEncoder()


class VoiceConvertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = Discriminator()
        self.convertor = ConvertorModules()
