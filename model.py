import torch.nn as nn

from module.decoder import Decoder
from module.discriminator import Discriminator
from module.flow import Flow
from module.posterior_encoder import PosteriorEncoder
from module.speaker_encoder import SpeakerEncoder
from module.content_extractor import from_intermediate_audio_file


class ConvertorModules(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.flow = Flow()
        self.posterior_encoder = PosteriorEncoder()
        self.speaker_encoder = SpeakerEncoder()
        self.content_extractor = from_intermediate_audio_file("intermediate.wav")


class VoiceConvertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = Discriminator()
        self.convertor = ConvertorModules()
