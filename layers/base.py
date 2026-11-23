from abc import ABC, abstractmethod


class GPTLayer(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass
