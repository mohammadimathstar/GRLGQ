import argparse
import numpy as np
import os

from util.grassmann import grassmann_repr
from util.glvq import *
from lvqmodel.prototypes import PrototypeLayer


LOW_BOUND_LAMBDA = 0.001

class Model(nn.Module):
    def __init__(self,
                 img_size: int,
                 num_classes: int,
                 args: argparse.Namespace,
                 device='cpu'
                 ):
        super().__init__()
        assert num_classes > 0

        self._num_classes = num_classes
        self._act_fun = args.cost_fun
        self._metric_type = 'geodesic'

        self._img_size = img_size
        self._num_prototypes = args.num_of_protos

        # create the prototype layers
        self.prototype_layer = PrototypeLayer(
            num_prototypes=self._num_prototypes,
            num_classes=self._num_classes,
            dim_of_data=self._img_size,
            dim_of_subspace=args.dim_of_subspace,
            metric_type=self._metric_type,
            device=device,
        )

    @property
    def prototypes_require_grad(self) -> bool:
        return self.prototype_layer.xprotos.requires_grad

    @prototypes_require_grad.setter
    def prototypes_require_grad(self, val: bool):
        self.prototype_layer.xprotos.requires_grad = val

    @property
    def features_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._net.parameters()])

    @features_require_grad.setter
    def features_require_grad(self, val: bool):
        for param in self._net.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_require_grad.setter
    def add_on_layers_require_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    def forward(self,
                xs: torch.Tensor,
                ):
        """
        Compute the distance between features (from Neural Net) and the prototypes
        xs: a batch of subspaces
        """

        # SVD decomposition: representation of net features as a point on the grassmann manifold
        xs_subspaces = grassmann_repr(xs, self.dim_of_subspaces)

        distance, Qw = self.prototype_layer(xs_subspaces) # SHAPE:(batch_size, num_prototypes, D: dim_of_data, d: dim_of_subspace)

        return distance, Qw

    def save(self, directory_path: str):
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        with open(directory_path + "/model.pth", 'wb') as f:
            torch.save(self, f)

    def save_state(self, directory_path: str):
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        with open(directory_path + "/model_state.pth", 'wb') as f:
            torch.save(self.state_dict(), f)

    @staticmethod
    def load(directory_path: str):
        return torch.load(directory_path + '/model.pth')




def return_model(fname):
    with np.load(fname + '.npz', allow_pickle=True) as f:
        xprotos, yprotos = f['xprotos'], f['yprotos']
        lamda = f['lamda']
        print(f"train accuracy: {f['accuracy_of_train_set'][-1]}, "
              f"\t validation accuracy: {f['accuracy_of_validation_set'][-1]} ({np.max(f['accuracy_of_validation_set'])})")

    return xprotos, yprotos, lamda
