import torch.nn as nn
from lvqmodel.prototypes_gradients import *
from util.grassmann import init_randn


def rotate_prototypes(xprotos, rotation_matrix, winner_ids):
    assert xprotos.ndim == 3, f"data should be of shape (nprotos, dim_of_data, dim_of_subspace), but it is of shape {xprotos.shape}"
    assert winner_ids.shape[
               1] == 2, f"There should only be two winners W^+- prototypes for each data. But now there are {winner_ids.shape[1]} winners."

    nbatch, nprotos = rotation_matrix.shape[:2]

    Qwinners = rotation_matrix[torch.arange(nbatch).unsqueeze(-1), winner_ids]  # shape: (batch_size, 2, d, d)
    Qwinners1, Qwinners2 = Qwinners[:, 0], Qwinners[:, 1]
    assert Qwinners1.shape[0] == nbatch, f"The size of Qwinner should be (nbatch, ...) but it is {Qwinners1.shape}"

    xprotos_winners = xprotos[winner_ids]
    xprotos1, xprotos2 = xprotos_winners[:, 0], xprotos_winners[:, 1]

    rotated_proto1 = torch.bmm(xprotos1, Qwinners1.to(xprotos1.dtype))
    rotated_proto2 = torch.bmm(xprotos2, Qwinners2.to(xprotos1.dtype))
    return rotated_proto1, rotated_proto2

class PrototypeLayer(nn.Module):
    def __init__(self,
                 num_prototypes,
                 num_classes,
                 dim_of_data,
                 dim_of_subspace,
                 metric_type='geodesic',
                 dtype=torch.float32,
                 device='cpu'
                ):
        super().__init__()
        self.nchannels = dim_of_data
        self.dim_of_subspaces = dim_of_subspace

        # Each prototype is a latent representation of shape (D, d)
        self.xprotos, self.yprotos, self.yprotos_mat, self.yprotos_comp_mat = init_randn(
            self.nchannels,
            self.dim_of_subspaces,
            num_of_protos=num_prototypes,
            num_of_classes=num_classes,
            device=device,
        )

        self.metric_type = metric_type
        self.number_of_prototypes = self.yprotos.shape[0]

        self.relevances = nn.Parameter(
            torch.ones((
                1, self.xprotos.shape[-1]), dtype=dtype, device=device
            ) / self.xprotos.shape[-1]
        )


    def forward(self, xs_subspace):

        return GeodesicPrototypeLayer.apply(
                xs_subspace,
                self.xprotos,
                self.relevances,
            )
