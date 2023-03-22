from math import e as exponential
import torch

__all__ = [
    "rane",
]


def rane(atomic_numbers, neighbors, r_ij):
    atom_num_exp = atomic_numbers.view(-1, atomic_numbers.size(1), 1)
    atom_nbh = atom_num_exp.expand(atom_num_exp.size(0), atom_num_exp.size(1), atom_num_exp.size(1))
    atom_nbh = torch.gather(atom_nbh, 1, neighbors)

    # atomic numbers modify with bond steps and then edit bond steps
    atom_mod = atom_num_exp * (atom_nbh + r_ij * 0.01)
    atom_mod_prod = atom_mod.clone()
    atom_mod_prod[atom_mod_prod == float(0)] = float(1)
    dist_edit = torch.exp(
        -0.01 * exponential * torch.log10(
            torch.pow(
                torch.prod(
                    atom_mod_prod, dim=2
                ), 1.01
            ) * torch.reciprocal(
                torch.pow(
                    torch.sum(
                        atom_mod, dim=2
                    ), 1.1
                )
            )
        )
    )
    dist_edit = dist_edit.unsqueeze(-1)
    atom_num_exp = atom_num_exp + dist_edit
    atom_nbh = atom_num_exp.expand(atom_num_exp.size(0), atom_num_exp.size(1), atom_num_exp.size(1))
    atom_nbh = torch.gather(atom_nbh, 1, neighbors)
    dist_edit = torch.exp(
        -0.4 * torch.exp(
            torch.pow(
                atom_num_exp * atom_nbh, 1.01
            ) * torch.reciprocal(
                torch.pow(
                    atom_num_exp + atom_nbh, 1.1
                )
            )
        )
    )

    # edit distances with atomic_numbers
    r_ij = r_ij + dist_edit
    return r_ij
