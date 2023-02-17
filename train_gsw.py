import typer
import torch
import gsw


def main(
        gsw_type: str,
        proj_type: str,
        lr: float = 1e-1,
        iterations: int = 10000,
        num_layer: int = 2,
        hidden_dim: int = 50,
        final_dim: int = 50,
        degree: int = 3,
        mgswd_lr: int = 1e-1,
        mgswd_iterations: int = 10,
        device: str = 'cuda:0'
):
    # SETUP Losses
    assert gsw_type in ('GSWD', 'MGSWD')
    assert proj_type in ('LinearProjector', 'NNProjector', 'PolyProjector')
    objective = getattr(gsw, gsw_type)
    projector_cls = getattr(gsw, proj_type)
    if gsw_type == 'GSWD':
        typer.echo('Running generalized sliced wasserstein')
        objective_kwargs = {}
    else:
        typer.echo('Running max generalized sliced wasserstein')
        objective_kwargs = {'lr': mgswd_lr, 'iterations': mgswd_iterations}
        final_dim = 1
        typer.echo(objective_kwargs)

    projector_kwargs = {}
    input_features = 2
    argcount = projector_cls.__init__.__code__.co_argcount
    for varname in projector_cls.__init__.__code__.co_varnames[:argcount]:
        if varname != 'self':
            projector_kwargs[varname] = locals()[varname]

    projector = projector_cls(**projector_kwargs)

    # Setup Data

    # Setup Model
    num_points = 50
    parameters = torch.randn(num_points, 2, device=device)
    parameters.requires_grad_(True)

    # Setup Training
    optimizer = torch.optim.Adam([parameters], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterations, eta_min=lr/100
    )
    projector.cuda(device)
    for iteration in range(iterations):
        optimizer.zero_grad()
        target = torch.rand_like(parameters.data)
        loss = objective(parameters, target, projector, **objective_kwargs)
        loss.backward()
        if iteration % 100 == 0:
            typer.echo(f'Loss {iteration}: {loss.item():.4f}')
        optimizer.step()
        scheduler.step()


if __name__ == '__main__':
    typer.run(main)
