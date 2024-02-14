import os
import torch
from pathlib import Path

# from colbert.utils.runs import Run
from colbert.infra import ColBERTConfig
from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS
# from colbert.infra.run import Run

def find_last_checkpoint(path: str):
    return str(Path(path) / "colbert")

def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)


def manage_checkpoints(config: ColBERTConfig, colbert, optimizer, batch_idx, savepath=None, consumed_all_triples=False):
    # arguments = dict(args)

    # TODO: Call provenance() on the values that support it??

    # checkpoints_path = savepath or os.path.join(Run().path_, 'checkpoints')
    checkpoints_path = savepath or config.checkpoint_path_
    name = None

    try:
        save = colbert.save
    except:
        save = colbert.module.save

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    
    path_save = None

    if consumed_all_triples or (batch_idx % 2000 == 0):
        # name = os.path.join(path, "colbert.dnn")
        # save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
        path_save = os.path.join(checkpoints_path, "colbert")

    if batch_idx in SAVED_CHECKPOINTS:
        # name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        # save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
        path_save = os.path.join(checkpoints_path, f"colbert-{batch_idx}")

    if path_save:
        print(f"#> Saving a checkpoint to {path_save} ..")

        checkpoint = {}
        checkpoint['batch'] = batch_idx
        checkpoint['epoch'] = 0
        # checkpoint['model_state_dict'] = model.state_dict()
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        # checkpoint['config'] = config

        save(path_save)
        torch.save(checkpoint, os.path.join(path_save, "misc.bin"))

    return path_save

def load_checkpoint_misc(config: ColBERTConfig):
    if config.checkpoint.endswith('.dnn'):
        # lagacy ColBERT v1 checkpoints
        raw_checkpoint = torch.load(config.checkpoint, map_location='cpu')
        if config.fix_broken_optimizer_state and min(raw_checkpoint['optimizer_state_dict']['state'].keys()) != 0:
            print_message(f"Found potentially broken optimizer state -- trying to fix it")
            print_message(f"This fix is specific for ColBERT-X with XLMR-large checkpoints created at the early stage.")

            new_state = {
                i: raw_checkpoint['optimizer_state_dict']['state'][390+i]
                for i in range(1, 390)
            }
            new_state[0] = raw_checkpoint['optimizer_state_dict']['state'][782]
            raw_checkpoint['optimizer_state_dict']['state'] = new_state
            raw_checkpoint['optimizer_state_dict']['param_groups'][0]['params'] = list(range(0, 392))

        return raw_checkpoint['batch'], raw_checkpoint['optimizer_state_dict']

    misc_fn = Path(config.checkpoint) / "misc.bin"
    assert misc_fn.exists(), f"{misc_fn} does not exists. Can't resume."
    checkpoint_misc = torch.load(misc_fn, map_location=torch.device('cpu'))
    return checkpoint_misc['batch'], checkpoint_misc['optimizer_state_dict']