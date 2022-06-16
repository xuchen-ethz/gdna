import pytorch_lightning as pl
import hydra
import torch
import wandb
import yaml
import os

from lib.gdna_model import BaseModel
from lib.dataset.datamodule import DataModule, DataProcessor

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

@hydra.main(config_path="config", config_name="config")
def main(opt):

    print(opt.pretty())

    pl.seed_everything(42, workers=True)

    torch.set_num_threads(10)

    callbacks = []

    datamodule = DataModule(opt.datamodule)
    datamodule.setup(stage='fit')

    meta_info = datamodule.meta_info

    data_processor = DataProcessor(opt.datamodule)

    with open('.hydra/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger = pl.loggers.WandbLogger(project='gdna', 
                                    name=opt.expname, 
                                    config=config, 
                                    offline=False, 
                                    resume=True, 
                                    settings=wandb.Settings(start_method='fork'))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=None,
                                                        monitor=None, 
                                                        dirpath='./checkpoints',
                                                        save_last=True,
                                                        every_n_val_epochs=1)
    callbacks.append(checkpoint_callback)

    checkpoint_path = './checkpoints/last.ckpt'
    if not (os.path.exists(checkpoint_path) and opt.resume):
        checkpoint_path = None
    
    trainer = pl.Trainer(logger=logger, 
                        callbacks=callbacks,
                        resume_from_checkpoint=checkpoint_path,
                        **opt.trainer)

    model = BaseModel(opt=opt.model, 
                    meta_info=meta_info,
                    data_processor=data_processor,
                    )

    starting_path = hydra.utils.to_absolute_path(opt.starting_path)
    if os.path.exists(starting_path) and checkpoint_path is None:
        model = model.load_from_checkpoint(starting_path, 
                                            strict=False,
                                            opt=opt.model, 
                                            meta_info=meta_info,
                                            data_processor=data_processor)

    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    main()