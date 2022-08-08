import pytorch_lightning as pl 

from project.data_modules.datamodule import DataModuleClass

class AmyloidPredictor(pl.LightningModule):
    def __init__(self):
        super(model, self).__init__()
        # define NN here 
        
    def forward(self, x):
    
    def configure_optimizers(self):
        
    def training_step(self, train_batch, batch_idx):
    
    def validation_step(self, valid_batch, batch_idx):
    

def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AmyloidPredictor.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    amyloid_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    amyloid_train, amyloid_val = random_split(dataset, [55000, 5000])
    
    train_loader = DataModuleClass.train_dataloader()
    val_loader = DataModuleClass.valid_dataloader()
    test_loader = DataModuleClass.test_dataloader() 

    ''' 
    train_loader = DataLoader(amyloid_train, batch_size=args.batch_size)
    val_loader = DataLoader(amyloid_val, batch_size=args.batch_size)
    test_loader = DataLoader(amyloid_test, batch_size=args.batch_size)
    '''

    # ------------
    # model
    # ------------
    model = AmyloidPredictor( Backbone(hidden_dim=args.hidden_dim), args.learning_rate )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()
        
    