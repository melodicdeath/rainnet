from datamodules import Rainnet_FMICompositeDataModule
from utils import load_config, setup_logging
from pyprojroot import here

home = here()

confpath = home / "config/example"
dsconf = load_config(confpath / "FMIComposite.yaml")
outputconf = load_config(confpath / "output.yaml")
modelconf = load_config(confpath / "rainnet.yaml")

# setup_logging(outputconf.logging)

datamodel = Rainnet_FMICompositeDataModule(dsconf, modelconf.train_params)
datamodel.setup("predict")
predict_dataloader = datamodel.predict_dataloader()

# iter(predict_dataloader).next()
for batch in predict_dataloader:
    print(batch)
    break
