import torch
from flagai.data.dataset.mm.clip_dataset import CsvDataset, clip_transform, collate_fn
from flagai.trainer import Trainer
from flagai.auto_model.auto_loader import AutoLoader

from load_data import get_wds_dataset
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cd examples/clip
# data_path = "./data/pairs.csv"
# img_dir = "./data/img"

trainer = Trainer(env_type="pytorch",
                  epochs=5,
                  pytorch_device=device,
                  batch_size=1,
                  lr=1e-4,
                  fp16=True,
                  log_interval=10,
                  num_gpus=2,
                  hostfile='./hostfile',
                  training_script=__file__,
                  deepspeed_config='./deepspeed.json'
                  )

loader = AutoLoader(task_name="txt_img_matching",#contrastive learning
                    model_name="EVA-CLIP",
                    )
model = loader.get_model()
tokenizer = loader.get_tokenizer()
transform = clip_transform(img_size=224)

ds = get_wds_dataset("/home/shunxing1234/fork/data/00000.tar", 
                        preprocess_img=transform, 
                        tokenizer=tokenizer, 
                        is_train=True, 
                        epoch=1)

dl = ds.dataloader


# train_dataset = CsvDataset(data_path,
#                             img_dir,
#                             transform,
#                             tokenizer)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
pdb.set_trace()
trainer.train(model,
              optimizer=optimizer,
              train_dataset=dl,
              collate_fn=collate_fn)

