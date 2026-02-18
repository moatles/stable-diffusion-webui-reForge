Optimized for amd users, nvidia compatibility
Vae blend on cpu, decode 1 tile on gpu
Memory fixes, cache fixes, lora memory fixes, ui memory 'leak' and responsiveness fixes
Redone dependancy system using uv instead of pip, optimized package system, no more nvidia packages on amd cards
webui.sh changes designed for arch based distros, tested on cachyos/amd rx6700

run "bash webui.sh" with a fresh pull, drop in your models extensions and configs from reforge, do not drop in anything else specifically venv

# known issues
if invoke is open, you will get a mismatched tensor device error


a fresh pull will download generative-models into /repositories
edit

/repositories/generative-models/sgm/modules/encoders/modules.py at line 165
```py
                ):
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    target_device = output[out_key].device
                    output[out_key] = torch.cat(
                        (output[out_key], emb.to(target_device)), self.KEY2CATDIM[out_key]
                    )
                else:
                    output[out_key] = emb
```

This will eliminate another tensor device error
