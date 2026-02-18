Optimized for amd users, nvidia compatibility

Vae blend on cpu, decode 1 tile on gpu

Memory fixes, cache fixes, lora memory fixes, ui memory 'leak' and responsiveness fixes

Redone dependancy system using uv instead of pip, optimized package system, no more nvidia packages on amd cards

webui.sh changes designed for arch based distros, tested on cachyos/amd rx6700

run "bash webui.sh" with a fresh pull, drop in your models extensions and configs from reforge, do not drop in anything else specifically venv




# known issues
if invoke is open, you will get a mismatched tensor device error


a fresh pull will download a fork of generative-models into /repositories which fixes a seperate tensor on two device error
/repositories/generative-models/sgm/modules/encoders/modules.py

