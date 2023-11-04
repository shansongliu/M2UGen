import os.path

import torch
import gc
from header import *
import numpy as np

from .anyToImageVideoAudio import MUGen


def adjust_learning_rate(optimizer, steps, base_lr=0.01, min_lr=0.0001, warmup_steps=0):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if steps < warmup_steps:
        lr = base_lr * steps / warmup_steps
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (steps - warmup_steps) / (steps - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class DeepSpeedAgent:

    def __init__(self, model :MUGen, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model
        self.model.set_default_trainability()
        # self.model.to("cuda")
        self.print_model_parameters()
        self.writer = SummaryWriter(args['log_path'])

        self.load_parameters(self.args['save_path'])
        # self.losses = []
        # self.mle_acces = []
        # load config parameters of deepspeed
        # ds_params = json.load(open(self.args['ds_config_path']))
        # ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        # ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(
        #     self.args['total_steps'] * self.args['warmup_rate']))
        self.optimizer = torch.optim.AdamW(self.model.get_trainable_params().values(), lr=0.1,
                                           betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0001)
        self.optimizer.zero_grad()
        # self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
        #     model=self.model,
        #     model_parameters=self.model.parameters(),
        #     config_params=ds_params,
        #     dist_init_required=True,
        #     args=types.SimpleNamespace(**args)
        # )

    @torch.no_grad()
    def predict(self):
        self.model.eval()
        output = self.model.generate(self.args)
        return output

    def train_model(self, batch, current_step=0, pbar=None):
        self.model.train()
        loss, mle_acc, mse_loss = self.model(batch)
        # loss /= 8
        loss.backward()
        # self.losses.append(loss.item())
        # self.losses = self.losses[-100:]
        # self.mle_acces.append(mle_acc)
        # self.mle_acces = self.mle_acces[-100:]
        self.writer.add_scalar('loss', loss, current_step)
        self.writer.add_scalar('mle_acc', mle_acc, current_step)
        # if isinstance(mse_loss, list):
        #     self.writer.add_scalar('img_mse_loss', mse_loss[0], current_step)
        #     self.writer.add_scalar('vid_mse_loss', mse_loss[1], current_step)
        #     self.writer.add_scalar('aud_mse_loss', mse_loss[2], current_step)
        if isinstance(mse_loss, torch.Tensor):
            self.writer.add_scalar('mse_loss', mse_loss, current_step)
        else:
            pass
        # self.writer.add_scalar('mse_loss', mse_loss, current_step)

        # if (current_step + 1) % 8 == 0:
        #     adjust_learning_rate(self.optimizer, current_step)
        #     self.optimizer.zero_grad()
        #     self.optimizer.step()
        # adjust_learning_rate(self.optimizer, current_step + 1)
        self.optimizer.zero_grad()
        self.optimizer.step()
        # self.ds_engine.backward(loss)
        # self.ds_engine.step()
        # pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; '
        #                      f'token_acc: {round(mle_acc * 100, 2)}; '
        #                      f'mse_loss: {round(mse_loss[0].item(), 4)} ')
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; '
                             f'token_acc: {round(mle_acc * 100, 2)}')
        pbar.update(1)
        if self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(
                f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
            # ; mse_loss: {round(mse_loss[0].item(), 4)}
        mle_acc *= 100

        loss = loss.detach()
        del batch, loss
        gc.collect()
        torch.cuda.empty_cache()
        return mle_acc

    def save_model(self, path, current_step):
        """
            this function also save the trainable parameters and specific name parameters
        """
        print(f'[!] saving model into {path}')
        # param_grad_dic = {
        #     k: v.requires_grad for (k, v) in self.model.named_parameters()
        # }
        state_dict = self.model.state_dict()
        checkpoint = OrderedDict()
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                checkpoint[k] = v
            if 'mu_mert' in k:
                checkpoint[k] = v
            if 'iu_vit' in k:
                checkpoint[k] = v
            if 'iu_vivit' in k:
                checkpoint[k] = v
            if 'gen_text_hidden_to_audio' in k:
                checkpoint[k] = v
        torch.save(checkpoint, f'{path}/pytorch_model.pt')
        # save tokenizer
        self.model.llama_tokenizer.save_pretrained(path)
        # save configuration
        self.model.llama_model.config.save_pretrained(path)
        print(f'[!] save model into {path}')

    def print_model_parameters(self, use_4bit=False):
        """
            Prints the number of trainable parameters in the model.
            """
        trainable_params = 0
        all_param = 0
        lora = 0
        image = 0
        video = 0
        audio = 0
        linear = 0
        llama = 0
        imagebind = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if 'lora' in name:
                lora += num_params
            elif 'iu_vivit' in name and param.requires_grad:
                video += num_params
            elif 'mu_mert' in name and param.requires_grad:
                audio += num_params
            elif 'iu_vit' in name and param.requires_grad:
                image += num_params
            elif 'gen_text_hidden_fcs_audio' in name:
                linear += num_params
            elif 'llama_model' in name:
                llama += num_params
            else:
                pass

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        if use_4bit:
            trainable_params /= 2
        print(
            f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
        )
        print(
            f'lora params: {lora:,d} || video params: {video:,d} || audio params: {audio:,d} || image params: {image:,d}')
        print(f'linear params: {linear:,d} || imagebind params: {imagebind:,d} || llama params: {llama:,d}')

    def load_parameters(self, path):
        if os.path.exists(os.path.join(path, 'pytorch_model.pt')):
            print('loading parameters from {}'.format(self.args['save_path']))
            delta_ckpt = torch.load(f'{path}/pytorch_model.pt', map_location=torch.device('cuda'))
            self.model.load_state_dict(delta_ckpt, strict=False)
