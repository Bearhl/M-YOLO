import torch

class Engine(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):

        if name in self.hooks:
            self.hooks[name](state)

    def train(self, network, iterator, max_epoch, optimizer, scheduler, step_accumulate=1):
        state = {
            "network": network,
            "iterator": iterator,
            "max_epoch": max_epoch,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": 0,
            "t": 1,
            "train": True,
            "best_miou": 0,
            "scaler": None,
            "step_accumulate": step_accumulate
        }
        self.hook("on_start", state)
        while state["epoch"] < state["max_epoch"]:
            self.hook("on_start_epoch", state)
            for sample in state["iterator"]:
                state["sample"] = sample
                # self.hook("on_sample", state)

                def closure():
                    if state["scaler"] is not None:
                        with torch.cuda.amp.autocast():
                            loss, output = state["network"](state["sample"], state["epoch"])
                        state["scaler"].scale(loss).backward()
                    else:
                        loss, output = state["network"](state["sample"], state["epoch"])
                        loss.backward()
                    state["output"] = output
                    state["loss"] = loss
                    self.hook("on_forward", state)
                    # to free memory in save_for_backward
                    state["output"] = None
                    state["loss"] = None
                    return loss

                if step_accumulate == 1:
                    state["optimizer"].zero_grad()
                    if state["scaler"] is None:
                        state["optimizer"].step(closure)
                    else:
                        closure()
                        state["scaler"].step(state["optimizer"])
                        state["scaler"].update()

                else:
                    closure()
                    if state["t"] % step_accumulate == 0 or state["t"] % state["test_interval"] == 0:
                        if state["scaler"] is None:
                            state["optimizer"].step()
                            state["optimizer"].zero_grad()
                        else:
                            state["scaler"].step(state["optimizer"])
                            state["scaler"].update()
                            state["optimizer"].zero_grad()

                self.hook("on_update", state)
                state["t"] += 1
            state["epoch"] += 1
            self.hook("on_end_epoch", state)
        self.hook("on_end", state)
        return state

    @torch.no_grad()
    def test(self, network, iterator, split):
        state = {"network": network,
                 'iterator': iterator,
                 "split": split,
                 "t": 0,
                 "train": False
                 }

        self.hook("on_test_start", state)
        for sample in state["iterator"]:
            state["sample"] = sample
            # self.hook("on_test_sample", state)

            def closure():
                loss, output = state["network"](state["sample"])
                state["output"] = output
                state["loss"] = loss
                self.hook("on_test_forward", state)

                state["output"] = None
                state["loss"] = None

            closure()
            state["t"] += 1
        self.hook("on_test_end", state)
        return state
