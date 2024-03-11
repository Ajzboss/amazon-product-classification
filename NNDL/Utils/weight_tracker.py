class ActivationMonitor:
    def __init__(self, model):
        self.activations = []
        self.hooks = []

        # Register hooks on all layers
        self.register_hooks(model)

    def register_hooks(self, model):
        for layer in model.children():
            hook = layer.register_forward_hook(self.hook_fn)
            self.hooks.append(hook)

    def hook_fn(self, module, input, output):
        # You can customize this function to store or print the activations
        self.activations.append(output)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
