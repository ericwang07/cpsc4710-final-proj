import math
import os

import foolbox as fb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from robustbench import load_cifar10


class AttackEvaluator:
    def __init__(self, device) -> None:
        self.test_dataset = None
        self.test_loader = None
        self.class_names = None
        self.device = device
        if self.device is None:
            raise ValueError("Device cannot be None")
        self.images = None
        self.labels = None

    def load_dataset(self, dataset_type):
        if dataset_type == "cifar10":
            torch.manual_seed(8)

            os.makedirs(os.path.join("data", "torchvision"), exist_ok=True)
            os.makedirs(os.path.join("results", "cifar10"), exist_ok=True)

            n_examples = 10000  # get all eval examples
            batch_size = 100
            self.images, self.labels = load_cifar10(n_examples=n_examples, data_dir="data")
            self.test_dataset = torch.utils.data.TensorDataset(self.images, self.labels)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=batch_size, shuffle=False
            )
            self.class_names = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
            print(len(self.test_dataset))
        elif dataset_type == "imagenet":
            pass
        else:
            raise ValueError("Incompatible dataset")

    def _per_class_accuracy(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        n_classes: int = 10,
        batch_size: int = 100,
    ):

        class_correct = torch.zeros(n_classes)
        class_total = torch.zeros(n_classes)

        n_batches = math.ceil(x.shape[0] / batch_size)
        with torch.no_grad():
            for counter in range(n_batches):
                x_curr = x[counter * batch_size : (counter + 1) * batch_size].to(
                    self.device
                )
                y_curr = y[counter * batch_size : (counter + 1) * batch_size].to(
                    self.device
                )

                output = model(x_curr)
                predictions = output.max(1)[1]

                for c in range(n_classes):
                    mask = y_curr == c
                    class_total[c] += mask.sum().item()
                    class_correct[c] += ((predictions == y_curr) & mask).sum().item()

        class_accuracies = class_correct / class_total
        overall_accuracy = class_correct.sum().item() / class_total.sum().item()

        return overall_accuracy, class_accuracies

    def eval_baseline(self, model, batch_size=100):
        acc_orig, per_class_orig = self._per_class_accuracy(model.to(self.device), self.images, self.labels, batch_size=batch_size)
        print(f"Overall Accuracy: {acc_orig:.4f}\n")
        print("Per-class Accuracy:")
        for i, (name, acc) in enumerate(zip(self.class_names, per_class_orig)):
                print(f"  {name:12s}: {acc:.4f}")

    def evaluate_robust_accuracy(
        self, fmodel, dataloader, rel_stepsize=0.01 / 0.3, steps=1, eps=8 / 255
    ):
        robust_success_list = []
        adv_examples_list = []
        l_inf_list = []

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            _, clipped, success = fb.attacks.LinfPGD(steps=steps)(
                fmodel, images, labels, epsilons=[eps]
            )
            clipped = clipped[0]

            robust_success_list.append(success.float().reshape(-1).cpu())

            perturb = (clipped - images).detach()
            l_inf = perturb.abs().amax(dim=(1, 2, 3)).cpu()
            l_inf_list.append(l_inf)

            # store adv imgs
            adv_examples_list.append(clipped.detach().cpu())

            # free GPU memory immediately
            del images, labels, perturb, clipped
            torch.cuda.empty_cache()

        # concatenate all batch results
        robust_success = torch.cat(robust_success_list)
        robust_accuracy = 1 - robust_success.mean()
        l_inf_vals = torch.cat(l_inf_list, dim=0)
        adv_examples = torch.cat(adv_examples_list, dim=0)
        return robust_accuracy, robust_success, l_inf_vals, adv_examples

    def show_original_and_adv(self, idx, original_dataset, adv_image, adv_label=None):
        """
        original_dataset: dataset, tensor [N,C,H,W]
        adv_image: tensor [C,H,W]
        adv_label: integer (adversarial predicted class)
        """

        # detach
        original = original_dataset[idx][0].detach().cpu()
        adv = adv_image.detach()

        # Convert CHW -> HWC for matplotlib
        orig_img = original.permute(1, 2, 0).numpy()
        adv_img = adv.permute(1, 2, 0).numpy()

        # Optional: clamp to valid range
        # orig_img = orig_img.clip(0, 1)
        # adv_img = adv_img.clip(0, 1)

        # Build captions
        orig_title = "Original"
        label = original_dataset[idx][1]
        if label is not None and self.class_names is not None:
            orig_title += f"\n (label: {self.class_names[label]})"

        adv_title = "Adversarial"
        if adv_label is not None and self.class_names is not None:
            adv_title += f"\n (pred: {self.class_names[adv_label]})"

        # Plot
        plt.figure(figsize=(4, 2))

        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.title(orig_title)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(adv_img)
        plt.title(adv_title)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def plot_orig_adv(self, model, idx, adv_examples):
        adv_image = adv_examples[idx]
        with torch.no_grad():
            pred_adv = model(adv_image.unsqueeze(0).to(self.device)).argmax().item()
        self.show_original_and_adv(idx, self.test_dataset, adv_image, pred_adv)
