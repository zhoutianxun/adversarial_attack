# one pixel (OP) attack
import numpy as np
from scipy.optimize import dual_annealing

# pytorch
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
import torchvision.models as models

class OnePixelAttack:
    def __init__(self, target_model: torchvision.models, transform: torchvision.transforms, device=torch.device('cpu')):
        self.net = target_model.to(device)
        self.transform = transform
        self.device = device

    def perturb_image(self, xs, img):
        xs = xs.reshape(-1, 5)
        xs = xs.astype(int)
        adv_example = img.clone()
        for pixel in xs:
            adv_example[:, 0, pixel[0], pixel[1]] = pixel[2]/255.0
            adv_example[:, 1, pixel[0], pixel[1]] = pixel[3]/255.0
            adv_example[:, 2, pixel[0], pixel[1]] = pixel[4]/255.0
        adv_example = self.transform(adv_example)
        return adv_example

    def predict_classes(self, xs, img, target_calss, minimize=True):
        imgs_perturbed = self.perturb_image(xs, img.clone()).to(self.device)
        with torch.no_grad():
            predictions = F.softmax(self.net(imgs_perturbed), dim=1).data.cpu().numpy()[:, target_calss]

        return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, target_calss, targeted_attack=False, verbose=False):

        attack_image = self.perturb_image(x, img.clone()).to(self.device)
        with torch.no_grad():
            confidence = F.softmax(self.net(attack_image), dim=1).data.cpu().numpy()[0]
        predicted_class = np.argmax(confidence)

        if (verbose):
            print(f"Confidence: {confidence[target_calss]:.4f}")
        if (targeted_attack and predicted_class == target_calss) or (not targeted_attack and predicted_class != target_calss):
            return True


    def attack(self, img, label, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
        # img: 1*3*W*H tensor
        # label: a number

        targeted_attack = target is not None
        target_calss = target if targeted_attack else label
        bounds = [(0, img.shape[-2]-1), (0, img.shape[-1]-1), (0, 255), (0, 255), (0, 255)] * pixels

        popmul = int(max(1, popsize/len(bounds)))

        predict_fn = lambda xs: self.predict_classes(xs, img, target_calss, target is None)
        #callback_fn = lambda x, convergence: self.attack_success(x, img, target_calss, targeted_attack, verbose)
        callback_fn = lambda x, f, context: self.attack_success(x, img, target_calss, targeted_attack)
        '''
        inits = np.zeros([popmul*len(bounds), len(bounds)])
        for init in inits:
            for i in range(pixels):
                init[i * 5 + 0] = np.random.random() * (img.shape[-2]-1)
                init[i * 5 + 1] = np.random.random() * (img.shape[-1]-1)
                init[i * 5 + 2] = np.clip(np.random.normal(128, 127), 0, 255)
                init[i * 5 + 3] = np.clip(np.random.normal(128, 127), 0, 255)
                init[i * 5 + 4] = np.clip(np.random.normal(128, 127), 0, 255)
        '''    
        #attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        #                                       recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)
        attack_result = dual_annealing(predict_fn, bounds, maxiter=maxiter, callback=callback_fn, no_local_search=True)

        attack_image = self.perturb_image(attack_result.x, img)
        with torch.no_grad():
            predicted_probs = F.softmax(self.net(attack_image.to(self.device)), dim=1).data.cpu().numpy()[0]

        predicted_class = np.argmax(predicted_probs)

        if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_calss):
            return 1, attack_result.x.astype(int), attack_result.nfev, attack_image
        return 0, attack_result.x.astype(int), attack_result.nfev, attack_image