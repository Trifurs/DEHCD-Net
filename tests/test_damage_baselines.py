import gc
import math
import unittest
from pathlib import Path
import torch

from models import build_model
from utils.losses import segmentation_loss
from utils.prediction import predict_outputs, decode_predictions
from utils.config import XMLConfigParser
from utils.experiment_statistics import paired_seed_comparison, holm_adjust

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class DamageBaselineTests(unittest.TestCase):
    def test_full_checkpoint_evaluation_does_not_reload_encoder_file(self):
        cfg = {'model': {'name': 'changeos', 'num_classes': 4,
                         'encoder_checkpoint': '/nonexistent/encoder-checkpoint.pth'}}
        with self.assertRaises(FileNotFoundError):
            build_model(cfg, 3, 1)
        model = build_model(cfg, 3, 1, initialize_encoder=False)
        self.assertEqual('full_checkpoint_expected', model.implementation_info['initialization'])
        from utils.model_metadata import model_metadata
        self.assertEqual('fp32', model_metadata(model, 'cpu')['core_precision_policy'])

    def test_diagnostic_head_metrics_keep_foreground_conditioning_explicit(self):
        from utils.damage_metrics import DamageHeadMeter
        # Full-map accuracy is 3/4, while damage on the two true foreground
        # pixels is perfect; the separate reports must not conflate the two.
        labels = torch.tensor([[[0, 1], [2, 0]]])
        pred = torch.tensor([[[1, 1], [2, 0]]])
        logits = torch.nn.functional.one_hot(pred, 3).permute(0,3,1,2).float()*5
        loc = torch.nn.functional.one_hot((labels>0).long(),2).permute(0,3,1,2).float()*5
        meter = DamageHeadMeter(3, localization_supervised=False); meter.update({'logits':logits,'localization_logits':loc},labels)
        result = meter.compute()
        self.assertEqual(2, result['labeled_foreground_pixels'])
        self.assertEqual(1., result['conditional_damage_hmean_f1'])
        self.assertIsNone(result['localization_f1'])
        self.assertFalse(result['localization_supervised'])
        self.assertEqual(1., result['diagnostic_localization_f1'])

    def test_real_models_multiclass_primary_backward_and_dual_heads(self):
        for name in ('changeos', 'damageformer', 'changemamba'):
            with self.subTest(model=name):
                torch.manual_seed(17)
                cfg = {'model': {'name': name, 'num_classes': 4, 'selective_scan_backend': 'torch',
                                  'compare_adapt_batchnorm': True, 'group_norm_eps': .001}}
                model = build_model(cfg, 3, 4).train()
                optical, sar = torch.randn(2, 3, 64, 64), torch.randn(2, 4, 64, 64)
                out = model(optical, sar)
                self.assertEqual((2, 4, 64, 64), tuple(out['logits'].shape))
                self.assertEqual((2, 2, 64, 64), tuple(out['localization_logits'].shape))
                target = torch.randint(0, 4, (2, 64, 64)); target[:, :2] = 255
                out['logits'].retain_grad(); out['localization_logits'].retain_grad()
                loss = segmentation_loss(out, target, {'loss': 'ce_dice', 'localization_loss_weight': 0}, 4)
                loss.backward()
                self.assertIsNotNone(out['logits'].grad)
                self.assertIsNone(out['localization_logits'].grad)
                active = [(key, parameter) for key, parameter in model.named_parameters() if parameter.grad is not None]
                self.assertTrue(active)
                for key, parameter in active:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                self.assertTrue(torch.isfinite(norm))
                torch.optim.SGD(model.parameters(), lr=1e-4).step()
                model.eval()
                with torch.no_grad():
                    odd = model(optical[:1, :, :61, :59], sar[:1, :, :61, :59])
                    self.assertEqual((1, 4, 61, 59), tuple(odd['logits'].shape))
                del model, out, loss, odd
                gc.collect()

    def test_binary_and_five_class_adapted_dual_heads(self):
        for name in ('changeos', 'damageformer', 'changemamba'):
            for classes, sar_channels in ((2, 1), (5, 3)):
                with self.subTest(model=name, classes=classes):
                    cfg = {'model': {'name': name, 'num_classes': classes, 'selective_scan_backend': 'torch',
                        'compare_adapt_batchnorm': True, 'share_damage_encoder': classes == 5,
                        'changemamba_variant': 'bda' if classes == 5 else 'multimodal'}}
                    model = build_model(cfg, 3, sar_channels).train()
                    out = model(torch.randn(2,3,64,64), torch.randn(2,sar_channels,64,64))
                    self.assertEqual(classes, out['logits'].shape[1])
                    loss = segmentation_loss(out, torch.randint(0, classes, (2,64,64)), {'loss':'ce_dice', 'localization_loss_weight': 0}, classes)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    active = [p for p in model.parameters() if p.grad is not None]
                    self.assertTrue(active)
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in active))
                    del model, out, loss
                    gc.collect()

    def test_dual_head_tta_and_ignore_supervision(self):
        class Pointwise(torch.nn.Module):
            def forward(self, x, y):
                return {'logits': torch.cat([x, y], 1), 'localization_logits': torch.cat([-x, x], 1)}
        x, y = torch.randn(1,1,7,11), torch.randn(1,1,7,11)
        expected = Pointwise()(x,y)
        for mode in ('none','flips','d4'):
            actual = predict_outputs(Pointwise(),x,y,False,mode)
            for key in expected:
                torch.testing.assert_close(actual[key],expected[key])
        for recipe in ('ce_dice', 'hierarchical_change'):
            out = {'logits': torch.randn(2,4,8,8,requires_grad=True),
                   'localization_logits': torch.randn(2,2,8,8,requires_grad=True)}
            labels = torch.full((2,8,8),255)
            loss = segmentation_loss(out, labels, {'loss':recipe, 'localization_loss_weight': 0}, 4)
            self.assertEqual(0.,loss.item());loss.backward()
            self.assertIsNotNone(out['logits'].grad); self.assertFalse(out['logits'].grad.any())
            self.assertIsNone(out['localization_logits'].grad)
            # Ignored pixels receive zero gradient; the localization head stays unsupervised.
            for value in out.values(): value.grad = None
            labels[:,4:] = 0
            segmentation_loss(out,labels,{'loss':recipe, 'localization_loss_weight': 0},4).backward()
            self.assertTrue(torch.isfinite(out['logits'].grad).all())
            self.assertFalse(out['logits'].grad[:,:,:4].any())
            self.assertIsNone(out['localization_logits'].grad)

    def test_changeos_object_vote_requires_correct_taxonomy(self):
        loc = torch.zeros(1,2,6,6);loc[:,0]=1;loc[:,1,1:3,1:3]=3
        dam = torch.zeros(1,5,6,6);dam[:,3]=2
        output = {'logits': dam,'localization_logits':loc}
        pred = decode_predictions(output,{'inference':{'prediction_rule':'changeos_object'}})
        self.assertEqual(12,int(pred.sum()))
        self.assertEqual(0,int(pred[0,0,0]))
        output['logits'] = dam[:,:4]
        with self.assertRaises(ValueError):
            decode_predictions(output,{'inference':{'prediction_rule':'changeos_object'}})


class ScanAndControlTests(unittest.TestCase):
    def test_selective_scan_reference_recurrence_and_gradients(self):
        from compare.official.changemamba.scan_backend import selective_scan_reference as scan
        u = torch.ones(1,4,5,dtype=torch.double)
        delta = torch.ones_like(u)
        A = -torch.ones(4,1,dtype=torch.double)
        B = C = torch.ones(1,2,1,5,dtype=torch.double)
        actual = scan(u,delta,A,B,C)
        expected = torch.tensor([sum(math.exp(-k) for k in range(t+1)) for t in range(5)],dtype=torch.double)
        torch.testing.assert_close(actual, expected.expand(1,4,5))
        args = tuple(t.clone().requires_grad_() for t in (u[:,:,:3], delta[:,:,:3], A, B[:,:,:,:3], C[:,:,:,:3]))
        self.assertTrue(torch.autograd.gradcheck(scan,args,eps=1e-6,atol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; fused scan numerical check remains pending')
    def test_cuda_scan_matches_reference_forward_and_backward(self):
        from compare.official.changemamba import vmamba
        from compare.official.changemamba.scan_backend import selective_scan_reference as scan
        if not hasattr(vmamba,'selective_scan_cuda_oflex'):
            self.skipTest('selective_scan_cuda_oflex extension not installed')
        torch.manual_seed(2)
        # Exercise tiny, cross-warp and long scans, including incomplete chunks.
        for length in (9, 257, 1025):
            with self.subTest(length=length):
                initial = [torch.randn(2,8,length,device='cuda'),torch.randn(2,8,length,device='cuda'),
                           -torch.rand(8,1,device='cuda'),torch.randn(2,4,1,length,device='cuda'),
                           torch.randn(2,4,1,length,device='cuda'),torch.randn(8,device='cuda'),torch.randn(8,device='cuda')]
                a=[t.clone().requires_grad_() for t in initial]; b=[t.clone().requires_grad_() for t in initial]
                reference=scan(*a,True,1,1,True)
                fused=vmamba.SelectiveScanOflex.apply(*b,True,1,1,True)
                torch.testing.assert_close(reference,fused,atol=1e-4,rtol=1e-4)
                reference.sum().backward();fused.sum().backward()
                for x,y in zip(a,b): torch.testing.assert_close(x.grad,y.grad,atol=1e-3,rtol=1e-3)

    def test_all_added_capacity_controls_are_active(self):
        base = XMLConfigParser(ROOT/'configs/dehcd/haiti_s.xml').parse().as_dict()
        for key,value in (('hog_prior','intensity_control'),('bicsf_mode','conv_matched'),
                          ('gcb_mode','conv_matched'),('irb_mode','feedforward_matched')):
            with self.subTest(control=key):
                cfg={**base,'model':{**base['model'],key:value}}
                model=build_model(cfg,3,4)
                loss=model(torch.randn(2,3,32,32),torch.randn(2,4,32,32)).square().mean()
                loss.backward()
                for name,p in model.named_parameters():
                    self.assertIsNotNone(p.grad,name)
                    self.assertTrue(torch.isfinite(p.grad).all(),name)
                for m in model.modules():
                    if hasattr(m,'control_parameters'):
                        self.assertEqual(m.control_parameters,sum(p.numel() for p in m.parameters()))
                if key == 'irb_mode':
                    # Distinct retained scale parameters must not reduce to one
                    # scalar mean with identical gradients (fake capacity padding).
                    grad = model.diffusion_refine.step_scale.grad
                    self.assertGreater(float(grad.max() - grad.min()), 0.)

    def test_seed_statistics(self):
        result=paired_seed_comparison(dict.fromkeys((42,1051,2060),.8),dict.fromkeys((42,1051,2060),.7),repeats=1000)
        self.assertAlmostEqual(.1,result['mean_difference'])
        self.assertEqual(.25,result['p_value'])
        self.assertEqual([.03,.08,.08],holm_adjust([.01,.04,.04]))
        with self.assertRaises(ValueError): paired_seed_comparison({1:.8,2:.7},{1:.2,3:.2})



if __name__ == '__main__':
    unittest.main()
